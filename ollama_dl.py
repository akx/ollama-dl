import argparse
import asyncio
import dataclasses
import logging
import pathlib
import time
from urllib.parse import urljoin

import httpx
from rich.logging import RichHandler
from rich.progress import Progress

log = logging.getLogger("ollama-dl")

media_type_to_file_template = {
    "application/vnd.ollama.image.license": "license-{shorthash}.txt",
    "application/vnd.ollama.image.model": "model-{shorthash}.gguf",
    "application/vnd.ollama.image.params": "params-{shorthash}.json",
    "application/vnd.ollama.image.system": "system-{shorthash}.txt",
    "application/vnd.ollama.image.template": "template-{shorthash}.txt",
}


def get_short_hash(layer: dict) -> str:
    assert layer["digest"].startswith("sha256:")
    return layer["digest"].partition(":")[2][:12]


def format_size(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    if size < 1048576:
        return f"{size // 1024} KB"
    return f"{size // 1048576} MB"


@dataclasses.dataclass(frozen=True)
class DownloadJob:
    layer: dict
    dest_path: pathlib.Path
    blob_url: str
    size: int


async def _inner_download(
    client: httpx.AsyncClient,
    *,
    url: str,
    temp_path: pathlib.Path,
    size: int,
    progress: Progress,
    task_id,
) -> None:
    if size < 1048576:
        resp = await client.get(url, follow_redirects=True)
        resp.raise_for_status()
        temp_path.write_bytes(resp.content)
        return

    if temp_path.is_file():
        start_offset = temp_path.stat().st_size
        headers = {"Range": f"bytes={start_offset}-"}

    else:
        start_offset = 0
        headers = {}

    async with client.stream(
        "GET",
        url,
        headers=headers,
        follow_redirects=True,
    ) as resp:
        assert resp.status_code == (206 if start_offset else 200)
        resp.raise_for_status()
        with temp_path.open("ab") as f:
            async for chunk in resp.aiter_bytes(1048576):
                f.write(chunk)
                progress.update(task_id, completed=f.tell())


async def download_blob(
    client: httpx.AsyncClient,
    job: DownloadJob,
    *,
    progress: Progress,
    num_retries: int = 10,
):
    job.dest_path.parent.mkdir(parents=True, exist_ok=True)
    task_desc = f"{job.dest_path} ({format_size(job.size)})"
    task = progress.add_task(task_desc, total=job.size)
    temp_path = job.dest_path.with_suffix(f".tmp-{time.time()}")
    try:
        for attempt in range(1, num_retries + 1):
            if attempt != 1:
                progress.update(
                    task, description=f"{task_desc} (retry {attempt}/{num_retries})"
                )
            try:
                await _inner_download(
                    client,
                    url=job.blob_url,
                    temp_path=temp_path,
                    size=job.size,
                    progress=progress,
                    task_id=task,
                )
            except httpx.TransportError as exc:
                log.warning(
                    "%s: Attempt %d/%d failed: %s",
                    job.blob_url,
                    attempt,
                    num_retries,
                    exc,
                )
                if attempt == num_retries:
                    raise
            else:
                break
        assert temp_path.stat().st_size == job.size
        temp_path.rename(job.dest_path)
        progress.update(task, completed=job.size)
    finally:
        if temp_path.is_file():
            temp_path.unlink()


async def get_download_jobs_for_image(
    *,
    client: httpx.AsyncClient,
    registry: str,
    dest_dir: str,
    name: str,
    version: str,
):
    manifest_url = urljoin(registry, f"v2/{name}/manifests/{version}")
    resp = await client.get(manifest_url)
    resp.raise_for_status()
    manifest_data = resp.json()
    assert (
        manifest_data["mediaType"]
        == "application/vnd.docker.distribution.manifest.v2+json"
    )
    for layer in sorted(manifest_data["layers"], key=lambda x: x["size"]):
        file_template = media_type_to_file_template.get(layer["mediaType"])
        if not file_template:
            log.warning(
                "Ignoring layer with unknown media type: %s",
                layer["mediaType"],
            )
            continue
        filename = file_template.format(shorthash=get_short_hash(layer))
        dest_path = pathlib.Path(dest_dir) / filename
        yield DownloadJob(
            layer=layer,
            dest_path=dest_path,
            blob_url=urljoin(registry, f"v2/{name}/blobs/{layer['digest']}"),
            size=layer["size"],
        )


async def download(*, registry: str, name: str, version: str, dest_dir: str):
    with Progress() as progress:
        async with httpx.AsyncClient() as client:
            tasks = []
            async for job in get_download_jobs_for_image(
                client=client,
                registry=registry,
                dest_dir=dest_dir,
                name=name,
                version=version,
            ):
                if job.dest_path.is_file():
                    log.info("Already have %s", job.dest_path)
                    continue
                tasks.append(download_blob(client, job, progress=progress))
            if tasks:
                await asyncio.gather(*tasks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--registry", default="https://registry.ollama.ai/")
    ap.add_argument("-d", "--dest-dir", default=None)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(
        format="%(message)s",
        level=(logging.DEBUG if args.verbose else logging.INFO),
        handlers=[RichHandler(show_path=False)],
    )
    if not args.verbose:
        logging.getLogger("httpx").setLevel(logging.WARNING)
    name = args.name
    if "/" not in name:
        name = f"library/{name}"
    dest_dir = args.dest_dir
    if not dest_dir:
        dest_dir = name.replace("/", "-").replace(":", "-")
    log.info("Downloading to: %s", dest_dir)
    if ":" not in name:
        name += ":latest"
    name, _, version = name.rpartition(":")
    asyncio.run(
        download(
            registry=args.registry,
            name=name,
            dest_dir=dest_dir,
            version=version,
        )
    )


if __name__ == "__main__":
    main()
