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
    "application/vnd.ollama.image.model": "model-{shorthash}.gguf",
    "application/vnd.ollama.image.template": "template-{shorthash}.txt",
    "application/vnd.ollama.image.license": "license-{shorthash}.txt",
    "application/vnd.ollama.image.params": "params-{shorthash}.json",
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


async def download_blob(
    client: httpx.AsyncClient,
    job: DownloadJob,
    *,
    progress: Progress,
):
    job.dest_path.parent.mkdir(parents=True, exist_ok=True)
    task = progress.add_task(
        f"{job.dest_path} ({format_size(job.size)})",
        total=job.size,
    )
    temp_path = job.dest_path.with_suffix(f".tmp-{time.time()}")
    try:
        if job.size < 1048576:
            resp = await client.get(job.blob_url, follow_redirects=True)
            resp.raise_for_status()
            temp_path.write_bytes(resp.content)
        else:
            async with client.stream(
                "GET",
                job.blob_url,
                follow_redirects=True,
            ) as resp:
                resp.raise_for_status()
                with temp_path.open("wb") as f:
                    async for chunk in resp.aiter_bytes(1048576):
                        f.write(chunk)
                        progress.update(task, completed=f.tell())
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
            log.warning("Unknown media type: %s", layer["mediaType"])
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
