# ollama-dl

Downloads models from the [Ollama](https://ollama.com/library) library.

## Example usage

If you have [`uv`](https://github.com/astral-sh/uv) installed, you can simply run:

```
uv run ollama_dl.py llama3.2
```

to have it install the dependencies required and run the script to download the default `latest`
version of [Meta Llama 3.2](https://ollama.com/library/llama3.2).

(If you don't have `uv`, set up a Python virtualenv, `pip install -e .` and run with `ollama-dl`.)

Once a model has been downloaded, you can use with e.g. [llama.cpp](https://github.com/ggerganov/llama.cpp/):

```
$ ollama-dl llama3.2:1b
[09/26/24 14:32:04] INFO     Downloading to: library-llama3.2-1b
library-llama3.2-1b/template-966de95ca8a6.txt (1 KB)  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00
library-llama3.2-1b/license-a70ff7e570d9.txt (5 KB)   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00
library-llama3.2-1b/license-fcc5a6bec9da.txt (7 KB)   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00
library-llama3.2-1b/model-74701a8c35f6.gguf (1259 MB) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00

$ ../llama.cpp/llama-cli -m library-llama3.2-1b/model-74701a8c35f6.gguf -p "We're no strangers to love"
build: 3828 (95bc82fb) with Apple clang version 16.0.0 (clang-1600.0.26.3) for arm64-apple-darwin23.6.0
[...]

We're no strangers to love, but we're no strangers to the chaos that comes with it. And, quite frankly, we're rather fond of a good argument.
^C
```
