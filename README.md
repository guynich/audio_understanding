Python standalone version adapted from `google-gemini/gemma-cookbook` notebook
for Gemma 3n model.

- [1. Installation](#1-installation)
- [2. Run script](#2-run-script)
  - [Shopping buddy result](#shopping-buddy-result)
  - [Journal enhancer result](#journal-enhancer-result)
  - [Time took](#time-took)
- [References](#references)

## 1. Installation

Tested on Apple Silicon (MacBook Air M3 16GB) running macOS Sequoia 15.7

1. Get an access token from your HuggingFace account
Needed for model loading.  Paste it to this command.
```bash
HF_TOKEN=<your token>
```

2. Clone this repo.
```bash
git clone git@github.com:guynich/raudio_understanding.git
```

3. Install dependencies

Create a Python environment and activate it.
```bash
cd
python3 -m venv .venv_audio_understanding
source .venv_audio_understanding/bin/activate
```

Upgrade package install.
```bash
python3 -m pip install --upgrade pip
```

Install packages.  Includes `transformers` version supporting Gemma 3n
(>= 4.53) and `timm` for image handling.
```bash
cd audio_understanding
python3 -m pip install -r requirements.txt
```

## 2. Run script

```bash
cd
source .venv_audio_understanding/bin/activate
cd audio_understanding

python3 main.py --hf_token=${HF_TOKEN}
```

On macOS Apple Silicon (M3) memory allocation can fail with this error.
```console
RuntimeError: Invalid buffer size: 9.10 GiB
```
The wrapper script `run.sh` mitigates this failure.
```bash
cd
cd audio_understanding
chmod +x ./run.sh

run.sh --hf_token=${HF_TOKEN}
```

### Shopping buddy result
```console
Here's your shopping list:

* Milk (1)
* Apples (3)
* Tofu (1)
```
```console
Here's your updated shopping list:

* 1 milk
* 4 bananas
* 1 tofu
```
```console
Here's your updated shopping list:

* 1 milk
* 4 bananas
* 1 tofu
* 1 dozen eggs
```
> I saw `donuts` instead of `dozen eggs` in one run.

### Journal enhancer result
```console
Give me a concise overview of these audio.

The audio features a person reflecting on their day. They describe feeling refreshed in the morning, enjoying coffee, and spending time at the park. They also mention finishing the day with a good book and feeling grateful for simple moments. The speaker seems to have had a pleasant day and appreciates their old friends. They express contentment and relaxation at the end of the day.
```

> I also saw `The audio seems to contain a stream of random characters and words,`
> in another run.

### Time took
MacBook Air M3 (16GB)
```console
Took: 1629.95 seconds
```

> The huggingface cache has 11GB for `gemma-3n-E2B-it`.  Suspect out of physical
> memory:
> `Some parameters are on the meta device because they were offloaded to the disk.`
> Using `Activity Monitor` the Python 3.12 process is using ~5GB.

## References

* Google Gemini notebook: https://github.com/google-gemini/gemma-cookbook/blob/main/Gemma/%5BGemma_3n%5DAudio_understanding_with_HF.ipynb
