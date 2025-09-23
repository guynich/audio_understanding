Python standalone version adapted from `google-gemini/gemma-cookbook` notebook
for Gemma 3n model.

- [1. Installation](#1-installation)
- [2. Run script](#2-run-script)
  - [Shopping buddy](#shopping-buddy)
  - [Journal enhancer](#journal-enhancer)
- [References](#references)

## 1. Installation

Tested on macOS Sequoia x86_64.

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

Expected output from [notebook](https://github.com/google-gemini/gemma-cookbook/blob/main/Gemma/%5BGemma_3n%5DAudio_understanding_with_HF.ipynb).


### Shopping buddy
```bash
Here's your shopping list:

* Milk (1)
* Apples (3)
* Tofu (1)
```

```console
user:
itemize it into a shopping list.
shopping1.wav
--------------------------------------------------------------------------------
assistant:
Here's your shopping list:

* 1 milk
* 3 apples
* 1 tofu
--------------------------------------------------------------------------------
user:
shopping2.wav
--------------------------------------------------------------------------------
assistant:
Here's the updated shopping list:

* 1 milk
* 4 bananas
* 1 tofu
--------------------------------------------------------------------------------
user:
shopping3.wav
--------------------------------------------------------------------------------
assistant:
Here's the updated shopping list:

* 1 milk
* 4 bananas
* 1 tofu
* 1 dozen eggs
--------------------------------------------------------------------------------
```

### Journal enhancer
```console
The audio snippets describe a pleasant day. The speaker mentions feeling refreshed in the morning with sunshine and coffee. They enjoyed an afternoon walk and saw beautiful views. The day ended with a good book and reflection on simple moments. The speaker expresses gratitude for a good day and reflects on enjoyable moments. They also mention a great lunch with an old friend and a satisfying evening.
```

## References

* Google Gemini notebook: https://github.com/google-gemini/gemma-cookbook/blob/main/Gemma/%5BGemma_3n%5DAudio_understanding_with_HF.ipynb
