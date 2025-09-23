"""
Example script to run Gemma 3n instruction tuned model from HuggingFace.

Memory allocation is fragile on Apple Silicon with 16GB RAM.

Application tasks include audio understanding.
"""

# @title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import sys
import time

import torch
from huggingface_hub import login
from transformers import AutoModelForImageTextToText, AutoProcessor

GEMMA_PATH = "google/gemma-3n-E2B-it"  # @param ["google/gemma-3n-E2B-it", "google/gemma-3n-E4B-it"]
RESOURCE_URL_PREFIX = "https://raw.githubusercontent.com/google-gemini/gemma-cookbook/refs/heads/main/Demos/sample-data/"

if sys.platform == "darwin":
    print("MPS available:", torch.backends.mps.is_available())
    print("MPS built:", torch.backends.mps.is_built())


class ChatState:
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        self.history = []

    def send_message(self, message, max_tokens=256):
        self.history.append(message)

        input_ids = self.processor.apply_chat_template(
            self.history,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        input_len = input_ids["input_ids"].shape[-1]

        input_ids = input_ids.to(self.model.device, dtype=self.model.dtype)
        outputs = self.model.generate(
            **input_ids, max_new_tokens=max_tokens, disable_compile=True
        )
        text = self.processor.batch_decode(
            outputs[:, input_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        self.history.append(
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": text[0]},
                ],
            }
        )

        # display chat
        for item in message["content"]:
            if item["type"] == "text":
                print(f"user:\n{item['text']}")
            elif item["type"] == "audio":
                print(f"user audio: {item['audio'].rsplit('/', 1)[-1]}")
            elif item["type"] == "image":
                print(f"user video: {item['image'].rsplit('/', 1)[-1]}")

        print(f"\nassistant:\n{text[0]}\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device_map", default="auto", choices=["auto", "cpu", "cuda", "mps"]
    )
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32"])
    parser.add_argument("--hf_token", required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    login(token=args.hf_token)

    processor = AutoProcessor.from_pretrained(GEMMA_PATH)
    model = AutoModelForImageTextToText.from_pretrained(
        GEMMA_PATH,
        dtype=args.dtype,
        device_map=args.device_map,
    )

    print(f"Device: {model.device}")
    print(f"DType: {model.dtype}")

    chat = ChatState(model, processor)

    start_time = time.time()

    # Audio-in: Shopping Buddy.  Shopping list example from three audio clips.
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "itemize it into a shopping list."},
                {"type": "audio", "audio": f"{RESOURCE_URL_PREFIX}shopping1.wav"},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": f"{RESOURCE_URL_PREFIX}shopping2.wav"},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": f"{RESOURCE_URL_PREFIX}shopping3.wav"},
            ],
        },
    ]

    chat.history = []
    print(f"{'-' * 40}")
    chat.send_message(messages[0])
    chat.send_message(messages[1])
    chat.send_message(messages[2])

    # Audio-in: Journal Enhancer
    prompt = {
        "role": "user",
        "content": [
            {"type": "audio", "audio": f"{RESOURCE_URL_PREFIX}journal1.wav"},
            {"type": "audio", "audio": f"{RESOURCE_URL_PREFIX}journal2.wav"},
            {"type": "audio", "audio": f"{RESOURCE_URL_PREFIX}journal3.wav"},
            {"type": "audio", "audio": f"{RESOURCE_URL_PREFIX}journal4.wav"},
            {"type": "audio", "audio": f"{RESOURCE_URL_PREFIX}journal5.wav"},
            {"type": "text", "text": "Give me a concise overview of these audio."},
        ],
    }

    chat.history = []
    print(f"{'-' * 40}")
    chat.send_message(prompt)

    # TODO: speech recognition and speech translation examples.
    chat.history = []

    print(f"Took: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
