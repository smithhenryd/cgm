import torch
import torchvision as tv

from openai import OpenAI

import os
from pathlib import Path
import base64, json

from typing import List, Union
import random
import string

client = OpenAI()

def make_schema(animals: List[str]):
    """(Optional) Keep this around if you later switch to Structured Outputs."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "animal_label",
            "schema": {
                "type": "object",
                "properties": {
                    "label": {"type": "string", "enum": animals + ["None"]},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": ["label", "confidence"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }


def classify_image(
    path: Union[Path, str],
    animals: List[str],
    model: str = "gpt-5-mini",
    detail: str = "low",
):
    # Encode the image
    b64 = base64.b64encode(open(path, "rb").read()).decode("utf-8")

    # Use Chat Completions JSON mode (works across SDKs)
    resp = client.chat.completions.create(
        model=model,
        response_format={
            "type": "json_object"
        },  # <-- JSON mode instead of Responses.response_format
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You are labeling animal photos. "
                            'Return JSON only: {"label": <one of the options>, "confidence": <0..1>}.\n'
                            f"Choose exactly one from: {', '.join(animals)} or None."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64}",
                            "detail": detail,
                        },
                    },
                ],
            }
        ],
    )

    # Parse and defensively validate
    data = json.loads(resp.choices[0].message.content)
    allowed = set(animals + ["None"])
    label = data.get("label", "None")
    conf = float(data.get("confidence", 0.0))
    if label not in allowed:
        label = "None"
    conf = max(0.0, min(1.0, conf))
    return {"label": label, "confidence": conf}


def save_image(x: torch.Tensor, output_dir: Union[Path, str]) -> str:
    tmp = "".join(random.choices(string.ascii_letters + string.digits, k=12))
    img_path = os.path.join(output_dir, tmp + ".png")
    tv.utils.save_image(x, img_path, normalize=True)
    return img_path