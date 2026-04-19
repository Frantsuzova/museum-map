from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from tqdm.auto import tqdm


def _batch_iter(items: list[Path], batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def _l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(p=2, dim=-1, keepdim=True) + eps)


class ClipImageEmbedder:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32") -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

    def transform(self, filepaths: list[Path], batch_size: int = 32) -> np.ndarray:
        all_embs: list[np.ndarray] = []
        total_batches = (len(filepaths) + batch_size - 1) // batch_size

        for batch_paths in tqdm(_batch_iter(filepaths, batch_size), total=total_batches, desc="CLIP embeddings"):
            images = [Image.open(path).convert("RGB") for path in batch_paths]
            inputs = self.processor(images=images, return_tensors="pt", padding=True).to(self.device)

            with torch.no_grad():
                outputs = self.model.vision_model(pixel_values=inputs["pixel_values"])
                pooled = outputs.pooler_output if outputs.pooler_output is not None else outputs.last_hidden_state[:, 0, :]
                feats = self.model.visual_projection(pooled)
                feats = _l2_normalize(feats)

            all_embs.append(feats.cpu().numpy())

        return np.vstack(all_embs)
