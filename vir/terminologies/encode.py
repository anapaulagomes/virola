import os

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from notebooks.helpers import get_or_download_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128
model_name = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
_tokenizer = None
_model = None


def _load_model_and_tokenizer():
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        torch.set_num_threads(os.cpu_count() or 4)
        model_path = get_or_download_model(model_name)
        _tokenizer = AutoTokenizer.from_pretrained(model_path)
        _model = AutoModel.from_pretrained(model_path)
        _model.eval()
    return _tokenizer, _model


@torch.inference_mode()
def encode_text(text):
    """Given a text, returns L2 embedding normalized (float32) from CLS."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Text is missing")

    tokenizer, model = _load_model_and_tokenizer()
    toks = tokenizer(
        [text],
        padding="max_length",
        max_length=25,  # TODO check this
        truncation=True,
        return_tensors="pt",
    )
    out = model(**toks).last_hidden_state  # [1, T, H]
    cls = out[:, 0, :].to("cpu").numpy().astype(np.float32, copy=False)  # [1, H]

    # L2 normalize
    norm = np.linalg.norm(cls, axis=1, keepdims=True) + 1e-12
    cls = cls / norm
    return cls[0]  # vector 1D (H,)
