"""Thin wrapper around the fine-tuned encoders (RoBERTa / ModernBERT)."""
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from cues import mask_cues, normalize


class Encoder:
    def __init__(self, name: str, path: str, nk_id: int | None = None,
                 batch_size: int = 32, max_length: int = 512):
        self.name = name
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path, torch_dtype=torch.bfloat16, device_map="mps").eval()
        self.device = "mps"
        self.model.to(self.device)
        self.batch_size = batch_size
        self.max_length = max_length
        self.nk_id = nk_id if nk_id is not None else self._find_nk_id()

    def _find_nk_id(self) -> int:
        """Find the NK class index from the model config; fall back to 1."""
        for idx, label in self.model.config.id2label.items():
            if label.replace("-", "_").upper() in {"NK", "NON_KNOWLEDGE", "NONKNOWLEDGE"}:
                return int(idx)
        print(f"[{self.name}] could not find NK in id2label "
              f"{self.model.config.id2label}; assuming NK = 1. Use --nk-id to override.")
        return 1

    @torch.no_grad()
    def predict(self, texts, mask: bool = False):
        texts = [normalize(t) for t in texts]
        if mask:  # <mask> for RoBERTa, [MASK] for ModernBERT
            texts = [mask_cues(t, self.tokenizer.mask_token) for t in texts]
        preds = []
        for i in range(0, len(texts), self.batch_size):
            batch = self.tokenizer(texts[i:i + self.batch_size], padding=True,
                                   truncation=True, max_length=self.max_length,
                                   return_tensors="pt").to(self.device)
            logits = self.model(**batch).logits
            preds += (logits.argmax(-1) == self.nk_id).int().tolist()
        return preds
