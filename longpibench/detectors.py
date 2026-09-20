"""Reproducible Hugging Face detection baselines with explicit long-text chunking."""

from __future__ import annotations

ALIASES = {
    "prompt-guard": "meta-llama/Prompt-Guard-86M",
    "deberta": "ProtectAI/deberta-v3-base-prompt-injection-v2",
    "distilbert": "fmops/distilbert-prompt-injection",
}


class TransformerDetector:
    def __init__(self, model_name: str, threshold: float = 0.5, chunk_tokens: int = 512):
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("install detection dependencies: pip install -e '.[detection]'") from exc
        self.torch = torch
        self.threshold = threshold
        self.chunk_tokens = chunk_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()

    def predict(self, text: str) -> int:
        encoded = self.tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"]
        chunks = [encoded[i : i + self.chunk_tokens] for i in range(0, len(encoded), self.chunk_tokens)]
        with self.torch.inference_mode():
            for chunk in chunks:
                tensor = self.torch.tensor([chunk], device=self.device)
                logits = self.model(input_ids=tensor).logits[0]
                probabilities = self.torch.softmax(logits, dim=-1)
                id2label = self.model.config.id2label
                malicious = max(
                    (
                        probabilities[index].item()
                        for index, label in id2label.items()
                        if str(label).upper() not in {"BENIGN", "SAFE", "LABEL_0"}
                    ),
                    default=probabilities[-1].item(),
                )
                if malicious >= self.threshold:
                    return 1
        return 0


def create_detector(name: str, threshold: float, chunk_tokens: int) -> TransformerDetector:
    return TransformerDetector(ALIASES.get(name, name), threshold, chunk_tokens)
