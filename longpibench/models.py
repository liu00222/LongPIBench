"""Small, dependency-lazy model adapters used by the public CLI."""

from __future__ import annotations

from typing import Protocol


class Model(Protocol):
    def query(self, system_prompt: str, data_prompt: str) -> str: ...


class OpenAIModel:
    """OpenAI Responses API adapter. Authentication comes from OPENAI_API_KEY."""

    def __init__(self, model_name: str, max_output_tokens: int = 4096):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("install the OpenAI extra: pip install -e '.[openai]'") from exc
        self.client = OpenAI()
        self.model_name = model_name
        self.max_output_tokens = max_output_tokens

    def query(self, system_prompt: str, data_prompt: str) -> str:
        response = self.client.responses.create(
            model=self.model_name,
            instructions=system_prompt,
            input=data_prompt,
            max_output_tokens=self.max_output_tokens,
        )
        return response.output_text


class HuggingFaceModel:
    """Text-only Transformers chat adapter for local causal language models."""

    def __init__(self, model_name: str, max_output_tokens: int = 4096):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("install the HF extra: pip install -e '.[hf]'") from exc
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto"
        )
        self.max_output_tokens = max_output_tokens

    def query(self, system_prompt: str, data_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": data_prompt},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_output_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = output[0, inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()


class SecAlignModel:
    """Meta-SecAlign LoRA adapter over Llama 3.1 8B."""

    def __init__(self, max_output_tokens: int = 4096):
        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("install the HF extra: pip install -e '.[hf]'") from exc
        self.torch = torch
        adapter = "facebook/Meta-SecAlign-8B"
        self.tokenizer = AutoTokenizer.from_pretrained(adapter, use_fast=False)
        base = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Meta-Llama-3.1-8B-Instruct", torch_dtype="auto", device_map="auto"
        )
        self.model = PeftModel.from_pretrained(base, adapter).eval()
        self.max_output_tokens = max_output_tokens

    def query(self, system_prompt: str, data_prompt: str) -> str:
        conversation = [
            {"role": "user", "content": system_prompt},
            {"role": "input", "content": data_prompt},
        ]
        prompt = self.tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_output_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = output[0, inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()


def create_model(backend: str, model_name: str | None, max_output_tokens: int) -> Model:
    if backend == "openai":
        if not model_name:
            raise ValueError("--model is required for the openai backend")
        return OpenAIModel(model_name, max_output_tokens)
    if backend == "hf":
        if not model_name:
            raise ValueError("--model is required for the hf backend")
        return HuggingFaceModel(model_name, max_output_tokens)
    if backend == "secalign":
        return SecAlignModel(max_output_tokens)
    raise ValueError(f"unknown backend: {backend}")
