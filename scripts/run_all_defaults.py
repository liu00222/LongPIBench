#!/usr/bin/env python3
"""Run all four clean/default LongPIBench conditions with one shared HF model."""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from longpibench.benchmark import parse_json_response, prepare_example, score_response

MODELS_MAP = {
    'qwen3-8b': ("Qwen/Qwen3-8B", "results/qwen3_8b_authority_spoof_defaults"),
    'llama31-8b': ("meta-llama/Llama-3.1-8B-Instruct", "results/llama31_8b_instruct_authority_spoof_defaults"),
    'llama32-3b': ("meta-llama/Llama-3.2-3B-Instruct", "results/llama32_3b_instruct_authority_spoof_defaults")
}

model = 'llama32-3b'

MODEL_NAME, OUTPUT_ROOT = MODELS_MAP[model]
OUTPUT_ROOT = Path(OUTPUT_ROOT)
SUITES = ("paper", "resume", "email", "code")
ATTACK = "authority_spoof"
GOALS = {"paper": "2", "resume": "2", "email": "0", "code": "2"}


@contextmanager
def memory_efficient_sdpa(query_chunk_size: int = 512):
    """Compute exact full-context attention in bounded query-row tiles."""
    original_sdpa = torch.nn.functional.scaled_dot_product_attention

    def tiled(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False,
              scale=None, enable_gqa=False):
        query_length = query.shape[-2]
        key_length = key.shape[-2]
        if query_length <= query_chunk_size:
            return original_sdpa(
                query, key, value, attn_mask, dropout_p, is_causal,
                scale=scale, enable_gqa=enable_gqa,
            )
        outputs = []
        for start in range(0, query_length, query_chunk_size):
            end = min(start + query_chunk_size, query_length)
            query_tile = query[..., start:end, :]
            tile_mask = attn_mask
            tile_is_causal = is_causal
            if attn_mask is not None and attn_mask.shape[-2] == query_length:
                tile_mask = attn_mask[..., start:end, :]
            elif is_causal:
                query_positions = torch.arange(start, end, device=query.device)[:, None]
                key_positions = torch.arange(key_length, device=query.device)[None, :]
                tile_mask = key_positions <= query_positions
                tile_is_causal = False
            outputs.append(
                original_sdpa(
                    query_tile, key, value, tile_mask, dropout_p, tile_is_causal,
                    scale=scale, enable_gqa=enable_gqa,
                )
            )
        return torch.cat(outputs, dim=-2)

    torch.nn.functional.scaled_dot_product_attention = tiled
    try:
        yield
    finally:
        torch.nn.functional.scaled_dot_product_attention = original_sdpa


class QwenModel:
    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
        capability = torch.cuda.get_device_capability(0)
        model_dtype = torch.bfloat16 if capability[0] >= 8 else torch.float16
        max_memory = None
        if torch.cuda.device_count() == 4:
            max_memory = {
                0: "4500MiB",
                1: "4500MiB",
                2: "4500MiB",
                3: "4500MiB",
                "cpu": "64GiB",
            }
        elif torch.cuda.device_count() == 2:
            # CUDA_VISIBLE_DEVICES maps the selected physical devices to 0 and 1.
            # These caps leave room for KV caches and the jobs already on each GPU.
            max_memory = {0: "5GiB", 1: "11GiB", "cpu": "64GiB"}
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            dtype=model_dtype,
            device_map="auto",
            max_memory=max_memory,
            low_cpu_mem_usage=True,
            local_files_only=True,
        ).eval()

    def query(self, system_prompt: str, data_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": data_prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer(text, return_tensors="pt")
        first_device = next(self.model.parameters()).device
        inputs = {key: value.to(first_device) for key, value in inputs.items()}
        with torch.inference_mode(), memory_efficient_sdpa():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=768,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = generated[0, inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def completed_records(path: Path) -> dict[int, dict]:
    records = {}
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
            records[int(record["item_id"])] = record
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return records


def run_suite(model: QwenModel, suite: str) -> dict:
    goal = GOALS[suite]
    output = OUTPUT_ROOT / f"{suite}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    records = completed_records(output)
    started = time.time()
    with output.open("a", encoding="utf-8") as handle:
        for item_id in range(100):
            if item_id in records:
                continue
            example = prepare_example(suite, item_id, ATTACK, goal)
            item_started = time.time()
            raw = model.query(example.system_prompt, example.data_prompt)
            record = {
                "suite": suite,
                "item_id": item_id,
                "attack": ATTACK,
                "goal": goal,
                "position": example.position,
                "model": MODEL_NAME,
                "raw_response": raw,
            }
            try:
                response = parse_json_response(raw)
                record.update(
                    response=response,
                    success=score_response(suite, goal, response),
                    error=None,
                )
            except (ValueError, json.JSONDecodeError) as exc:
                record.update(response=None, success=False, error=str(exc))
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            records[item_id] = record
            print(
                f"{suite} {item_id + 1}/100 success={record['success']} "
                f"error={record['error'] is not None} seconds={time.time() - item_started:.1f}",
                flush=True,
            )
    ordered = [records[item_id] for item_id in range(100)]
    summary = {
        "suite": suite,
        "model": MODEL_NAME,
        "attack": ATTACK,
        "goal": goal,
        "count": len(ordered),
        "success_rate": sum(bool(record["success"]) for record in ordered) / len(ordered),
        "parse_errors": sum(record["error"] is not None for record in ordered),
        "elapsed_seconds_this_invocation": time.time() - started,
    }
    print(json.dumps(summary, sort_keys=True), flush=True)
    return summary


def main() -> None:
    print(
        json.dumps(
            {
                "model": MODEL_NAME,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "cuda_device_count": torch.cuda.device_count(),
                "condition": {"attack": ATTACK, "goals": GOALS},
            },
            sort_keys=True,
        ),
        flush=True,
    )
    model = QwenModel()
    summaries = [run_suite(model, suite) for suite in SUITES]
    (OUTPUT_ROOT / "summary.json").write_text(
        json.dumps(summaries, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
