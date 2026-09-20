"""Evaluation loops shared by the command-line interface."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .benchmark import parse_json_response, prepare_example, score_response


def item_ids(start: int, count: int) -> Iterable[int]:
    if start < 0 or count < 1 or start + count > 100:
        raise ValueError("the requested [start, start + count) range must fit within 0..100")
    return range(start, start + count)


def run_model(
    model,
    *,
    suite: str,
    attack: str,
    goal: str,
    position: str,
    start: int,
    count: int,
    output: Path,
) -> float:
    output.parent.mkdir(parents=True, exist_ok=True)
    successes = []
    with output.open("w", encoding="utf-8") as handle:
        for item_id in item_ids(start, count):
            example = prepare_example(suite, item_id, attack, goal, position)
            raw = model.query(example.system_prompt, example.data_prompt)
            record = example.to_dict()
            record.pop("system_prompt")
            record.pop("data_prompt")
            record["raw_response"] = raw
            try:
                response = parse_json_response(raw)
                success = score_response(suite, goal, response)
                record.update(response=response, success=success, error=None)
                successes.append(int(success))
            except (ValueError, json.JSONDecodeError) as exc:
                record.update(response=None, success=False, error=str(exc))
                successes.append(0)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
    return sum(successes) / len(successes)


def run_detector(
    detector,
    *,
    suite: str,
    attack: str,
    goal: str,
    position: str,
    start: int,
    count: int,
    output: Path,
) -> float:
    output.parent.mkdir(parents=True, exist_ok=True)
    label = int(attack != "no")
    errors = []
    with output.open("w", encoding="utf-8") as handle:
        for item_id in item_ids(start, count):
            example = prepare_example(suite, item_id, attack, goal, position)
            prediction = detector.predict(example.data_prompt)
            errors.append(int(prediction != label))
            handle.write(
                json.dumps(
                    {
                        "suite": suite,
                        "item_id": item_id,
                        "attack": attack,
                        "goal": goal,
                        "position": example.position,
                        "prediction": prediction,
                        "label": label,
                    }
                )
                + "\n"
            )
            handle.flush()
    return sum(errors) / len(errors)
