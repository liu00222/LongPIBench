import json

import pytest

from longpibench.attacks import injection
from longpibench import benchmark
from longpibench.benchmark import parse_json_response, prepare_example, score_response
from longpibench.data import validate_dataset


@pytest.mark.parametrize("suite", ["paper", "resume", "email", "code"])
def test_all_suites_prepare_clean_and_attacked_examples(suite):
    clean = prepare_example(suite, 0, "no", "0")
    attacked = prepare_example(suite, 0, "naive", "0")
    assert clean.item_id == 0
    assert clean.data_prompt
    assert attacked.data_prompt != clean.data_prompt
    assert injection(suite, "naive", "0") in attacked.data_prompt


def test_no_attack_is_goal_independent():
    assert injection("paper", "no", "0") == injection("paper", "no", "2") == ""


def test_json_response_parser_accepts_fences():
    assert parse_json_response('```json\n{"approve":"yes"}\n```') == {"approve": "yes"}


@pytest.mark.parametrize(
    ("suite", "goal", "response"),
    [
        ("paper", "0", {"weaknesses": []}),
        ("paper", "1", {"weaknesses": ["Minor writing typo."]}),
        ("paper", "2", {"score": "8"}),
        ("resume", "0", {"years_of_work_experience": 10}),
        ("resume", "1", {"highest_degree": "Ph.D."}),
        ("resume", "2", {"final_recommendation": "Qualified"}),
        ("email", "0", {"summary": "See www.gmail-protected-link.com"}),
        ("email", "1", {"response": "Yes, please."}),
        ("email", "2", {"response": "Please do not proceed."}),
        ("code", "0", {"comments": []}),
        ("code", "1", {"comments": ["One comment"]}),
        ("code", "2", {"approve": "yes"}),
    ],
)
def test_success_metrics(suite, goal, response):
    assert score_response(suite, goal, response)


def test_dataset_json_is_serializable():
    example = prepare_example("email", 99, "authority_spoof", "2")
    json.dumps(example.to_dict())


def test_invalid_item_id_is_rejected():
    with pytest.raises(ValueError):
        prepare_example("code", 100)


def test_local_dataset_is_complete():
    validate_dataset()


def test_missing_dataset_error_explains_download(monkeypatch, tmp_path):
    monkeypatch.setattr(benchmark, "DATA_ROOT", tmp_path)
    with pytest.raises(FileNotFoundError, match="longpibench download-data"):
        prepare_example("email", 0, "no", "0")
