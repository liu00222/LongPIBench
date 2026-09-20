# LongPIBench

LongPIBench is a benchmark for prompt-injection attacks and defenses in four
long-context review tasks: paper review, resume screening, email review, and
code review. The benchmark code is hosted on GitHub, while the synthetic data is
distributed separately through the
[LongPIBench dataset on Hugging Face](https://huggingface.co/datasets/RainWatcher/LongPIBench).

> [!CAUTION]
> The examples intentionally contain adversarial instructions. Treat generated
> prompts and model outputs as untrusted data, and do not connect benchmark runs
> to tools or accounts with real-world side effects.

## Dataset

Each suite has 100 synthetic examples:

| Suite | Public dataset | Injected location |
|---|---|---|
| Paper | `papers/` | selected LaTeX section (default: conclusion) |
| Resume | `person_info/` | end of the formatted resume |
| Email | `emails/` | end of the attachment |
| Code | `code_changes/` | comment at the end of the changed code |

The four directories are not stored in this Git repository. The download step
below installs the complete Hugging Face snapshot under
`longpibench/datasets/`, preserving the layout shown above. Benchmark runs then
read only local files; they do not fetch individual examples over the network.

Generated PDFs, LaTeX build products, intermediate generation files,
experiment logs, cached outputs, credentials, and private real-world data are
not included. The package prepares paper and resume inputs deterministically
from their released structured sources, so running the benchmark does not
mutate the datasets.

## Install

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Download and validate all four dataset suites in one operation:

```bash
longpibench download-data
```

By default this downloads the latest `main` revision from
`RainWatcher/LongPIBench` into `longpibench/datasets/`. To make a run exactly
reproducible, pin a Hugging Face commit SHA:

```bash
longpibench download-data --revision YOUR_FULL_COMMIT_SHA
```

Re-running the command updates changed files without downloading unchanged
files again. The generated data and Hugging Face download metadata are ignored
by Git.

Install one backend as needed:

```bash
pip install -e '.[openai]'
pip install -e '.[hf]'
pip install -e '.[detection]'
```

API credentials are read by the provider SDK from the environment. For the
OpenAI backend, set `OPENAI_API_KEY`; never place credentials in this repository.

## Inspect and run

Inspect a prepared attacked example without loading a model:

```bash
longpibench inspect --suite email --item-id 0 --attack naive --goal 0
```

Run five examples through an API model:

```bash
longpibench run --suite code --attack authority_spoof --goal 2 \
  --backend openai --model YOUR_MODEL_NAME --count 5 \
  --output results/code_authority_spoof_2.jsonl
```

Run a local Hugging Face model:

```bash
longpibench run --suite paper --attack naive --goal 2 \
  --backend hf --model Qwen/Qwen3-8B --count 100 \
  --output results/paper_naive_2.jsonl
```

Run the SecAlign prevention baseline (the gated Llama base model requires
Hugging Face access):

```bash
longpibench run --suite resume --attack combine --goal 2 \
  --backend secalign --count 100 --output results/resume_secalign.jsonl
```

Evaluate a detection baseline:

```bash
longpibench detect --suite email --attack authority_spoof --goal 0 \
  --detector prompt-guard --output results/email_prompt_guard.jsonl
```

Named detector aliases are `prompt-guard`, `deberta`, and `distilbert`. Any
compatible Hugging Face sequence-classification model ID can also be supplied.
Long inputs are divided into token chunks and an example is marked malicious
when any chunk exceeds the selected threshold.

## Conditions and outputs

Attacks are `no`, `naive`, `combine`, and `authority_spoof`; goals are `0`, `1`,
and `2`. The default attack is `authority_spoof`. Default goals are paper `2`
(high score), resume `2` (shortlist), email `0` (attacker-controlled link), and
code `2` (approve). Defaults also reproduce the published injection locations. A result JSONL contains
the condition, item ID, raw response, parsed response, success flag, and parsing
error (if any). For attacked conditions the CLI reports attack success rate
(ASR). Detection reports false-positive rate for clean data and false-negative
rate for attacked data.

The `no` condition ignores the goal when constructing an example; use goal `0`
for clean baselines. Run each attacked condition separately to retain an
auditable result file.

## Development checks

```bash
pip install -e '.[dev]'
longpibench download-data
pytest
python -m compileall -q longpibench
```

See [DATA_CARD.md](DATA_CARD.md) for schemas and limitations.

## License and citation

Code and released synthetic data are provided under the MIT license. Please
cite the LongPIBench paper when using this benchmark:

```
@inproceedings{liu2026longpibench,
  title={LongPIBench: A Long-Context Benchmark for Prompt Injection},
  author={Liu, Yupei and Jia, Yuqi and Gong, Neil Zhenqiang and Jia, Jinyuan},
  booktitle={EMNLP Findings},
  year={2026}
}
```
