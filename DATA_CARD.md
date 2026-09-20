# LongPIBench synthetic data card

## Scope

This release contains 400 synthetic examples across four review suites. It does
not contain the benchmark's real-world evaluation data. IDs are integers from 0
through 99 in every suite. The data is hosted in the public
[`RainWatcher/LongPIBench`](https://huggingface.co/datasets/RainWatcher/LongPIBench)
Hugging Face dataset repository rather than duplicated in the GitHub code
repository.

Download the complete snapshot before running an experiment:

```bash
longpibench download-data
```

This stores all four suites under `longpibench/datasets/`. Runtime access is
local and does not download examples individually.

## Schemas

- `longpibench/datasets/papers/<id>/` contains a LaTeX paper project. The benchmark reads
  `abstract.tex`, `intro.tex`, `rw.tex`, `method.tex`, `eval.tex`, `dl.tex`, and
  `conclusion.tex` in that order.
- `longpibench/datasets/person_info/<id>.json` contains a synthetic name, work experience,
  education, publications, and skills.
- `longpibench/datasets/emails/<id>.json` contains `email_out`, `email_in`, and
  `attachment_content` strings.
- `longpibench/datasets/code_changes/<id>.json` contains `language`, `task_type`, `task_description`,
  `before_code`, and `after_code` strings.

## Intended use

The data is intended for controlled research on indirect prompt injection in
long-context LLM tasks, including attack measurement, detector evaluation, and
defense research. It is not a security certification suite and should not be
used as the sole evidence that a deployed system is safe.

## Limitations

- All released examples are synthetic and may not represent the distribution or
  complexity of production data.
- Attack-success metrics are goal-specific string/structure checks; they do not
  capture every harmful model behavior.
- Model serving stacks, chat templates, quantization, and generation settings
  can materially change results and should be reported with scores.
- Detection aliases may refer to gated or externally hosted checkpoints and may
  require separate terms or access approval from their publishers.
- This streamlined release builds deterministic text inputs from the public
  source data. It does not distribute generated PDF artifacts.

## Privacy

The released collections were generated for benchmarking. They are not the
private real-world suite. If an apparent real person's information is found,
please report it through the repository's issue tracker without reposting the
content.
