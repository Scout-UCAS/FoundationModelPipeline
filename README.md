# Foundation Model Ops

Primary language: English. For the Chinese version, see [README.zh-CN.md](README.zh-CN.md).

Foundation Model Ops is a runnable control-plane framework for next-generation foundation model programs. It turns a broad research and engineering plan into validated configuration, executable pipelines, production adapter plans, reference model implementations, evaluation reports, deployment checks, and audit-ready artifacts.

The framework is intentionally lightweight: the core uses the Python standard library, while the architecture reference implementations use PyTorch as an optional dependency. The implementation now includes both local dry-run orchestration and production integration adapters for data lakes, distributed schedulers, model conversion, evaluation harnesses, monitoring, release gates, and serving validation.

## Overall Workflow

![Foundation Model Ops overall workflow](docs/assets/fmops_overall_flow.svg)

The diagram shows the full research-to-production loop: governed data ingestion, quality processing, staged mixture construction, architecture comparison, 400-GPU training, evaluation gates, checkpoint conversion, deployment validation, monitoring, and feedback into the next data/model iteration.

## What This Framework Covers

1. Large-scale training data system
   - Integrates open-source, newly collected, and business/internal data.
   - Models data cleaning, deduplication, clustering, quality assessment, lineage, and staged mixture design.
   - Covers pure text, multimodal image-text, video pretraining, and VLA data.
   - The default manifest targets 2500T+ scale and 20+ languages.

2. Architecture research and fair comparison
   - Covers MoE, Sparse / Linear Attention, RNN-like Backbone, SSM / Selective Scan, Retention / RetNet, Long Convolution, MLA / KV-Compressed Attention, Hybrid Architecture, MTP, Latent Reasoning, dLLM, Memory-augmented LLM, Mixture-of-Depths, Test-Time Memory, Token-free Byte-level LLM, Omni-modal Architecture, VLA / Robotics Transformer, JEPA / Latent World Model, Neuromorphic / Spiking Backbone, and Reasoning-native Architecture.
   - Provides runnable PyTorch reference implementations for all twenty families.
   - Keeps experiment setup, hardware budget, metrics, and ranking logic in one place.

3. Four-hundred-GPU training pipeline
   - Models Pre-training, SFT, RL, and Agentic RL.
   - Includes data handoff, distributed launch planning, stability monitoring, checkpoint conversion, and deployment validation.
   - The default hardware target is 50 nodes x 8 H100 GPUs = 400 GPUs.

4. Comprehensive evaluation system
   - Covers general capability, reasoning, multimodal understanding, VLA, long context, tool calling, agent capability, and edge deployment efficiency.
   - Produces JSON reports and normalized evaluation weights for release decisions.

5. Mature framework kernel
   - Includes schema validation, registries, dataset catalog, dry-run runners, production adapters, checkpoint manifests, deployment checks, run tracking, plugins, dashboard generation, tests, CI, and Makefile targets.

## Repository Layout

```text
.
+-- configs/
|   +-- architecture_experiments.json
|   +-- benchmark_catalog.json
|   +-- data_manifest.json
|   +-- datasets_catalog.json
|   +-- evaluation_suite.json
|   +-- production_integration.json
|   +-- training_pipeline.json
+-- jobs/
|   +-- checkpoint_convert.py
|   +-- data_ingest.py
|   +-- data_mixture.py
|   +-- data_quality.py
|   +-- deployment_validate.py
|   +-- evaluation_launch.py
|   +-- monitoring_export.py
|   +-- release_gate.py
|   +-- training_launch.py
+-- src/fmops/
|   +-- architecture_impl.py
|   +-- architectures.py
|   +-- benchmark_catalog.py
|   +-- checkpoint.py
|   +-- cli.py
|   +-- dashboard.py
|   +-- data.py
|   +-- data_pipeline.py
|   +-- dataset_catalog.py
|   +-- deployment.py
|   +-- evaluation.py
|   +-- evaluation_runner.py
|   +-- plugins.py
|   +-- production.py
|   +-- registry.py
|   +-- schema.py
|   +-- tracking.py
|   +-- training.py
|   +-- training_runner.py
+-- train/
|   +-- agentic_rl.py
|   +-- pretrain.py
|   +-- rl.py
|   +-- sft.py
+-- eval/
|   +-- run.py
|   +-- smoke/
+-- plugins/example_evaluator/plugin.json
+-- tests/
+-- reports/
+-- artifacts/runs/
+-- Makefile
+-- pyproject.toml
```

## Installation

Core framework only:

```bash
python -m pip install -e .
```

With PyTorch architecture implementations:

```bash
python -m pip install -e ".[architectures]"
```

With the native PyTorch trainer:

```bash
python -m pip install -e ".[training]"
```

The repository can also run directly without installation:

```bash
PYTHONPATH=src python -m fmops.cli validate
```

## Quick Start

Validate all program-level configuration:

```bash
PYTHONPATH=src python -m fmops.cli schema-validate
PYTHONPATH=src python -m fmops.cli validate
```

Inspect registries and dataset catalog:

```bash
PYTHONPATH=src python -m fmops.cli registry
PYTHONPATH=src python -m fmops.cli datasets --priority P0
PYTHONPATH=src python -m fmops.cli datasets --family VLA-robotics
PYTHONPATH=src python -m fmops.cli datasets --modality video
PYTHONPATH=src python -m fmops.cli benchmarks --dimension vla
PYTHONPATH=src python -m fmops.cli benchmarks --harness lm-eval
```

Generate dry-run artifacts:

```bash
PYTHONPATH=src python -m fmops.cli data-run
PYTHONPATH=src python -m fmops.cli train-run
PYTHONPATH=src python -m fmops.cli train-run --stage SFT
PYTHONPATH=src python -m fmops.cli eval-run --model-id reference-model
PYTHONPATH=src python -m fmops.cli deploy-check
```

Run a real local PyTorch training smoke test:

```bash
PYTHONPATH=src python train/pretrain.py --mode native --max-steps 2 --batch-size 2 --seq-len 64
PYTHONPATH=src python train/sft.py --mode native --max-steps 2 --batch-size 2 --seq-len 64
PYTHONPATH=src python train/rl.py --mode native --max-steps 2 --batch-size 2 --seq-len 64
PYTHONPATH=src python train/agentic_rl.py --mode native --max-steps 2 --batch-size 2 --seq-len 64
```

Generate reports:

```bash
PYTHONPATH=src python -m fmops.cli report --output reports/foundation_model_plan.md
PYTHONPATH=src python -m fmops.cli dashboard --output reports/dashboard.html
```

Generate production integration artifacts:

```bash
PYTHONPATH=src python -m fmops.cli production-plan
PYTHONPATH=src python -m fmops.cli production-check
PYTHONPATH=src python -m fmops.cli production-run --area monitoring
```

Production execution is guarded. External commands only run with both `--execute` and `FMOPS_ALLOW_PRODUCTION_EXECUTE=1`; `production-check` reports missing binaries and environment variables before anything is submitted to Spark, Ray, Slurm, benchmark harnesses, or serving tools.

Run tests:

```bash
PYTHONPATH=src python -m unittest discover -s tests
```

Or use Makefile shortcuts:

```bash
make validate
make test
make data-run
make train-run
make eval-run
make deploy-check
make production-plan
make production-check
make report
make dashboard
```

## Main CLI Commands

| Command | Purpose | Default output |
| --- | --- | --- |
| `validate` | Runs semantic validation across data, architecture, training, and evaluation configs. | stdout |
| `schema-validate` | Validates config files against registered lightweight schemas. | stdout |
| `registry` | Prints registered model components. | stdout |
| `datasets` | Prints dataset catalog summary and filtered entries. | stdout |
| `benchmarks` | Prints benchmark catalog summary and filtered entries. | stdout |
| `data-plan` | Prints data system summary and staged mixture plan. | stdout |
| `arch-compare` | Prints ranked architecture comparison table. | stdout |
| `train-plan` | Prints stage launch commands from the training config. | stdout |
| `eval-plan` | Prints normalized evaluation dimension weights. | stdout |
| `data-run` | Writes a dry-run data pipeline artifact. | `artifacts/runs/data_pipeline_plan.json` |
| `train-run` | Writes dry-run distributed training launch artifacts. | `artifacts/runs/training_plan.json` |
| `eval-run` | Writes evaluation report JSON. | `artifacts/runs/evaluation_report.json` |
| `deploy-check` | Writes deployment envelope checks. | `artifacts/runs/deployment_report.json` |
| `production-plan` | Writes the full production integration task plan. | `artifacts/production/production_plan.json` |
| `production-check` | Checks external production adapter dependencies and required environment variables. | `artifacts/production/preflight_report.json` |
| `production-run` | Plans or executes production adapter tasks; execution requires `--execute` and `FMOPS_ALLOW_PRODUCTION_EXECUTE=1`. | `artifacts/production/execution_report.json` |
| `checkpoint-convert` | Writes a checkpoint conversion manifest. | user-specified target |
| `track-run` | Creates an experiment run manifest. | `artifacts/runs/<run_id>/run_manifest.json` |
| `plugins` | Discovers and validates local plugins. | stdout |
| `dashboard` | Generates a static HTML dashboard. | `reports/dashboard.html` |
| `report` | Generates a Markdown program plan. | `reports/foundation_model_plan.md` |

## Implementation Coverage Matrix

| Requirement area | Implemented assets | Concrete runnable entry points |
| --- | --- | --- |
| 2500T+ data system across open-source, newly collected, and business data | `configs/data_manifest.json`, `configs/datasets_catalog.json`, `src/fmops/data.py`, `src/fmops/data_pipeline.py`, `src/fmops/dataset_catalog.py`, `jobs/data_ingest.py`, `jobs/data_quality.py`, `jobs/data_mixture.py` | `fmops data-plan`, `fmops datasets`, `fmops data-run`, `production-plan --area data`, `production-check --area data` |
| Cleaning, deduplication, clustering, quality assessment, lineage, contamination checks, and staged mixture design | Data operation graph in `configs/data_manifest.json`; production adapter tasks in `configs/production_integration.json`; lineage artifacts from `DataPipelineRunner` | `fmops data-run`, `python jobs/data_quality.py`, `python jobs/data_mixture.py`, guarded `production-run --area data` |
| LLM/VLM/video/VLA dataset documentation with download links | Human-readable directory in this README and `README.zh-CN.md`; machine-readable catalog in `configs/datasets_catalog.json` | `fmops datasets --priority P0`, `fmops datasets --family VLA-robotics`, `fmops datasets --modality video` |
| Twenty next-generation architecture families with fair comparison | `configs/architecture_experiments.json`, `src/fmops/architecture_impl.py`, `src/fmops/architectures.py`, `src/fmops/registry.py`, `tests/test_architecture_impl.py` | `fmops registry`, `fmops arch-compare`, `python -m unittest discover -s tests -p "test_architecture_impl.py"` |
| 400-GPU Pre-training, SFT, RL, and Agentic RL pipeline | `configs/training_pipeline.json`, `src/fmops/training_runner.py`, `src/fmops/native_training.py`, `train/pretrain.py`, `train/sft.py`, `train/rl.py`, `train/agentic_rl.py`, `jobs/training_launch.py` | `fmops train-plan`, `fmops train-run`, native smoke commands under "Quick Start", guarded `production-run --area training` |
| Real evaluation system and benchmark catalog | `configs/evaluation_suite.json`, `configs/benchmark_catalog.json`, `src/fmops/evaluation.py`, `src/fmops/evaluation_runner.py`, `src/fmops/benchmark_catalog.py`, `eval/run.py`, `eval/smoke/*`, `jobs/evaluation_launch.py` | `fmops eval-plan`, `fmops benchmarks`, `fmops eval-run`, `python eval/run.py --samples-dir ...`, guarded `production-run --area evaluation` |
| Checkpoint conversion, deployment validation, monitoring, governance, and release gates | `src/fmops/checkpoint.py`, `src/fmops/deployment.py`, `src/fmops/production.py`, `jobs/checkpoint_convert.py`, `jobs/deployment_validate.py`, `jobs/monitoring_export.py`, `jobs/release_gate.py` | `fmops checkpoint-convert`, `fmops deploy-check`, `fmops production-plan`, `fmops production-check`, guarded `fmops production-run` |
| Framework maturity layer | `src/fmops/schema.py`, `src/fmops/tracking.py`, `src/fmops/plugins.py`, `src/fmops/dashboard.py`, tests, CI, Makefile targets | `fmops schema-validate`, `fmops track-run`, `fmops plugins`, `fmops dashboard`, `make validate`, `make test` |

## Configuration Files

### `configs/data_manifest.json`

Defines the large-scale training data system:

- Target scale: 2500T+.
- Minimum languages: 20+.
- Required modalities: pure text, multimodal, video pretraining, VLA.
- Data sources: open source, newly collected, and business/internal.
- Quality gates: minimum quality score and maximum duplicate rate.
- Processing stages: ingestion, cleaning, deduplication, clustering, quality assessment, packing.
- Mixture stages: warmup, core pretraining, multimodal/video expansion, VLA alignment, reasoning final stage.

### `configs/datasets_catalog.json`

Machine-readable dataset registry:

- Dataset family, modalities, language coverage, license, access mode, download URL, size, priority, risk tags, and schema.
- Supports filtering by family, modality, and priority.
- Provides a production-oriented bridge from documentation to actual ingestion planning.

### `configs/architecture_experiments.json`

Defines fair architecture experiments:

- Unified tokenizer.
- Training token budget.
- Context length.
- Optimizer.
- Hardware budget.
- Required architecture families: MoE, Sparse / Linear Attention, RNN-like Backbone, SSM / Selective Scan, Retention / RetNet, Long Convolution, MLA / KV-Compressed Attention, Hybrid Architecture, MTP, Latent Reasoning, dLLM, Memory-augmented LLM, Mixture-of-Depths, Test-Time Memory, Token-free Byte-level LLM, Omni-modal Architecture, VLA / Robotics Transformer, JEPA / Latent World Model, Neuromorphic / Spiking Backbone, and Reasoning-native Architecture.
- Candidate metrics: validation loss, throughput, reasoning score, GPU memory, stability.

### `configs/training_pipeline.json`

Defines the 400-GPU training pipeline:

- Hardware: 50 nodes x 8 H100-80GB GPUs.
- Stages: Pre-training, SFT, RL, Agentic RL.
- Framework hints: Megatron-DeepSpeed, FSDP, PPO/GRPO-style RL, asynchronous actor-learner.
- Monitors: loss spikes, gradient norms, NaN/Inf, throughput, expert balance, data skew, KL, reward, tool errors.
- Artifacts: checkpoints, tokenizer bundles, data snapshots, rollout traces, deployment candidates.

### `configs/evaluation_suite.json`

Defines release-quality evaluation:

- General multilingual capability.
- Reasoning, math, and code.
- Multimodal understanding.
- VLA action policy.
- Long context.
- Tool calling.
- Agent tasks.
- Edge/car-side deployment efficiency.

### `configs/benchmark_catalog.json`

Defines the machine-readable benchmark catalog:

- 100+ LLM, VLM, video, audio, VLA, agent, tool-use, long-context, safety, and efficiency benchmarks.
- Dimension, family, modality, primary metric, supported harnesses, download URL, license notes, and tags.
- Filterable through `PYTHONPATH=src python -m fmops.cli benchmarks --dimension vla`.

### `configs/production_integration.json`

Defines the production adapter layer:

- Data lake tasks for source ingestion, lineage, cleaning, deduplication, clustering, quality scoring, contamination checks, and staged mixture materialization.
- Slurm-based 400-GPU launch tasks for Pre-training, SFT, RL, and Agentic RL.
- Evaluation harness tasks for lm-eval, VLMEvalKit/OpenCompass-style multimodal evaluation, simulator-backed VLA, and agent benchmarks.
- Checkpoint conversion to HF/safetensors-style serving artifacts.
- vLLM, TensorRT-LLM, GenAI-Perf, and car-side replay deployment validation.
- Prometheus/Grafana monitoring bundle export.
- Release-gate aggregation for data, training, checkpoint, evaluation, deployment, safety, and governance sign-off.

## Framework Components

| Component | File | Responsibility |
| --- | --- | --- |
| Schema validation | `src/fmops/schema.py` | Lightweight object schemas, version checks, config directory validation. |
| Registry | `src/fmops/registry.py` | Model, dataset, trainer, and evaluator registries. |
| Dataset catalog | `src/fmops/dataset_catalog.py` | Machine-readable dataset entries and risk summaries. |
| Benchmark catalog | `src/fmops/benchmark_catalog.py` | Machine-readable benchmark entries, harness mappings, and coverage summaries. |
| Data pipeline | `src/fmops/data_pipeline.py` | Dry-run data pipeline artifacts and lineage URIs. |
| Training runner | `src/fmops/training_runner.py` | Dry-run distributed training launch plans. |
| Native trainer | `src/fmops/native_training.py` | Real PyTorch training loops for Pre-training, SFT, RL, and Agentic RL with checkpoints and metrics. |
| Evaluation runner | `src/fmops/evaluation_runner.py` | Real JSONL/prediction/model-command evaluation, metric aggregation, transcripts, and release gates. |
| Checkpoint conversion | `src/fmops/checkpoint.py` | Checkpoint conversion manifests and optional file copy. |
| Deployment validation | `src/fmops/deployment.py` | Latency, memory, throughput, and power envelope checks. |
| Experiment tracking | `src/fmops/tracking.py` | Run manifests with artifacts, metrics, config refs, and environment. |
| Plugin system | `src/fmops/plugins.py` | Local plugin discovery and loading. |
| Production integration | `src/fmops/production.py` | External adapter planning, preflight checks, guarded execution, and audit reports. |
| Dashboard | `src/fmops/dashboard.py` | Static HTML dashboard generation. |
| Architecture implementation | `src/fmops/architecture_impl.py` | PyTorch reference implementations for twenty model families. |

## Training Entry Points

The training scripts support three modes: local `dry-run` planning, real local `native` PyTorch training, and production `external` dispatch. In `external` mode they hand off to stage-specific backend commands from `FMOPS_PRETRAIN_BACKEND_COMMAND`, `FMOPS_SFT_BACKEND_COMMAND`, `FMOPS_RL_BACKEND_COMMAND`, `FMOPS_AGENTIC_RL_BACKEND_COMMAND`, or the generic `FMOPS_TRAINING_BACKEND_COMMAND`.

```bash
PYTHONPATH=src python train/pretrain.py --config-dir configs --output artifacts/runs/pretrain.json
PYTHONPATH=src python train/sft.py --config-dir configs --output artifacts/runs/sft.json
PYTHONPATH=src python train/rl.py --config-dir configs --output artifacts/runs/rl.json
PYTHONPATH=src python train/agentic_rl.py --config-dir configs --output artifacts/runs/agentic_rl.json
```

Native PyTorch training writes `trainer_state.json`, `metrics.jsonl`, and checkpoint `.pt` files:

```bash
PYTHONPATH=src python train/rl.py \
  --mode native \
  --data-path path/to/rollouts.jsonl \
  --output-dir artifacts/native_training/rl \
  --max-steps 100 \
  --batch-size 8 \
  --seq-len 512
```

Example production handoff:

```bash
FMOPS_AGENTIC_RL_BACKEND_COMMAND="python -m your_actor_learner --mixture {mixture} --world-size {world_size}" \
PYTHONPATH=src python train/agentic_rl.py --mode external --mixture agentic_env_rollouts --world-size 400
```

Production integration is now handled by `configs/production_integration.json`, `src/fmops/production.py`, and the `jobs/*` wrappers. The training adapters generate Slurm submission scripts for Megatron-DeepSpeed/FSDP/RL/actor-learner stages and are guarded by preflight checks.

```bash
PYTHONPATH=src python -m fmops.cli production-plan --area training
PYTHONPATH=src python -m fmops.cli production-check --area training
FMOPS_ALLOW_PRODUCTION_EXECUTE=1 PYTHONPATH=src python -m fmops.cli production-run --area training --execute
```

## Evaluation Entry Point

```bash
PYTHONPATH=src python eval/run.py \
  --config-dir configs \
  --model-id reference-model \
  --output artifacts/runs/evaluation_report.json
```

The runner now performs real local evaluation over JSONL samples. By default it uses the smoke fixtures in `eval/smoke`, computes metrics, applies gates, and writes per-benchmark transcripts next to the report. To evaluate a real model or a precomputed prediction file:

```bash
PYTHONPATH=src python eval/run.py \
  --samples-dir /path/to/eval-jsonl \
  --predictions /path/to/predictions.jsonl \
  --benchmark general_multilingual_core \
  --output artifacts/runs/evaluation_report.json

PYTHONPATH=src python eval/run.py \
  --samples-dir /path/to/eval-jsonl \
  --model-command "python serve_one_sample.py" \
  --fail-on-gate
```

Each JSONL sample may contain `id`, `benchmark`, `dataset`, `prompt` or `question`, `answer` or `reference`, optional `choices`, optional `prediction`, and task-specific fields such as `expected_tool`, `reference_action`, `success`, `collision_free`, `prefill_ms`, `decode_tok_s`, `memory_gb`, and `power_w`.

## Evaluation Benchmark Catalog

`configs/evaluation_suite.json` keeps release benchmark groups, weights, metrics, cadence, and gates. `configs/benchmark_catalog.json` is the machine-readable comprehensive benchmark catalog with 100+ public benchmark entries, dimensions, modalities, metrics, harnesses, download links, and license notes. The table below is the human-readable directory to use when wiring external harnesses. Public benchmarks are useful for comparability; production releases should also include private, contamination-screened, multilingual, multimodal, VLA, and car-side holdouts.

### Evaluation Harnesses

| Harness | Coverage | Link |
| --- | --- | --- |
| EleutherAI LM Evaluation Harness | Text LLM tasks, few-shot prompting, language modeling, QA, reasoning, and many public benchmark adapters. | [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) |
| HELM | Scenario-based model evaluation, robustness, calibration, fairness, toxicity, and transparency reporting. | [stanford-crfm/helm](https://github.com/stanford-crfm/helm) |
| OpenCompass | LLM and VLM evaluation platform with broad Chinese, English, reasoning, code, and multimodal benchmark support. | [open-compass/opencompass](https://github.com/open-compass/opencompass) |
| LightEval | Lightweight Hugging Face evaluation framework for LLM benchmark execution. | [huggingface/lighteval](https://github.com/huggingface/lighteval) |
| VLMEvalKit | VLM evaluation toolkit covering image, document, OCR, chart, video, and multimodal reasoning benchmarks. | [open-compass/VLMEvalKit](https://github.com/open-compass/VLMEvalKit) |
| lmms-eval | Large multimodal model evaluation harness for image, video, and multi-image tasks. | [EvolvingLMMs-Lab/lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) |
| OpenAI Evals | Model behavior evaluation framework and example eval registry. | [openai/evals](https://github.com/openai/evals) |

### Benchmarks by Capability

| Capability | Benchmarks and links | Primary metrics |
| --- | --- | --- |
| General English knowledge and instruction following | [MMLU](https://github.com/hendrycks/test), [MMLU-Pro](https://github.com/TIGER-AI-Lab/MMLU-Pro), [BIG-bench](https://github.com/google/BIG-bench), [BBH](https://github.com/suzgunmirac/BIG-Bench-Hard), [ARC](https://allenai.org/data/arc), [HellaSwag](https://rowanzellers.com/hellaswag/), [TruthfulQA](https://github.com/sylinrl/TruthfulQA), [AGIEval](https://github.com/ruixiangcui/AGIEval) | accuracy, normalized accuracy, calibration, refusal correctness |
| Chinese and multilingual | [C-Eval](https://cevalbenchmark.com/), [CMMLU](https://huggingface.co/datasets/haonan-li/cmmlu), [CLUE](https://github.com/CLUEbenchmark/CLUE), [TyDi QA](https://github.com/google-research-datasets/tydiqa), [FLORES-200](https://github.com/facebookresearch/flores), [Belebele](https://github.com/facebookresearch/belebele), [XTREME](https://github.com/google-research/xtreme), [XQuAD](https://github.com/deepmind/xquad), [MGSM](https://github.com/google-research/url-nlp/tree/main/mgsm) | macro average, per-language score, cross-lingual transfer, translation quality |
| Math, science, and reasoning | [GSM8K](https://huggingface.co/datasets/openai/gsm8k), [MATH](https://huggingface.co/datasets/hendrycks/competition_math), [MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500), [GPQA](https://github.com/idavidrein/gpqa), [OlympiadBench](https://github.com/OpenBMB/OlympiadBench), [NuminaMath](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT), [SciBench](https://github.com/mandyyyyii/scibench), [SciCode](https://github.com/scicode-bench/SciCode) | pass@1, consensus pass, verified answer rate, reasoning trace quality |
| Code and software engineering | [HumanEval](https://github.com/openai/human-eval), [MBPP](https://github.com/google-research/google-research/tree/master/mbpp), [LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench), [APPS](https://github.com/hendrycks/apps), [CodeContests](https://huggingface.co/datasets/deepmind/code_contests), [CRUXEval](https://github.com/facebookresearch/cruxeval), [BigCodeBench](https://github.com/bigcode-project/bigcodebench), [SWE-bench](https://github.com/swe-bench/SWE-bench), [SWE-bench Verified](https://www.swebench.com/) | pass@k, compile rate, unit-test pass rate, issue resolution rate |
| Long context and retrieval reasoning | [LongBench](https://github.com/THUDM/LongBench), [LongBench v2](https://github.com/THUDM/LongBench), [RULER](https://github.com/NVIDIA/RULER), [InfiniteBench](https://github.com/OpenBMB/InfiniteBench), [Needle-in-a-Haystack](https://github.com/gkamradt/LLMTest_NeedleInAHaystack), [BABILong](https://github.com/booydar/babilong), [L-Eval](https://github.com/OpenLMLab/LEval), [Lost in the Middle](https://github.com/nelson-liu/lost-in-the-middle), [NarrativeQA](https://github.com/deepmind/narrativeqa), [Qasper](https://allenai.org/data/qasper), [HotpotQA](https://hotpotqa.github.io/), [2WikiMultiHopQA](https://github.com/Alab-NII/2wikimultihop), [MuSiQue](https://github.com/stonybrooknlp/musique), [QMSum](https://github.com/Yale-LILY/QMSum), [GovReport](https://gov-report-data.github.io/), [RepoBench](https://github.com/Leolty/repobench) | exact match, citation precision, retrieval hit rate, long-range consistency |
| Tool calling and structured output | [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html), [Gorilla OpenFunctions](https://github.com/ShishirPatil/gorilla), [ToolBench](https://github.com/OpenBMB/ToolBench), [StableToolBench](https://github.com/THUNLP-MT/StableToolBench), [API-Bank](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/api-bank), [ToolAlpaca](https://github.com/tangqiaoyu/ToolAlpaca), [ToolQA](https://github.com/night-chen/ToolQA), [tau-bench](https://github.com/sierra-research/tau-bench) | JSON/schema validity, function selection accuracy, tool success rate, repair rate |
| Agent and web workflow | [WebArena](https://github.com/web-arena-x/webarena), [VisualWebArena](https://github.com/web-arena-x/visualwebarena), [MiniWoB++](https://github.com/Farama-Foundation/miniwob-plusplus), [Mind2Web](https://huggingface.co/datasets/osunlp/Mind2Web), [OSWorld](https://github.com/xlang-ai/OSWorld), [WorkArena](https://github.com/ServiceNow/WorkArena), [BrowserGym](https://github.com/ServiceNow/BrowserGym), [GAIA](https://huggingface.co/datasets/gaia-benchmark/GAIA), [AgentBench](https://github.com/THUDM/AgentBench), [AppWorld](https://github.com/StonyBrookNLP/appworld), [WebVoyager](https://github.com/MinorJerry/WebVoyager) | task success, steps to success, unsafe action rate, cost per success |
| Image VLM understanding | [MMMU](https://github.com/MMMU-Benchmark/MMMU), [MMMU-Pro](https://github.com/MMMU-Benchmark/MMMU-Pro), [MMBench](https://github.com/open-compass/MMBench), [MMStar](https://github.com/MMStar-Benchmark/MMStar), [SEED-Bench](https://github.com/AILab-CVC/SEED-Bench), [MM-Vet](https://github.com/yuweihao/MM-Vet), [POPE](https://github.com/AoiDragon/POPE), [HallusionBench](https://github.com/tianyi-lab/HallusionBench), [RealWorldQA](https://huggingface.co/datasets/xai-org/RealworldQA), [VQAv2](https://visualqa.org/download.html), [GQA](https://cs.stanford.edu/people/dorarad/gqa/download.html), [OK-VQA](https://okvqa.allenai.org/download.html), [A-OKVQA](https://allenai.org/project/a-okvqa/home), [VizWiz](https://vizwiz.org/tasks-and-datasets/vqa/) | accuracy, hallucination rate, visual reasoning accuracy, answer grounding |
| Document, OCR, chart, and diagram VLM | [TextVQA](https://textvqa.org/), [DocVQA](https://www.docvqa.org/), [ChartQA](https://github.com/vis-nlp/ChartQA), [InfoVQA](https://rrc.cvc.uab.es/?ch=17), [OCRBench](https://github.com/Yuliang-Liu/MultimodalOCR), [AI2D](https://allenai.org/data/diagrams), [ScienceQA](https://scienceqa.github.io/), [MathVista](https://github.com/lupantech/MathVista), [TallyQA](https://github.com/manoja328/tallyqa), [ScreenSpot](https://github.com/njucckevin/SeeClick) | OCR F1, table/chart QA accuracy, diagram reasoning, grounded pointing accuracy |
| Video, audio, and omni-modal | [Video-MME](https://github.com/BradyFU/Video-MME), [MVBench](https://github.com/OpenGVLab/Ask-Anything/tree/main/video_chat2), [LongVideoBench](https://github.com/longvideobench/LongVideoBench), [MLVU](https://github.com/JUNJIE99/MLVU), [TempCompass](https://github.com/llyx97/TempCompass), [EgoSchema](https://github.com/egoschema/EgoSchema), [NExT-QA](https://github.com/doc-doc/NExT-QA), [ActivityNet-QA](https://github.com/MILVLG/activitynet-qa), [TVQA](https://tvqa.cs.unc.edu/), [TGIF-QA](https://github.com/YunseokJANG/tgif-qa), [AVQA](https://mn.cs.tsinghua.edu.cn/avqa/), [AudioCaps](https://audiocaps.github.io/), [AudioSet](https://research.google.com/audioset/), [Clotho](https://zenodo.org/records/4783391), [MusicCaps](https://google-research.github.io/seanet/musiclm/examples/) | temporal reasoning, video QA accuracy, audio-caption quality, cross-modal grounding |
| VLA, robotics, and embodied tasks | [LIBERO](https://libero-project.github.io/main.html), [CALVIN](https://calvin.cs.uni-freiburg.de/), [RLBench](https://github.com/stepjam/RLBench), [Language Table](https://github.com/google-research/language-table), [Meta-World](https://meta-world.github.io/), [ManiSkill](https://maniskill.readthedocs.io/), [robomimic](https://robomimic.github.io/docs/datasets/overview.html), [RoboCasa](https://github.com/robocasa/robocasa), [robosuite](https://github.com/ARISE-Initiative/robosuite), [SimplerEnv](https://github.com/simpler-env/SimplerEnv), [Open X-Embodiment](https://github.com/google-deepmind/open_x_embodiment), [Habitat-Lab](https://github.com/facebookresearch/habitat-lab), [AI2-THOR](https://ai2thor.allenai.org/), [ALFRED](https://askforalfred.com/), [TEACh](https://github.com/alexa/teach), [BEHAVIOR-1K](https://behavior.stanford.edu/) | task success, recovery rate, action L2, collision-free rate, instruction grounding |
| Autonomous driving and car-side VLA | [nuScenes](https://www.nuscenes.org/download), [nuPlan](https://www.nuplan.org/nuplan), [Waymo Open Dataset](https://waymo.com/open/), [Argoverse 2](https://argoverse.org/av2.html), [INTERACTION](https://interaction-dataset.com/), [CARLA Leaderboard](https://leaderboard.carla.org/), [Bench2Drive](https://github.com/Thinklab-SJTU/Bench2Drive), [NAVSIM](https://github.com/autonomousvision/navsim), [BDD100K](https://bdd-data.berkeley.edu/), [KITTI](https://www.cvlibs.net/datasets/kitti/) | route completion, driving score, collision rate, planning error, frame-to-action latency |
| Safety, bias, and robustness | [BBQ](https://github.com/nyu-mll/BBQ), [RealToxicityPrompts](https://github.com/allenai/real-toxicity-prompts), [ToxiGen](https://github.com/microsoft/TOXIGEN), [HarmBench](https://github.com/centerforaisafety/HarmBench), [SafetyBench](https://github.com/thu-coai/SafetyBench), [AdvBench](https://github.com/llm-attacks/llm-attacks), [JailbreakBench](https://github.com/JailbreakBench/jailbreakbench), [DecodingTrust](https://github.com/AI-secure/DecodingTrust) | policy violation rate, jailbreak success rate, bias score, robustness under perturbation |
| Deployment and inference efficiency | [MLPerf Inference](https://mlcommons.org/benchmarks/inference-datacenter/), [MLPerf Inference Edge](https://mlcommons.org/benchmarks/inference-edge/), [MLPerf Tiny](https://mlcommons.org/benchmarks/inference-tiny/), [GenAI-Perf](https://github.com/triton-inference-server/perf_analyzer/tree/main/genai-perf), [AIPerf](https://www.benchcouncil.org/aiperf.html), [vLLM benchmarks](https://docs.vllm.ai/en/latest/contributing/benchmarks.html), [TensorRT-LLM benchmarking](https://nvidia.github.io/TensorRT-LLM/performance/perf-benchmarking.html), [LLMPerf](https://github.com/ray-project/llmperf) | TTFT, prefill latency, decode tokens/s, memory, power, throughput per dollar |

## Generated Artifacts

| Artifact | Generated by | Description |
| --- | --- | --- |
| `reports/foundation_model_plan.md` | `fmops report` | Program-level Markdown plan. |
| `reports/dashboard.html` | `fmops dashboard` | Static overview dashboard. |
| `artifacts/runs/data_pipeline_plan.json` | `fmops data-run` | Data pipeline stage artifacts and lineage URIs. |
| `artifacts/runs/training_plan.json` | `fmops train-run` | Distributed launch plan for all training stages. |
| `artifacts/runs/evaluation_report.json` | `fmops eval-run` | Benchmark results and pass/fail summary. |
| `artifacts/runs/deployment_report.json` | `fmops deploy-check` | Deployment envelope checks. |
| `artifacts/production/production_plan.json` | `fmops production-plan` | Production adapter task graph, commands, gates, and owners. |
| `artifacts/production/preflight_report.json` | `fmops production-check` | Missing external binaries and environment variables before production execution. |
| `artifacts/production/execution_report.json` | `fmops production-run` | Guarded production execution results and log paths. |
| `checkpoint_manifest.json` | `fmops checkpoint-convert` | Checkpoint source/target conversion metadata. |
| `run_manifest.json` | `fmops track-run` | Experiment tracking manifest. |

## Architecture Implementations and References

The reference implementations in `src/fmops/architecture_impl.py` are meant to be readable, runnable, and testable. They are not high-performance replacements for production kernels such as FlashAttention, expert parallel MoE kernels, fused state-space kernels, or inference-specific KV-cache engines.

When a paper does not provide an official implementation, the code column points to widely used framework integrations, author-adjacent releases, or community reference implementations.

Install the optional PyTorch dependency:

```bash
pip install -e ".[architectures]"
PYTHONPATH=src python -m unittest discover -s tests -p "test_architecture_impl.py"
```

| Architecture family | Implementation classes | Implemented mechanism | Papers | GitHub / code |
| --- | --- | --- | --- | --- |
| MoE | `MoETransformerBlock`, `MoEFeedForward`, `TopKRouter` | Top-k routing, expert FFNs, auxiliary load-balance loss, attention + MoE residual block. | [GShard](https://arxiv.org/abs/2006.16668), [Switch Transformers](https://arxiv.org/abs/2101.03961), [Mixtral of Experts](https://arxiv.org/abs/2401.04088), [DeepSeek-V3](https://arxiv.org/abs/2412.19437) | [tensorflow/lingvo GShard](https://github.com/tensorflow/lingvo/blob/master/lingvo/core/gshard_builder.py), [google-research/t5x](https://github.com/google-research/t5x), [mistralai/mistral-inference](https://github.com/mistralai/mistral-inference), [deepseek-ai/DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3) |
| Sparse / Linear Attention | `SparseLinearAttentionBlock`, `SlidingWindowAttention`, `CausalLinearAttention` | Causal sliding-window sparse attention plus ELU feature-map causal linear attention with a learnable mixing gate. | [Longformer](https://arxiv.org/abs/2004.05150), [BigBird](https://arxiv.org/abs/2007.14062), [Linear Transformers](https://arxiv.org/abs/2006.16236), [Performer](https://arxiv.org/abs/2009.14794) | [allenai/longformer](https://github.com/allenai/longformer), [google-research/bigbird](https://github.com/google-research/bigbird), [idiap/fast-transformers](https://github.com/idiap/fast-transformers), [google-research Performer](https://github.com/google-research/google-research/tree/master/performer) |
| RNN-like Backbone | `RNNBackboneBlock`, `RecurrentTokenMixer` | Token-by-token recurrent state update, learned decay, gated state output, FFN residual block. | [RWKV](https://arxiv.org/abs/2305.13048), [RetNet](https://arxiv.org/abs/2307.08621), [Mamba](https://arxiv.org/abs/2312.00752) | [BlinkDL/RWKV-LM](https://github.com/BlinkDL/RWKV-LM), [microsoft/torchscale RetNet](https://github.com/microsoft/torchscale/blob/main/torchscale/architecture/retnet.py), [state-spaces/mamba](https://github.com/state-spaces/mamba) |
| SSM / Selective Scan | `SelectiveStateSpaceBlock` | Mamba-style input-dependent decay, gated depthwise convolution, selective recurrent state, FFN residual block. | [Mamba](https://arxiv.org/abs/2312.00752), [Mamba-2](https://arxiv.org/abs/2405.21060), [VMamba](https://arxiv.org/abs/2401.10166), [MambaByte](https://arxiv.org/abs/2401.13660) | [state-spaces/mamba](https://github.com/state-spaces/mamba), [MzeroMiko/VMamba](https://github.com/MzeroMiko/VMamba), [goombalab/hydra](https://github.com/goombalab/hydra) |
| Retention / RetNet | `RetentionBlock` | Multi-head decayed causal retention with attention-like parallel training behavior and recurrent-cache interpretation. | [Retentive Network](https://arxiv.org/abs/2307.08621), [TorchScale RetNet](https://github.com/microsoft/torchscale) | [microsoft/torchscale RetNet](https://github.com/microsoft/torchscale/blob/main/torchscale/architecture/retnet.py), [Jamie-Stirling/RetNet](https://github.com/Jamie-Stirling/RetNet), [syncdoth/RetNet](https://github.com/syncdoth/RetNet) |
| Long Convolution | `LongConvolutionBlock` | Causal depthwise long convolution with gating and residual FFN, covering Hyena/H3-style attention replacement. | [Hyena Hierarchy](https://proceedings.mlr.press/v202/poli23a.html), [H3](https://arxiv.org/abs/2212.14052), [StripedHyena](https://arxiv.org/abs/2311.09431) | [HazyResearch/safari](https://github.com/HazyResearch/safari), [togethercomputer/stripedhyena](https://github.com/togethercomputer/stripedhyena) |
| MLA / KV-Compressed Attention | `KVCompressedAttentionBlock` | GQA/MQA-style shared KV heads plus latent K/V compression to reduce decode cache and memory bandwidth. | [MQA](https://arxiv.org/abs/1911.02150), [GQA](https://arxiv.org/abs/2305.13245), [DeepSeek-V2 MLA](https://arxiv.org/abs/2405.04434), [DeepSeek-V3](https://arxiv.org/abs/2412.19437) | [deepseek-ai/DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3), [deepseek-ai/FlashMLA](https://github.com/deepseek-ai/FlashMLA), [huggingface/transformers GQA](https://github.com/huggingface/transformers) |
| Hybrid Architecture | `HybridArchitectureModel` | Interleaves Transformer attention blocks with recurrent blocks under one causal LM head. | [Jamba](https://arxiv.org/abs/2403.19887), [Griffin / RecurrentGemma](https://arxiv.org/abs/2402.19427), [Mamba](https://arxiv.org/abs/2312.00752) | [huggingface/transformers Jamba](https://github.com/huggingface/transformers/tree/main/src/transformers/models/jamba), [google-deepmind/recurrentgemma](https://github.com/google-deepmind/recurrentgemma), [state-spaces/mamba](https://github.com/state-spaces/mamba) |
| MTP | `MultiTokenPredictionModel` | Shared decoder trunk with multiple future-token prediction heads and offset-specific cross-entropy losses. | [Better & Faster LLMs via Multi-token Prediction](https://arxiv.org/abs/2404.19737), [Medusa](https://arxiv.org/abs/2401.10774), [DeepSeek-V3](https://arxiv.org/abs/2412.19437) | [FasterDecoding/Medusa](https://github.com/FasterDecoding/Medusa), [deepseek-ai/DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3), [vllm-project/vllm MTP issue](https://github.com/vllm-project/vllm/issues/12181) |
| Latent Reasoning | `LatentReasoningModel` | Inserts learnable latent thought tokens into the sequence and hides those positions from visible-token loss. | [Coconut](https://arxiv.org/abs/2412.06769), [Quiet-STaR](https://arxiv.org/abs/2403.09629) | [facebookresearch/coconut](https://github.com/facebookresearch/coconut), [ezelikman/quiet-star](https://github.com/ezelikman/quiet-star) |
| dLLM | `DiscreteDiffusionLanguageModel` | Discrete masked diffusion process, timestep embedding, bidirectional denoising Transformer, masked-token reconstruction loss. | [Diffusion-LM](https://arxiv.org/abs/2205.14217), [DiffuSeq](https://arxiv.org/abs/2210.08933), [LLaDA](https://arxiv.org/abs/2502.09992) | [XiangLi1999/Diffusion-LM](https://github.com/XiangLi1999/Diffusion-LM), [Shark-NLP/DiffuSeq](https://github.com/Shark-NLP/DiffuSeq), [ML-GSAI/LLaDA](https://github.com/ML-GSAI/LLaDA) |
| Memory-augmented LLM | `MemoryAugmentedLM`, `MemoryAugmentedBlock`, `DifferentiableMemory` | Learned or external memory key-value bank, differentiable retrieval, gated memory fusion. | [RETRO](https://arxiv.org/abs/2112.04426), [Memorizing Transformers](https://arxiv.org/abs/2203.08913), [REALM](https://arxiv.org/abs/2002.08909), [MemGPT](https://arxiv.org/abs/2310.08560) | [lucidrains/RETRO-pytorch](https://github.com/lucidrains/RETRO-pytorch), [lucidrains/memorizing-transformers-pytorch](https://github.com/lucidrains/memorizing-transformers-pytorch), [google-research/language REALM](https://github.com/google-research/language/tree/master/language/realm), [letta-ai/letta](https://github.com/letta-ai/letta) |
| Mixture-of-Depths | `MixtureOfDepthsModel` | Per-token depth router, capacity-limited optional blocks, straight-through routing mask, shared LM head. | [Mixture-of-Depths](https://arxiv.org/abs/2404.02258), [Adaptive Computation Time](https://arxiv.org/abs/1603.08983) | [kyegomez/Mixture-of-Depths](https://github.com/kyegomez/Mixture-of-Depths), [astramind-ai/Mixture-of-depths](https://github.com/astramind-ai/Mixture-of-depths) |
| Test-Time Memory | `TestTimeMemoryModel` | Context-derived fast memory, learned memory bank, associative read, gated fusion into decoder hidden states. | [TTT](https://arxiv.org/abs/2407.04620), [Titans](https://arxiv.org/abs/2501.00663), [Memorizing Transformers](https://arxiv.org/abs/2203.08913) | [test-time-training/ttt-lm-pytorch](https://github.com/test-time-training/ttt-lm-pytorch), [test-time-training/ttt-lm-jax](https://github.com/test-time-training/ttt-lm-jax), [lucidrains/titans-pytorch](https://github.com/lucidrains/titans-pytorch) |
| Token-free Byte-level LLM | `ByteLevelLanguageModel` | UTF-8/byte vocabulary, byte patch convolution, decoder backbone, byte-level LM loss. | [ByT5](https://arxiv.org/abs/2105.13626), [MEGABYTE](https://arxiv.org/abs/2305.07185), [MambaByte](https://arxiv.org/abs/2401.13660), [SpaceByte](https://arxiv.org/abs/2404.14408) | [google-research/byt5](https://github.com/google-research/byt5), [lucidrains/MEGABYTE-pytorch](https://github.com/lucidrains/MEGABYTE-pytorch), [kjslag/spacebyte](https://github.com/kjslag/spacebyte) |
| Omni-modal Architecture | `OmniModalArchitecture` | Projects text, image, video, audio, and action inputs into a shared token space with modality embeddings and text/action heads. | [Chameleon](https://arxiv.org/abs/2405.09818), [Unified-IO 2](https://arxiv.org/abs/2312.17172), [ImageBind](https://arxiv.org/abs/2305.05665), [AnyGPT](https://arxiv.org/abs/2402.12226), [NExT-GPT](https://arxiv.org/abs/2309.05519) | [facebookresearch/chameleon](https://github.com/facebookresearch/chameleon), [allenai/unified-io-2](https://github.com/allenai/unified-io-2), [facebookresearch/ImageBind](https://github.com/facebookresearch/ImageBind), [OpenMOSS/AnyGPT](https://github.com/OpenMOSS/AnyGPT), [NExT-GPT/NExT-GPT](https://github.com/NExT-GPT/NExT-GPT) |
| VLA / Robotics Transformer | `VLARoboticsTransformer` | Text/image/proprioception/action-history fusion with continuous action, discrete action, and success heads. | [RT-1](https://arxiv.org/abs/2212.06817), [RT-2](https://arxiv.org/abs/2307.15818), [OpenVLA](https://arxiv.org/abs/2406.09246), [Octo](https://arxiv.org/abs/2405.12213) | [google-research/robotics_transformer](https://github.com/google-research/robotics_transformer), [openvla/openvla](https://github.com/openvla/openvla), [octo-models/octo](https://github.com/octo-models/octo) |
| JEPA / Latent World Model | `LatentWorldModel` | Action-conditioned latent predictive objective over context and target features, suitable for video/VLA world modeling. | [I-JEPA](https://arxiv.org/abs/2301.08243), [V-JEPA](https://arxiv.org/abs/2404.08471), [World Models](https://arxiv.org/abs/1803.10122), [DreamerV3](https://arxiv.org/abs/2301.04104) | [facebookresearch/ijepa](https://github.com/facebookresearch/ijepa), [facebookresearch/jepa](https://github.com/facebookresearch/jepa), [danijar/dreamerv3](https://github.com/danijar/dreamerv3) |
| Neuromorphic / Spiking Backbone | `SpikingBackboneModel` | Leaky integrate-and-fire recurrent state with surrogate spike gradient, LM head for event/edge exploration. | [Spikformer](https://arxiv.org/abs/2209.15425), [SpikeGPT](https://arxiv.org/abs/2302.13939), [SpikingBERT](https://arxiv.org/abs/2308.10873) | [ZK-Zhou/spikformer](https://github.com/ZK-Zhou/spikformer), [Rydrgn/SpikeGPT](https://github.com/ridgerchu/SpikeGPT), [fangwei123456/spikingjelly](https://github.com/fangwei123456/spikingjelly) |
| Reasoning-native Architecture | `ReasoningNativeArchitecture` | Shared decoder trunk with policy, verifier/process-reward, planner, and value heads. | [STaR](https://arxiv.org/abs/2203.14465), [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050), [Training Verifiers](https://arxiv.org/abs/2110.14168), [DeepSeek-R1](https://arxiv.org/abs/2501.12948) | [ezelikman/STaR](https://github.com/ezelikman/STaR), [openai/prm800k](https://github.com/openai/prm800k), [openai/grade-school-math](https://github.com/openai/grade-school-math), [deepseek-ai/DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1), [huggingface/open-r1](https://github.com/huggingface/open-r1) |

Minimal model example:

```python
import torch
from fmops.architecture_impl import ModelConfig, build_reference_implementation

cfg = ModelConfig(vocab_size=32000, d_model=512, n_heads=8, n_layers=4, d_ff=2048)
model = build_reference_implementation("MTP", cfg, prediction_offsets=4)
tokens = torch.randint(0, cfg.vocab_size, (2, 128))
out = model(tokens, labels=tokens)
print(out["loss"], out["logits"].shape)
```

## Dataset Catalog

The framework has two dataset layers:

- `configs/datasets_catalog.json` is the machine-readable catalog used by CLI and tests.
- This README contains a broader human-readable directory for planning large LLM, VLM, video, VLA, robotics, and autonomous-driving data programs.

Important note: many web, image, video, and robot datasets provide metadata, URLs, video IDs, timestamps, RLDS/TFDS shards, or access portals rather than immediately downloadable media. Production ingestion must preserve source terms, license, opt-out handling, privacy checks, and actual rehydration success rates.

### Machine-readable P0 Catalog

| Dataset | Family | Modalities | Access | Download |
| --- | --- | --- | --- | --- |
| FineWeb | LLM-pretrain | text | Hugging Face | [link](https://huggingface.co/datasets/HuggingFaceFW/fineweb) |
| FineWeb-2 | LLM-pretrain | text | Hugging Face | [link](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) |
| Dolma | LLM-pretrain | text, code | Hugging Face | [link](https://huggingface.co/datasets/allenai/dolma) |
| RedPajama-Data-V2 | LLM-pretrain | text | Hugging Face | [link](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-V2) |
| The Stack v2 | LLM-code | code, text | Hugging Face | [link](https://huggingface.co/datasets/bigcode/the-stack-v2) |
| OpenWebMath | LLM-math | text | Hugging Face | [link](https://huggingface.co/datasets/open-web-math/open-web-math) |
| DataComp CommonPool | VLM-pretrain | image, text | official | [link](https://www.datacomp.ai/dcclip/getting_started.html) |
| Re-LAION-5B | VLM-pretrain | image, text | official | [link](https://laion.ai/blog/relaion-5b/) |
| MINT-1T | VLM-interleaved | image, text | Hugging Face | [link](https://huggingface.co/datasets/mlfoundations/MINT-1T-HTML) |
| ShareGPT4V | VLM-instruction | image, text | Hugging Face | [link](https://huggingface.co/datasets/Lin-Chen/ShareGPT4V) |
| InternVid | Video-pretrain | video, text | Hugging Face | [link](https://huggingface.co/datasets/OpenGVLab/InternVid) |
| HowTo100M | Video-pretrain | video, text, audio | official | [link](https://www.di.ens.fr/willow/research/howto100m/) |
| Ego4D | Video-egocentric | video, text, audio | application | [link](https://ego4d-data.org/docs/start-here/) |
| Open X-Embodiment | VLA-robotics | video, text, action, proprioception | official | [link](https://github.com/google-deepmind/open_x_embodiment) |
| DROID | VLA-robotics | video, text, action, proprioception | official | [link](https://droid-dataset.github.io/droid/the-droid-dataset) |
| BridgeData V2 | VLA-robotics | video, text, action, proprioception | official | [link](https://rail-berkeley.github.io/bridgedata/) |
| nuScenes | VLA-driving | video, lidar, radar, map, trajectory | official | [link](https://www.nuscenes.org/download) |
| Waymo Open Dataset | VLA-driving | video, lidar, trajectory | official | [link](https://waymo.com/open/) |

### LLM Pretraining Data

| Dataset | Scope | Typical use | Access |
| --- | --- | --- | --- |
| Common Crawl | PB-scale WARC/WAT/WET web data | Raw multilingual web corpus | [Get Started](https://commoncrawl.org/get-started), [Latest Crawl](https://commoncrawl.org/latest-crawl) |
| FineWeb | 18T+ English tokens from cleaned Common Crawl | High-quality English web pretraining | [HuggingFaceFW/fineweb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) |
| FineWeb-Edu | Education-filtered FineWeb | Knowledge and reasoning-heavy pretraining | [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) |
| FineWeb-2 | 1000+ language web corpus | Large multilingual LLM pretraining | [HuggingFaceFW/fineweb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) |
| Dolma | OLMo pretraining corpus with web, code, books, wiki, papers | Open LLM reproduction and scaling | [allenai/dolma](https://huggingface.co/datasets/allenai/dolma), [Dolma docs](https://allenai.github.io/dolma/) |
| Dolma 3 | OLMo 3 mix, mid-training, long-context recipes | Modern open data mixture reference | [allenai/dolma3](https://github.com/allenai/dolma3) |
| RedPajama-Data-V2 | 84 Common Crawl snapshots with quality signals | Massive web filtering and pretraining | [RedPajama-Data-V2](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-V2) |
| RedPajama-Data-1T | CommonCrawl, C4, GitHub, arXiv, Wiki, StackExchange | LLaMA-style clean-room baseline | [RedPajama-Data-1T](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T) |
| DCLM Baseline | DataComp for Language Models baseline | Data filtering and 7B-scale training | [mlfoundations/dclm-baseline-1.0](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0) |
| SlimPajama-627B | Cleaned and deduplicated RedPajama subset | Small/mid-size LLM training | [cerebras/SlimPajama-627B](https://huggingface.co/datasets/cerebras/SlimPajama-627B) |
| The Pile | 825 GiB English corpus with 22 subsets | Classic LLM baseline | [EleutherAI/pile](https://huggingface.co/datasets/EleutherAI/pile) |
| Common Pile v0.1 | Public-domain/open-licensed text | License-conscious training pool | [Common Pile collection](https://huggingface.co/collections/common-pile/common-pile-v01) |
| ROOTS / BigScience Data | BLOOM multilingual data ecosystem | Multilingual governance reference | [BigScience Data](https://huggingface.co/bigscience-data) |
| OSCAR | Language-classified Common Crawl | Multilingual pretraining | [OSCAR](https://oscar-project.org/), [OSCAR-2201](https://huggingface.co/datasets/oscar-corpus/OSCAR-2201) |
| CulturaX | 6.3T tokens across 167 languages | Multilingual LLM training | [uonlp/CulturaX](https://huggingface.co/datasets/uonlp/CulturaX) |
| mC4 / C4 | Cleaned Common Crawl for 100+ languages | T5/mT5-style pretraining | [allenai/c4](https://huggingface.co/datasets/allenai/c4), [legacy-datasets/mc4](https://huggingface.co/datasets/legacy-datasets/mc4) |
| CC100 | 100+ language Common Crawl corpus | Multilingual encoder/LLM training | [statmt/cc100](https://data.statmt.org/cc-100/) |
| MADLAD-400 | 400+ language web text | Low-resource language coverage | [allenai/MADLAD-400](https://huggingface.co/datasets/allenai/MADLAD-400) |
| Wikimedia Dumps | Multilingual encyclopedia XML dumps | High-quality knowledge corpus | [Wikimedia Dumps](https://dumps.wikimedia.org/) |
| Wiki40B | Clean Wikipedia text in 40+ languages | Multilingual pretraining/evaluation | [google/wiki40b](https://huggingface.co/datasets/google/wiki40b) |
| Project Gutenberg | Public-domain books | Long-form text and literature | [Gutenberg feeds](https://www.gutenberg.org/cache/epub/feeds/) |
| PG-19 | Project Gutenberg-derived long-form benchmark | Long-context language modeling | [deepmind/pg19](https://huggingface.co/datasets/deepmind/pg19) |
| arXiv bulk data | Paper source/PDF/metadata | Science, math, long documents | [arXiv Bulk Access](https://info.arxiv.org/help/bulk_data/index.html) |
| PubMed / MEDLINE | Biomedical abstracts and citation XML | Biomedical pretraining | [PubMed Download](https://pubmed.ncbi.nlm.nih.gov/download/), [ncbi/pubmed](https://huggingface.co/datasets/ncbi/pubmed) |
| PMC Open Access | Open-access biomedical full text | Medical full-text pretraining | [PMC OA](https://www.ncbi.nlm.nih.gov/pmc/tools/openftlist/) |
| Stack Exchange Data Dump | QA community text | Technical QA and reasoning data | [Internet Archive StackExchange](https://archive.org/details/stackexchange) |
| The Stack v2 | 3B+ files, 600+ programming languages | Code LLM pretraining | [bigcode/the-stack-v2](https://huggingface.co/datasets/bigcode/the-stack-v2) |
| StarCoderData | Code, issues, commits, notebooks | Code pretraining and FIM | [bigcode/starcoderdata](https://huggingface.co/datasets/bigcode/starcoderdata) |
| OpenWebMath | 14.7B math tokens | Math pretraining and continued training | [open-web-math/open-web-math](https://huggingface.co/datasets/open-web-math/open-web-math) |
| Proof-Pile-2 | 55B math/science tokens | Math, proof, science reasoning | [EleutherAI/proof-pile-2](https://huggingface.co/datasets/EleutherAI/proof-pile-2) |
| FineMath | High-quality math tokens | Math reasoning enhancement | [HuggingFaceTB/finemath](https://huggingface.co/datasets/HuggingFaceTB/finemath) |

### LLM Instruction, Preference, Reasoning, and Agent Data

| Dataset | Scope | Typical use | Access |
| --- | --- | --- | --- |
| Stanford Alpaca | 52K instruction-following samples | SFT baseline | [tatsu-lab/alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) |
| Databricks Dolly 15K | Human-written instruction data | SFT baseline | [databricks/databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) |
| OpenAssistant OASST1/OASST2 | Multi-turn message trees | SFT, reward modeling, dialogue | [oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1), [oasst2](https://huggingface.co/datasets/OpenAssistant/oasst2) |
| UltraChat / UltraChat 200K | Large synthetic multi-turn dialogue | SFT and chat alignment | [UltraChat](https://huggingface.co/datasets/openbmb/UltraChat), [ultrachat_200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) |
| OpenHermes 2.5 | Instruction/chat compilation | General SFT | [teknium/OpenHermes-2.5](https://huggingface.co/datasets/teknium/OpenHermes-2.5) |
| OpenOrca / SlimOrca | GPT-augmented FLAN-style data | Explanation-style SFT and distillation | [OpenOrca](https://huggingface.co/datasets/Open-Orca/OpenOrca), [SlimOrca](https://huggingface.co/datasets/Open-Orca/SlimOrca) |
| FLAN Collection | Multi-task instruction mixture | Instruction generalization | [Muennighoff/flan](https://huggingface.co/datasets/Muennighoff/flan) |
| Tulu 3 SFT Mixture | Open post-training mixture | Modern SFT recipe reference | [allenai/tulu-3-sft-mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) |
| LMSYS-Chat-1M | Real user-model conversations | Real prompt distribution analysis | [lmsys/lmsys-chat-1m](https://huggingface.co/datasets/lmsys/lmsys-chat-1m) |
| WildChat | Real ChatGPT interaction logs | Real prompt distribution and SFT | [allenai/WildChat](https://huggingface.co/datasets/allenai/WildChat) |
| No Robots | Human-written instruction data | High-quality small SFT set | [HuggingFaceH4/no_robots](https://huggingface.co/datasets/HuggingFaceH4/no_robots) |
| Anthropic HH-RLHF | Chosen/rejected preference pairs | RM, DPO, RLHF | [Anthropic/hh-rlhf](https://huggingface.co/datasets/Anthropic/hh-rlhf) |
| UltraFeedback | Feedback and preference data | DPO/RM/preference optimization | [openbmb/UltraFeedback](https://huggingface.co/datasets/openbmb/UltraFeedback) |
| Nectar | 7-way ranked preference data | Reward modeling and ranking | [berkeley-nest/Nectar](https://huggingface.co/datasets/berkeley-nest/Nectar) |
| HelpSteer | Multi-attribute helpfulness/correctness labels | Controllable reward modeling | [nvidia/HelpSteer](https://huggingface.co/datasets/nvidia/HelpSteer) |
| GSM8K / MATH / NuminaMath | Math problems and solutions | Reasoning SFT, RLVR, evaluation | [GSM8K](https://huggingface.co/datasets/openai/gsm8k), [MATH](https://huggingface.co/datasets/hendrycks/competition_math), [NuminaMath](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT) |
| APPS / CodeContests / SWE-bench | Code tasks, contests, issue-to-patch data | Code SFT, coding agents, RLVR | [APPS](https://huggingface.co/datasets/codeparrot/apps), [CodeContests](https://huggingface.co/datasets/deepmind/code_contests), [SWE-bench](https://huggingface.co/datasets/princeton-nlp/SWE-bench) |
| ToolBench / APIBench | Tool and API calling tasks | Tool-use and function-calling agents | [ToolBench](https://github.com/OpenBMB/ToolBench), [APIBench](https://huggingface.co/datasets/gorilla-llm/APIBench) |
| WebArena / MiniWoB++ / Mind2Web | Web-agent tasks and browser actions | Agentic RL and web-agent evaluation | [WebArena](https://github.com/web-arena-x/webarena), [MiniWoB++](https://github.com/Farama-Foundation/miniwob-plusplus), [Mind2Web](https://huggingface.co/datasets/osunlp/Mind2Web) |

### VLM Image-Text, Interleaved, OCR, and VQA Data

| Dataset | Scope | Typical use | Access |
| --- | --- | --- | --- |
| LAION-400M / LAION-5B / Re-LAION-5B | Large-scale image-text metadata | CLIP/VLM pretraining | [LAION-400M](https://laion.ai/blog/laion-400-open-dataset/), [LAION-5B](https://laion.ai/laion-5b-a-new-era-of-open-large-scale-multi-modal-datasets/), [Re-LAION-5B](https://laion.ai/blog/relaion-5b/) |
| DataComp / CommonPool | 12.8B image-text candidate pool | Data filtering and CLIP training | [DataComp](https://www.datacomp.ai/dcclip/getting_started.html), [datacomp_1b](https://huggingface.co/datasets/mlfoundations/datacomp_1b) |
| Conceptual Captions 3M / 12M | Web image-caption pairs | Image-text pretraining | [CC3M](https://ai.google.com/research/ConceptualCaptions/download), [CC12M](https://github.com/google-research-datasets/conceptual-12m) |
| SBU Captions / YFCC100M / RedCaps | Flickr/Reddit image-text data | VLM pretraining | [SBU](https://www.cs.rice.edu/~vo9/sbucaptions/), [YFCC100M](https://multimediacommons.wordpress.com/yfcc100m-core-dataset/), [RedCaps](https://redcaps.xyz/) |
| WIT | Wikipedia image-text examples across 108 languages | Multilingual VLM pretraining | [google/wit](https://huggingface.co/datasets/google/wit) |
| OBELICS / MMC4 / MINT-1T | Interleaved image-text web documents | Large multimodal LMM pretraining | [OBELICS](https://huggingface.co/datasets/HuggingFaceM4/OBELICS), [MMC4](https://github.com/allenai/mmc4), [MINT-1T](https://huggingface.co/datasets/mlfoundations/MINT-1T-HTML) |
| COCO Captions / Visual Genome / Open Images | Captions, boxes, relationships, labels | Captioning, grounding, visual reasoning | [COCO](https://cocodataset.org/#download), [Visual Genome](https://homes.cs.washington.edu/~ranjay/visualgenome/index.html), [Open Images](https://storage.googleapis.com/openimages/web/index.html) |
| VQAv2 / GQA / OK-VQA / A-OKVQA / VizWiz | Image question answering | VLM SFT and evaluation | [VQA](https://visualqa.org/download.html), [GQA](https://cs.stanford.edu/people/dorarad/gqa/download.html), [OK-VQA](https://okvqa.allenai.org/download.html), [A-OKVQA](https://allenai.org/project/a-okvqa/home), [VizWiz](https://vizwiz.org/tasks-and-datasets/vqa/) |
| TextVQA / ST-VQA / OCR-VQA / DocVQA / MP-DocVQA | OCR and document VQA | OCR-VLM and document understanding | [TextVQA](https://textvqa.org/dataset/), [ST-VQA](https://rrc.cvc.uab.es/?ch=11), [OCR-VQA](https://ocr-vqa.github.io/), [DocVQA](https://www.docvqa.org/datasets) |
| ChartQA / PlotQA / AI2D / ScienceQA | Charts, diagrams, science QA | Visual reasoning and chart understanding | [ChartQA](https://huggingface.co/datasets/HuggingFaceM4/ChartQA), [PlotQA](https://github.com/NiteshMethani/PlotQA), [AI2D](https://prior.allenai.org/projects/diagram-understanding), [ScienceQA](https://huggingface.co/datasets/derek-thomas/ScienceQA) |
| LLaVA Pretrain / LLaVA-Instruct / LLaVA-NeXT | Multimodal instruction data | LLaVA-style VLM pretraining/SFT | [LLaVA-Pretrain](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain), [LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K), [LLaVA-NeXT-Data](https://huggingface.co/datasets/lmms-lab/LLaVA-NeXT-Data) |
| ShareGPT4V / ALLaVA / Cambrian-10M | Dense captions and instruction mixtures | VLM SFT and capability coverage | [ShareGPT4V](https://huggingface.co/datasets/Lin-Chen/ShareGPT4V), [ALLaVA](https://huggingface.co/datasets/FreedomIntelligence/ALLaVA-4V), [Cambrian-10M](https://huggingface.co/datasets/nyu-visionx/Cambrian-10M) |

### Video Pretraining and Video-Language Data

| Dataset | Scope | Typical use | Access |
| --- | --- | --- | --- |
| WebVid-10M | Video-text metadata | Text-video pretraining and retrieval | [webvid](https://github.com/m-bain/webvid), [HF metadata](https://huggingface.co/datasets/TempoFunk/webvid-10M) |
| InternVid | 7M+ videos, 234M clips, dense captions | Large video-text pretraining | [OpenGVLab/InternVid](https://huggingface.co/datasets/OpenGVLab/InternVid) |
| HowTo100M | Instructional narrated videos | ASR-supervised instructional video pretraining | [HowTo100M](https://www.di.ens.fr/willow/research/howto100m/) |
| HD-VILA-100M | High-resolution video-language clips | High-res video representation learning | [HD-VILA](https://github.com/microsoft/XPretrain/blob/main/hd-vila-100m/README.md) |
| Panda-70M | High-quality video-caption pairs | Video captioning and video-language pretraining | [Panda-70M](https://snap-research.github.io/Panda-70M/) |
| Ego4D / EPIC-KITCHENS-100 | Egocentric videos and narrations | Embodied video, hand-object interaction, planning priors | [Ego4D](https://ego4d-data.org/docs/start-here/), [EPIC-KITCHENS](https://epic-kitchens.github.io/) |
| Something-Something V2 / Kinetics | Human-object and action videos | Temporal and action understanding | [Something-Something V2](https://www.qualcomm.com/developer/software/something-something-v-2-dataset), [Kinetics](https://github.com/cvdfoundation/kinetics-dataset) |
| ActivityNet Captions / MSR-VTT / VATEX | Video captions and retrieval data | Captioning, retrieval, video QA | [ActivityNet](https://activity-net.org/download.html), [MSR-VTT](https://huggingface.co/datasets/friedrichor/MSR-VTT), [VATEX](https://eric-xw.github.io/vatex-website/download.html) |
| YouCook2 / COIN / CrossTask | Instructional videos with steps | Procedure learning and temporal grounding | [YouCook2](http://youcook2.eecs.umich.edu/), [COIN](https://coin-dataset.github.io/), [CrossTask](https://github.com/DmZhukov/CrossTask) |
| TVQA / TGIF-QA / NExT-QA / EgoSchema | Video QA and long-form video reasoning | Video-LLM SFT and evaluation | [TVQA](https://tvqa.cs.unc.edu/), [TGIF-QA](https://github.com/YunseokJANG/tgif-qa), [NExT-QA](https://github.com/doc-doc/NExT-QA), [EgoSchema](https://github.com/egoschema/EgoSchema) |
| VideoInstruct-100K / VideoChatGPT / ShareGPT4Video / LLaVA-Video-178K | Video instruction and QA data | Video LMM instruction training | [VideoInstruct-100K](https://huggingface.co/datasets/MBZUAI/VideoInstruct-100K), [VideoChatGPT](https://huggingface.co/datasets/lmms-lab/VideoChatGPT), [ShareGPT4Video](https://huggingface.co/datasets/ShareGPT4Video/ShareGPT4Video), [LLaVA-Video-178K](https://huggingface.co/datasets/lmms-lab/LLaVA-Video-178K) |
| BDD100K | Driving videos and frames | Car-side video pretraining and perception | [BDD100K](https://bdd-data.berkeley.edu/) |

### VLA, Robotics, and Autonomous Driving Data

| Dataset | Scope | Typical use | Access |
| --- | --- | --- | --- |
| Open X-Embodiment / RT-X | 1M+ robot trajectories across many embodiments | Cross-robot VLA pretraining | [project](https://robotics-transformer-x.github.io/), [GitHub](https://github.com/google-deepmind/open_x_embodiment) |
| RT-1 / Fractal20220817 | Real robot episodes and tasks | RT-1/RT-X-style behavior cloning | [RT-1](https://robotics-transformer1.github.io/), [TFDS](https://www.tensorflow.org/datasets/catalog/fractal20220817_data) |
| OpenVLA OXE setup | Open X training code and recipes | VLA training baseline | [openvla/openvla](https://github.com/openvla/openvla) |
| DROID | 76K demonstrations, 350 hours, 564 scenes | In-the-wild robot manipulation | [DROID](https://droid-dataset.github.io/), [download docs](https://droid-dataset.github.io/droid/the-droid-dataset) |
| BridgeData V2 | 60K robot trajectories | Low-cost multi-task robot learning | [BridgeData V2](https://rail-berkeley.github.io/bridgedata/) |
| RoboNet | 15M frames, 7 robot platforms | Video prediction and robot pretraining | [RoboNet](https://www.robonet.wiki/), [GitHub](https://github.com/SudeepDasari/RoboNet) |
| Language Table / CALVIN / LIBERO | Language-conditioned manipulation benchmarks | VLA fine-tuning and evaluation | [Language Table](https://github.com/google-research/language-table), [CALVIN](https://calvin.cs.uni-freiburg.de/), [LIBERO](https://libero-project.github.io/main.html) |
| RLBench / Meta-World / robomimic | Simulated and HDF5 demonstration benchmarks | Imitation learning and RL baselines | [RLBench](https://github.com/stepjam/RLBench), [Meta-World](https://meta-world.github.io/), [robomimic](https://robomimic.github.io/docs/datasets/overview.html) |
| ManiSkill / BEHAVIOR / Habitat / AI2-THOR / ALFRED / TEACh | Simulation and embodied AI environments | Embodied planning and agent data generation | [ManiSkill](https://maniskill.readthedocs.io/), [BEHAVIOR](https://behavior.stanford.edu/), [Habitat](https://github.com/facebookresearch/habitat-lab), [AI2-THOR](https://ai2thor.allenai.org/), [ALFRED](https://askforalfred.com/), [TEACh](https://github.com/alexa/teach) |
| LeRobot Hub / ALOHA / Mobile ALOHA / PushT | Community robotics datasets and demos | Unified robotics data and bimanual policies | [LeRobot](https://huggingface.co/lerobot), [ALOHA](https://tonyzhaozh.github.io/aloha/), [Mobile ALOHA](https://mobile-aloha.github.io/), [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) |
| nuScenes / nuPlan / Waymo Open Dataset / Argoverse 2 | Autonomous driving perception, prediction, planning data | Driving VLA, planning, closed-loop simulation | [nuScenes](https://www.nuscenes.org/download), [nuPlan](https://www.nuplan.org/nuplan), [Waymo](https://waymo.com/open/), [Argoverse 2](https://argoverse.org/av2.html) |
| KITTI / nuImages / Cityscapes / Mapillary Vistas / PandaSet | Autonomous-driving perception and street-scene data | Car-side vision pretraining and evaluation | [KITTI](https://www.cvlibs.net/datasets/kitti/), [nuImages](https://www.nuscenes.org/nuimages), [Cityscapes](https://www.cityscapes-dataset.com/), [Mapillary Vistas](https://www.mapillary.com/dataset/vistas), [PandaSet](https://pandaset.org/) |

## Recommended Data Onboarding Priority

| Priority | Area | Suggested starting mix |
| --- | --- | --- |
| P0 | LLM base | FineWeb, FineWeb-2, Dolma, RedPajama-V2, DCLM, The Stack v2, OpenWebMath, FineMath, Common Pile |
| P0 | VLM base | DataComp, Re-LAION, CC12M, WIT, OBELICS, MINT-1T, ShareGPT4V, COCO, Visual Genome |
| P0 | Video base | InternVid, HowTo100M, HD-VILA-100M, Panda-70M, Ego4D, EPIC-KITCHENS, ActivityNet Captions |
| P0 | VLA base | Open X-Embodiment, DROID, BridgeData V2, RoboNet, CALVIN, LIBERO, LeRobot, nuPlan, nuScenes, Waymo |
| P1 | Post-training | Tulu 3 SFT, UltraChat, OpenHermes, OpenOrca, OASST, HH-RLHF, UltraFeedback, Nectar, ToolBench, SWE-bench |
| P1 | OCR and documents | DocVQA, TextVQA, ChartQA, AI2D, ScienceQA, InfographicVQA, OCR-VQA |
| P1 | Car-side efficiency and planning | BDD100K, Argoverse 2, KITTI, nuImages, Cityscapes, INTERACTION, Mapillary Vistas |

## Compliance and Quality Gates

Production ingestion should enforce:

- Dataset license allowlist and source-specific terms review.
- URL/media rehydration logs and actual sample availability statistics.
- PII, face, plate, child-safety, NSFW, and unsafe-content filtering.
- Exact and near-duplicate removal.
- Benchmark contamination checks.
- Data lineage: source URL, crawl snapshot, processing version, filtering version, and shard digest.
- Language identification and domain sampling caps.
- OCR/ASR quality filtering.
- VLA schema normalization for action space, proprioception, camera pose, episode boundaries, success labels, failure labels, embodiment IDs, and language instructions.
- Autonomous-driving separation of perception, prediction, planning, and closed-loop simulation data.

## Plugin System

Plugins are discovered from `plugins/*/plugin.json`.

Example:

```json
{
  "name": "example-evaluator",
  "version": "0.1.0",
  "kind": "evaluator",
  "module": "fmops.example_plugins",
  "entrypoint": "build_example_evaluator"
}
```

Validate plugins:

```bash
PYTHONPATH=src python -m fmops.cli plugins
```

## Checkpoint Conversion

Create a conversion manifest:

```bash
PYTHONPATH=src python -m fmops.cli checkpoint-convert \
  --source path/to/training_checkpoint \
  --target artifacts/converted/model \
  --source-format training \
  --target-format inference
```

The core converter writes an audit manifest and can optionally copy files with `--copy-files`. The production wrapper `jobs/checkpoint_convert.py` records serving metadata, tokenizer-bundle verification, safetensors readiness, and runtime targets; use `production-run --area checkpoint --execute` to invoke it through the guarded adapter path.

## Deployment Validation

The default deployment validator checks three target envelopes:

- `server-h100-vllm`
- `edge-orin`
- `cockpit-soc`

Metrics checked:

- prefill/decode latency envelope
- memory budget
- decode tokens per second
- edge power budget, where applicable

Run:

```bash
PYTHONPATH=src python -m fmops.cli deploy-check
```

## CI and Tests

CI is defined in `.github/workflows/ci.yml`.

Local verification:

```bash
make validate
make test
```

Current test coverage includes:

- Config semantic validation.
- Schema validation.
- Dataset catalog validation.
- Model registry.
- All twenty PyTorch architecture families.
- Data, training, evaluation, and deployment runners.
- Checkpoint manifest conversion.
- Experiment tracking.
- Dashboard generation.
- Plugin discovery.
- Production integration planning, preflight checks, guarded non-execution runs, and required area coverage.

## Production Readiness

The production adapter layer is implemented, but it deliberately refuses to submit external jobs until the target environment is configured. A real production run requires the relevant binaries and environment variables reported by `production-check`, such as Spark/Ray data infrastructure, Slurm or another scheduler, model/evaluation endpoints, checkpoint storage, vLLM or TensorRT-LLM serving tools, and simulator registries.

Recommended promotion flow:

```bash
make validate
make test
make production-plan
make production-check
FMOPS_ALLOW_PRODUCTION_EXECUTE=1 PYTHONPATH=src python -m fmops.cli production-run --execute
```

Use `--area data`, `--area training`, `--area evaluation`, `--area checkpoint`, `--area deployment`, `--area monitoring`, or `--area governance` to roll out one subsystem at a time.
