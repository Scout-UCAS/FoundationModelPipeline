# Foundation Model Program Plan

Validation status: **PASS**

## Data System

- Target scale: 2500 TB
- Planned scale: 2835 TB
- Languages: 25
- Modalities: audio_speech, multimodal, pure_text, video_pretraining, vla

| Modality | Approx TB |
| --- | ---: |
| audio_speech | 125.0 |
| multimodal | 690.0 |
| pure_text | 1175.0 |
| video_pretraining | 515.0 |
| vla | 330.0 |

| Mixture Stage | Objective | TB | Modality Split | Language Split |
| --- | --- | ---: | --- | --- |
| warmup_2k | stabilize tokenizer, optimizer, and early curriculum | 150 | audio_speech 3.0%, multimodal 16.0%, pure_text 69.0%, video_pretraining 8.0%, vla 4.0% | en 38.0%, other 38.0%, zh 24.0% |
| core_pretrain | maximize broad language, code, knowledge, and multimodal coverage | 1700 | audio_speech 4.0%, multimodal 22.0%, pure_text 54.0%, video_pretraining 15.0%, vla 5.0% | en 35.0%, other 41.0%, zh 24.0% |
| multimodal_video_expansion | improve temporal understanding and image-video grounding | 450 | audio_speech 4.0%, multimodal 31.0%, pure_text 20.0%, video_pretraining 37.0%, vla 8.0% | en 34.0%, other 40.0%, zh 26.0% |
| vla_alignment | bind perception, language instruction, and action policy traces | 120 | audio_speech 5.0%, multimodal 19.0%, pure_text 8.0%, video_pretraining 24.0%, vla 44.0% | en 40.0%, other 25.0%, zh 35.0% |
| reasoning_final | raise high-quality reasoning, tool-use, and instruction-following density | 80 | audio_speech 5.0%, multimodal 17.0%, pure_text 59.0%, video_pretraining 8.0%, vla 11.0% | en 42.0%, other 28.0%, zh 30.0% |

## Architecture Experiments

- Unified tokenizer: fm-unified-256k
- Training budget: 300B tokens
- Context length: 32768
- Hardware budget: 400 GPUs

| Rank | Candidate | Family | Active/Total Params | Loss | Tok/s/GPU | Reasoning | Memory GB | Stability | Utility |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | mla-gqa-kv-compressed | MLA / KV-Compressed Attention | 75.0B/75.0B | 1.940 | 2960 | 66.4 | 58.0 | 0.87 | 0.733 |
| 2 | mamba-selective-scan | SSM / Selective Scan | 72.0B/72.0B | 2.010 | 3380 | 62.1 | 51.0 | 0.84 | 0.648 |
| 3 | moe-top2-64e | MoE | 45.0B/320.0B | 1.910 | 2320 | 66.8 | 72.0 | 0.88 | 0.642 |
| 4 | hybrid-mamba-transformer | Hybrid Architecture | 86.0B/86.0B | 1.960 | 2740 | 65.2 | 65.0 | 0.87 | 0.641 |
| 5 | mtp-4head-decoder | MTP | 74.0B/74.0B | 1.930 | 2470 | 66.1 | 70.0 | 0.86 | 0.615 |
| 6 | linear-sparse-attn-32k | Sparse / Linear Attention | 72.0B/72.0B | 1.980 | 2860 | 63.5 | 62.0 | 0.84 | 0.587 |
| 7 | mixture-of-depths-token-routing | Mixture-of-Depths | 58.0B/82.0B | 1.970 | 2720 | 66.9 | 64.0 | 0.81 | 0.583 |
| 8 | retnet-retention-decoder | Retention / RetNet | 72.0B/72.0B | 2.030 | 3240 | 61.4 | 53.0 | 0.83 | 0.581 |
| 9 | reasoning-native-verifier-policy | Reasoning-native Architecture | 90.0B/90.0B | 1.940 | 2130 | 72.4 | 79.0 | 0.82 | 0.570 |
| 10 | latent-reasoning-scratchpad | Latent Reasoning | 78.0B/78.0B | 1.950 | 2210 | 69.6 | 74.0 | 0.83 | 0.562 |
| 11 | hyena-long-conv | Long Convolution | 70.0B/70.0B | 2.050 | 3310 | 60.8 | 50.0 | 0.82 | 0.562 |
| 12 | rwkv-style-rnn-backbone | RNN-like Backbone | 70.0B/70.0B | 2.060 | 3150 | 59.4 | 54.0 | 0.82 | 0.495 |
| 13 | test-time-memory-adapter | Test-Time Memory | 80.0B/80.0B | 1.980 | 2240 | 67.5 | 73.0 | 0.80 | 0.473 |
| 14 | latent-world-model-jepa | JEPA / Latent World Model | 84.0B/84.0B | 2.020 | 2360 | 64.6 | 70.0 | 0.83 | 0.458 |
| 15 | omni-modal-unified-tokens | Omni-modal Architecture | 96.0B/96.0B | 1.970 | 2060 | 68.9 | 80.0 | 0.80 | 0.453 |
| 16 | memory-augmented-retrieval-state | Memory-augmented LLM | 82.0B/82.0B | 1.990 | 2180 | 67.7 | 78.0 | 0.81 | 0.452 |
| 17 | byte-level-token-free-decoder | Token-free Byte-level LLM | 76.0B/76.0B | 2.080 | 2580 | 62.7 | 68.0 | 0.84 | 0.422 |
| 18 | robotics-transformer-vla-policy | VLA / Robotics Transformer | 88.0B/88.0B | 1.990 | 2060 | 65.8 | 78.0 | 0.79 | 0.383 |
| 19 | spiking-event-backbone | Neuromorphic / Spiking Backbone | 48.0B/48.0B | 2.180 | 3540 | 55.6 | 42.0 | 0.76 | 0.350 |
| 20 | diffusion-llm-dllm | dLLM | 68.0B/68.0B | 2.110 | 1980 | 61.2 | 76.0 | 0.79 | 0.201 |

## Training Pipeline

- Hardware: 50 nodes x 8 H100-80GB GPUs = 400 GPUs
- Interconnect: 400G InfiniBand with hierarchical all-reduce
- Storage: NVMe cache + object-store shards, 1.2 TB/s aggregate read target

| Stage | Objective | Data | Framework | Parallelism | Gates |
| --- | --- | --- | --- | --- | --- |
| Pre-training | train base model on 2500T+ staged mixture, including audio-speech alignment shards | core_pretrain | Megatron-DeepSpeed | context=1, data=10, pipeline=4, tensor=8 | max_loss_spike_ratio<=1.35, max_nan_incidents<=0, min_gpu_utilization>=0.88 |
| SFT | align instruction following, multilingual dialogue, multimodal QA, spoken interaction, and VLA commands | reasoning_final | FSDP | context=1, data=50, pipeline=2, tensor=4 | max_eval_regression<=0.02, min_instruction_win_rate>=0.58 |
| RL | optimize reasoning, preference, safety, speech interaction, and tool-use rewards | rl_reasoning_tool_mix | verl-compatible PPO/GRPO | context=1, data=50, pipeline=2, tensor=4 | max_kl<=0.12, min_reward_score>=0.62 |
| Agentic RL | train multi-step tool, browser, code, and vehicle-agent behavior | agentic_env_rollouts | asynchronous actor-learner | context=1, data=100, pipeline=2, tensor=2 | max_safety_violation<=0.01, max_tool_error_rate<=0.08, min_task_success>=0.55 |

| Stage | Launch Command |
| --- | --- |
| Pre-training | `torchrun --nnodes=50 --nproc_per_node=8 train/pretrain.py --mode external --mixture core_pretrain --world-size 400 --output artifacts/production/training/pretraining_rank0.json` |
| SFT | `torchrun --nnodes=50 --nproc_per_node=8 train/sft.py --mode external --mixture reasoning_final --world-size 400 --output artifacts/production/training/sft_rank0.json` |
| RL | `torchrun --nnodes=50 --nproc_per_node=8 train/rl.py --mode external --mixture rl_reasoning_tool_mix --world-size 400 --output artifacts/production/training/rl_rank0.json` |
| Agentic RL | `torchrun --nnodes=50 --nproc_per_node=8 train/agentic_rl.py --mode external --mixture agentic_env_rollouts --world-size 400 --output artifacts/production/training/agentic_rl_rank0.json` |

## Evaluation Suite

| Dimension | Benchmark | Datasets | Metrics | Weight | Gate | Cadence |
| --- | --- | --- | --- | ---: | --- | --- |
| general | general_multilingual_core | MMLU, MMLU-Pro, BBH, ARC-Challenge, HellaSwag, TruthfulQA, CMMLU, C-Eval, CLUE, TyDiQA, FLORES-200, Belebele, MGSM | accuracy, macro_average, calibration_error | 1.00 | macro_average>=0.72 | daily-and-release |
| reasoning | reasoning_math_code | GSM8K, MATH, MATH-500, AIME-style private holdout, GPQA, OlympiadBench, HumanEval, MBPP, LiveCodeBench, APPS, CodeContests, CRUXEval, SciCode, SWE-bench Verified, formal-proof-mini | pass_at_1, consensus_pass, verified_solution_rate | 1.40 | pass_at_1>=0.48 | daily-and-release |
| multimodal_understanding | multimodal_understanding_core | MMMU, MMMU-Pro, MMBench, MMStar, SEED-Bench, MM-Vet, MathVista, ScienceQA, AI2D, VQAv2, GQA, OK-VQA, A-OKVQA, VizWiz, TextVQA, DocVQA, ChartQA, InfoVQA, OCRBench, Video-MME, MVBench | accuracy, ocr_f1, temporal_reasoning | 1.20 | accuracy>=0.66 | daily-and-release |
| audio_speech | audio_speech_core | LibriSpeech ASR, Common Voice ASR, FLEURS, CoVoST 2, SUPERB, HEAR, Speech Commands, AudioSet, AudioCaps, Clotho, VoxCeleb, DNS Challenge, LibriTTS, private-spoken-dialogue-holdout | wer, cer, bleu, accuracy, caption_cider, clap_score, speaker_eer, der, mos | 0.90 | accuracy>=0.75, cer<=0.12, wer<=0.18 | daily-and-release |
| vla | vla_action_policy | LIBERO, CALVIN, RLBench, Language Table, Meta-World, ManiSkill, robomimic, RoboCasa, SimplerEnv, Open X-Embodiment heldout, robot-sim-heldout, nuPlan, CARLA Leaderboard, Bench2Drive, NAVSIM, driving-action-heldout | success_rate, collision_free_rate, action_l2, recovery_rate | 1.20 | success_rate>=0.58 | release |
| long_context | long_context_retrieval_reasoning | LongBench, LongBench v2, RULER, InfiniteBench, Needle-in-a-Haystack, BABILong, L-Eval, Lost-in-the-Middle, NarrativeQA, Qasper, HotpotQA, 2WikiMultiHopQA, MuSiQue, QMSum, GovReport, RepoBench, LongVideoBench, MLVU | exact_match, citation_precision, long_range_consistency | 0.90 | exact_match>=0.7 | daily-and-release |
| tool_calling | tool_calling_reliability | Berkeley Function Calling Leaderboard, Gorilla OpenFunctions, ToolBench, StableToolBench, API-Bank, ToolAlpaca, ToolQA, tau-bench, function-call-json, sql-tool-use, code-interpreter-tasks, retrieval-tool-use | schema_validity, task_success, tool_error_rate | 0.80 | schema_validity>=0.97 | daily-and-release |
| agent | agent_multi_step | WebArena, VisualWebArena, MiniWoB++, Mind2Web, OSWorld, WorkArena, BrowserGym, GAIA, AgentBench, AppWorld, WebVoyager, SWE-bench, SWE-bench Verified, web-agent-mini, coding-agent-mini, workflow-agent-suite, vehicle-copilot-sim | task_success, steps_to_success, unsafe_action_rate, cost_per_success | 1.10 | task_success>=0.5 | release |
| edge_efficiency | car_side_deployment_efficiency | MLPerf Inference Datacenter, MLPerf Inference Edge, MLPerf Tiny, GenAI-Perf, AIPerf, vLLM benchmark, TensorRT-LLM benchmark, LLMPerf, orin-replay, qualcomm-cockpit-replay, streaming-video-vla, CARLA closed-loop replay | prefill_ms, decode_tok_s, memory_gb, power_w, frame_action_latency_ms | 0.90 | decode_tok_s>=24 | release |

| Dimension | Normalized Weight |
| --- | ---: |
| agent | 0.1170 |
| audio_speech | 0.0957 |
| edge_efficiency | 0.0957 |
| general | 0.1064 |
| long_context | 0.0957 |
| multimodal_understanding | 0.1277 |
| reasoning | 0.1489 |
| tool_calling | 0.0851 |
| vla | 0.1277 |

## Production Integration

- Artifact root: `artifacts/production`
- Execution guard: `FMOPS_ALLOW_PRODUCTION_EXECUTE`
- Tasks: 15
- Release gates: 9

| Area | Task | Adapter | Owner | Required Binaries | Required Env |
| --- | --- | --- | --- | --- | --- |
| data | ingest_open_new_business_sources | spark-data-lake | data-platform | spark-submit | FMOPS_DATA_LAKE_URI, FMOPS_METADATA_STORE_URI |
| data | clean_dedup_cluster_quality | ray-quality-dedup | data-quality | ray | FMOPS_DATA_LAKE_URI, FMOPS_QUALITY_MODEL_URI, FMOPS_DEDUP_INDEX_URI |
| data | materialize_multistage_mixtures | spark-shard-writer | data-platform | spark-submit | FMOPS_DATA_LAKE_URI, FMOPS_SHARD_STORE_URI |
| training | submit_pretraining_400gpu | slurm-megatron-deepspeed | training-platform | sbatch | FMOPS_CLUSTER_ACCOUNT, FMOPS_CONTAINER_IMAGE, FMOPS_CHECKPOINT_ROOT, FMOPS_PRETRAIN_BACKEND_COMMAND |
| training | submit_sft_400gpu | slurm-fsdp | training-platform | sbatch | FMOPS_CLUSTER_ACCOUNT, FMOPS_CONTAINER_IMAGE, FMOPS_CHECKPOINT_ROOT, FMOPS_SFT_BACKEND_COMMAND |
| training | submit_rl_400gpu | slurm-verl-rl | rl-platform | sbatch | FMOPS_CLUSTER_ACCOUNT, FMOPS_CONTAINER_IMAGE, FMOPS_CHECKPOINT_ROOT, FMOPS_REWARD_MODEL_URI, FMOPS_RL_BACKEND_COMMAND |
| training | submit_agentic_rl_400gpu | slurm-actor-learner | agent-platform | sbatch | FMOPS_CLUSTER_ACCOUNT, FMOPS_CONTAINER_IMAGE, FMOPS_CHECKPOINT_ROOT, FMOPS_AGENT_ENV_REGISTRY, FMOPS_AGENTIC_RL_BACKEND_COMMAND |
| evaluation | run_text_reasoning_eval | lm-eval | evaluation-platform | lm_eval | FMOPS_MODEL_ID, FMOPS_MODEL_ENDPOINT |
| evaluation | run_multimodal_video_eval | opencompass-vlmevalkit | evaluation-platform | vlm_eval | FMOPS_MODEL_ID, FMOPS_MODEL_ENDPOINT, FMOPS_VLM_DATA_ROOT |
| evaluation | run_audio_speech_eval | espnet-speechbrain-nemo | evaluation-platform | python | FMOPS_MODEL_ID, FMOPS_MODEL_ENDPOINT, FMOPS_AUDIO_EVAL_DATA_ROOT |
| evaluation | run_vla_agent_eval | simulator-agent-eval | evaluation-platform | python | FMOPS_MODEL_ID, FMOPS_SIMULATOR_REGISTRY, FMOPS_AGENT_ENV_REGISTRY |
| checkpoint | convert_checkpoint_for_serving | megatron-hf-safetensors | model-platform | python | FMOPS_CHECKPOINT_SOURCE, FMOPS_CHECKPOINT_TARGET |
| deployment | run_server_and_edge_deployment_validation | vllm-tensorrt-llm-genai-perf | serving-platform | vllm, genai-perf | FMOPS_MODEL_PATH, FMOPS_DEPLOYMENT_TARGETS |
| monitoring | export_monitoring_and_alerting | prometheus-grafana | observability | python | - |
| governance | enforce_release_gates | release-gate | release | python | - |

| Release Gate | Source | Owner |
| --- | --- | --- |
| data_license_and_privacy_approved | `artifacts/production/data/quality_report.json` | data-governance |
| data_scale_and_modalities_met | `artifacts/production/data/ingestion_manifest.json` | data-platform |
| training_stability_met | `artifacts/production/training/pretraining_launch.json` | training-platform |
| checkpoint_conversion_verified | `artifacts/production/checkpoint/conversion_manifest.json` | model-platform |
| core_evaluation_passed | `artifacts/production/evaluation/text_reasoning_eval.json` | evaluation-platform |
| multimodal_vla_evaluation_passed | `artifacts/production/evaluation/vla_agent_eval.json` | evaluation-platform |
| audio_speech_evaluation_passed | `artifacts/production/evaluation/audio_speech_eval.json` | evaluation-platform |
| deployment_envelope_passed | `artifacts/production/deployment/deployment_report.json` | serving-platform |
| safety_and_governance_signoff | `artifacts/production/governance/release_gate_report.json` | release |
