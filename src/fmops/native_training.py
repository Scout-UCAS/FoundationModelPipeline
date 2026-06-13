from __future__ import annotations

import copy
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .architecture_impl import AutoregressiveLM, ModelConfig


STAGE_TO_KIND = {
    "Pre-training": "pretrain",
    "SFT": "sft",
    "RL": "rl",
    "Agentic RL": "agentic_rl",
}


@dataclass(frozen=True)
class NativeTrainingConfig:
    stage: str
    mixture: str
    output_dir: str
    data_path: str | None = None
    max_steps: int = 10
    batch_size: int = 2
    seq_len: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    seed: int = 7
    device: str = "auto"
    vocab_size: int = 260
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 256
    save_every: int = 0
    use_ddp: bool = True


@dataclass(frozen=True)
class NativeTrainingResult:
    stage: str
    mode: str
    status: str
    output_dir: str
    checkpoint_path: str
    metrics_path: str
    steps: int
    final_loss: float
    tokens_seen: int
    duration_seconds: float
    device: str
    world_size: int


class ByteTokenizer:
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2
    offset = 3
    vocab_size = 260

    def encode(self, text: str, seq_len: int) -> list[int]:
        raw = list(text.encode("utf-8", errors="replace"))
        tokens = [self.bos_token_id] + [min(byte, 255) + self.offset for byte in raw] + [self.eos_token_id]
        if len(tokens) < seq_len:
            tokens.extend([self.pad_token_id] * (seq_len - len(tokens)))
        return tokens[:seq_len]


def _extract_text(payload: dict[str, Any]) -> str:
    if "text" in payload:
        return str(payload["text"])
    if "prompt" in payload and "response" in payload:
        return f"{payload['prompt']}\n{payload['response']}"
    if "instruction" in payload and "output" in payload:
        return f"{payload['instruction']}\n{payload.get('input', '')}\n{payload['output']}"
    if "messages" in payload and isinstance(payload["messages"], list):
        parts = []
        for message in payload["messages"]:
            if isinstance(message, dict):
                parts.append(f"{message.get('role', 'user')}: {message.get('content', '')}")
        return "\n".join(parts)
    if "trajectory" in payload:
        return json.dumps(payload["trajectory"], ensure_ascii=False, sort_keys=True)
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _load_records(path: str | Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"training data file not found: {data_path}")
    records: list[dict[str, Any]] = []
    with data_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                value = json.loads(stripped)
                if isinstance(value, dict):
                    records.append(value)
                else:
                    records.append({"text": str(value)})
            except json.JSONDecodeError:
                records.append({"text": stripped})
    return records


def _synthetic_records(stage: str, count: int) -> list[dict[str, Any]]:
    records = []
    for index in range(count):
        if stage == "Pre-training":
            records.append({"text": f"foundation model pretraining sample {index} with multilingual knowledge and code."})
        elif stage == "SFT":
            records.append({"prompt": f"Explain task {index}.", "response": f"Step by step answer for task {index}."})
        elif stage == "RL":
            records.append({"prompt": f"Solve reasoning problem {index}.", "response": f"answer {index}", "reward": 1.0 if index % 2 == 0 else 0.2})
        else:
            records.append(
                {
                    "trajectory": {
                        "instruction": f"complete agent workflow {index}",
                        "tools": ["search", "code", "vehicle_sim"],
                        "actions": ["plan", "call_tool", "verify", "finish"],
                    },
                    "reward": 1.0 if index % 3 != 0 else 0.1,
                    "success": index % 3 != 0,
                }
            )
    return records


class TextRecordDataset(Dataset[dict[str, Tensor]]):
    def __init__(self, records: list[dict[str, Any]], stage: str, seq_len: int, tokenizer: ByteTokenizer) -> None:
        self.records = records
        self.stage = stage
        self.seq_len = seq_len
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        payload = self.records[index % len(self.records)]
        text = _extract_text(payload)
        input_ids = torch.tensor(self.tokenizer.encode(text, self.seq_len), dtype=torch.long)
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        reward = float(payload.get("reward", 1.0 if payload.get("success", True) else 0.0))
        return {
            "input_ids": input_ids,
            "labels": labels,
            "rewards": torch.tensor(reward, dtype=torch.float32),
        }


def _cycle(loader: Iterable[dict[str, Tensor]]) -> Iterable[dict[str, Tensor]]:
    while True:
        for batch in loader:
            yield batch


def _select_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _init_distributed_if_needed(use_ddp: bool) -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if use_ddp and world_size > 1 and not torch.distributed.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch.distributed.init_process_group(backend=backend)
    return world_size, rank, local_rank


def _cleanup_distributed() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def _move_batch(batch: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _response_logprobs(logits: Tensor, labels: Tensor) -> Tensor:
    target = labels[:, 1:]
    mask = target.ne(-100)
    safe_target = target.masked_fill(~mask, 0)
    log_probs = F.log_softmax(logits[:, :-1], dim=-1)
    gathered = log_probs.gather(-1, safe_target.unsqueeze(-1)).squeeze(-1)
    return (gathered * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)


def _rl_loss(model: nn.Module, reference_model: nn.Module, batch: dict[str, Tensor], kl_coef: float = 0.02) -> tuple[Tensor, dict[str, float]]:
    output = model(batch["input_ids"], labels=batch["labels"])
    logits = output["logits"]
    with torch.no_grad():
        ref_logits = reference_model(batch["input_ids"])["logits"]
    rewards = batch["rewards"]
    advantages = rewards - rewards.mean()
    logp = _response_logprobs(logits, batch["labels"])
    ref_logp = _response_logprobs(ref_logits, batch["labels"])
    policy_loss = -(advantages.detach() * logp).mean()
    kl = (logp - ref_logp).pow(2).mean()
    supervised = output.get("loss", torch.zeros((), device=logits.device))
    loss = policy_loss + kl_coef * kl + 0.2 * supervised
    return loss, {
        "policy_loss": float(policy_loss.detach().cpu()),
        "kl": float(kl.detach().cpu()),
        "mean_reward": float(rewards.mean().detach().cpu()),
        "supervised_loss": float(supervised.detach().cpu()),
    }


def _stage_loss(
    stage: str,
    model: nn.Module,
    batch: dict[str, Tensor],
    reference_model: nn.Module | None,
) -> tuple[Tensor, dict[str, float]]:
    if stage in {"Pre-training", "SFT"}:
        output = model(batch["input_ids"], labels=batch["labels"])
        loss = output["loss"]
        return loss, {"lm_loss": float(loss.detach().cpu())}
    if reference_model is None:
        raise ValueError("RL stages require a reference model")
    return _rl_loss(model, reference_model, batch, kl_coef=0.02 if stage == "RL" else 0.01)


def _save_checkpoint(
    output_dir: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: NativeTrainingConfig,
    metrics: list[dict[str, Any]],
) -> Path:
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"step_{step:06d}.pt"
    module = model.module if hasattr(model, "module") else model
    torch.save(
        {
            "step": step,
            "model_state_dict": module.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": asdict(config),
            "metrics": metrics,
        },
        path,
    )
    return path


def run_native_training(config: NativeTrainingConfig) -> NativeTrainingResult:
    if config.stage not in STAGE_TO_KIND:
        raise ValueError(f"unsupported stage for native training: {config.stage}")
    if config.max_steps < 1:
        raise ValueError("max_steps must be positive")
    if config.vocab_size < ByteTokenizer.vocab_size:
        raise ValueError(f"vocab_size must be >= {ByteTokenizer.vocab_size}")

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    world_size, rank, local_rank = _init_distributed_if_needed(config.use_ddp)
    device = _select_device(config.device)
    if device.type == "cuda":
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)

    try:
        tokenizer = ByteTokenizer()
        records = _load_records(config.data_path)
        if not records:
            records = _synthetic_records(config.stage, max(config.batch_size * 8, 16))
        dataset = TextRecordDataset(records, config.stage, config.seq_len, tokenizer)
        loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=False)
        batches = _cycle(loader)

        model_config = ModelConfig(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            d_ff=config.d_ff,
            max_seq_len=config.seq_len,
            dropout=0.0,
        )
        model: nn.Module = AutoregressiveLM(model_config).to(device)
        reference_model: nn.Module | None = None
        if config.stage in {"RL", "Agentic RL"}:
            reference_model = copy.deepcopy(model).eval().requires_grad_(False).to(device)
        if world_size > 1:
            model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank] if device.type == "cuda" else None)

        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        metrics: list[dict[str, Any]] = []
        tokens_seen = 0
        last_loss = math.nan
        last_checkpoint = output_dir / "checkpoints" / "not_saved.pt"
        started = time.monotonic()
        for step in range(1, config.max_steps + 1):
            batch = _move_batch(next(batches), device)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss, extra = _stage_loss(config.stage, model, batch, reference_model)
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip).detach().cpu())
            optimizer.step()
            token_count = int(batch["input_ids"].numel()) * world_size
            tokens_seen += token_count
            last_loss = float(loss.detach().cpu())
            metrics.append(
                {
                    "step": step,
                    "loss": last_loss,
                    "grad_norm": grad_norm,
                    "tokens": token_count,
                    **extra,
                }
            )
            if rank == 0 and config.save_every > 0 and step % config.save_every == 0:
                last_checkpoint = _save_checkpoint(output_dir, model, optimizer, step, config, metrics)

        duration = round(time.monotonic() - started, 3)
        if rank == 0:
            last_checkpoint = _save_checkpoint(output_dir, model, optimizer, config.max_steps, config, metrics)
            metrics_path = output_dir / "metrics.jsonl"
            with metrics_path.open("w", encoding="utf-8") as handle:
                for item in metrics:
                    handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            result = NativeTrainingResult(
                stage=config.stage,
                mode="native",
                status="succeeded",
                output_dir=str(output_dir),
                checkpoint_path=str(last_checkpoint),
                metrics_path=str(metrics_path),
                steps=config.max_steps,
                final_loss=last_loss,
                tokens_seen=tokens_seen,
                duration_seconds=duration,
                device=str(device),
                world_size=world_size,
            )
            (output_dir / "trainer_state.json").write_text(
                json.dumps(asdict(result), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            return result

        return NativeTrainingResult(
            stage=config.stage,
            mode="native",
            status="succeeded",
            output_dir=str(output_dir),
            checkpoint_path=str(last_checkpoint),
            metrics_path=str(output_dir / "metrics.jsonl"),
            steps=config.max_steps,
            final_loss=last_loss,
            tokens_seen=tokens_seen,
            duration_seconds=round(time.monotonic() - started, 3),
            device=str(device),
            world_size=world_size,
        )
    finally:
        _cleanup_distributed()


def native_config_from_options(stage: str, mixture: str, options: dict[str, Any] | None) -> NativeTrainingConfig:
    values = dict(options or {})
    output_dir = values.get("output_dir") or str(Path(str(values.get("output", "artifacts/native_training"))).with_suffix(""))
    return NativeTrainingConfig(
        stage=stage,
        mixture=mixture,
        output_dir=str(output_dir),
        data_path=values.get("data_path"),
        max_steps=int(values.get("max_steps", 10)),
        batch_size=int(values.get("batch_size", 2)),
        seq_len=int(values.get("seq_len", 64)),
        learning_rate=float(values.get("learning_rate", 3e-4)),
        weight_decay=float(values.get("weight_decay", 0.01)),
        grad_clip=float(values.get("grad_clip", 1.0)),
        seed=int(values.get("seed", 7)),
        device=str(values.get("device", "auto")),
        vocab_size=int(values.get("vocab_size", 260)),
        d_model=int(values.get("d_model", 128)),
        n_heads=int(values.get("n_heads", 4)),
        n_layers=int(values.get("n_layers", 2)),
        d_ff=int(values.get("d_ff", 256)),
        save_every=int(values.get("save_every", 0)),
        use_ddp=bool(values.get("use_ddp", True)),
    )


def add_native_training_arguments(parser: Any) -> None:
    parser.add_argument("--data-path", help="Optional JSONL/text training data path")
    parser.add_argument("--output-dir", help="Directory for native trainer checkpoints and metrics")
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--vocab-size", type=int, default=260)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--no-ddp", action="store_true", help="Disable native DDP initialization")


def native_options_from_args(args: Any) -> dict[str, Any]:
    return {
        "data_path": args.data_path,
        "output_dir": args.output_dir,
        "output": args.output,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "seed": args.seed,
        "device": args.device,
        "vocab_size": args.vocab_size,
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "d_ff": args.d_ff,
        "save_every": args.save_every,
        "use_ddp": not args.no_ddp,
    }
