from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 32000
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 4
    d_ff: int = 2048
    max_seq_len: int = 2048
    dropout: float = 0.0


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight


class SwiGLUFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.w_in = nn.Linear(d_model, 2 * d_ff, bias=False)
        self.w_out = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        gate, value = self.w_in(x).chunk(2, dim=-1)
        return self.w_out(self.dropout(F.silu(gate) * value))


class SelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, causal: bool = True) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim**-0.5
        self.causal = causal
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        batch, seq_len, _ = x.shape
        qkv = self.qkv(x).view(batch, seq_len, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if self.causal:
            causal_mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device).tril()
            scores = scores.masked_fill(~causal_mask, torch.finfo(scores.dtype).min)
        if attention_mask is not None:
            mask = attention_mask[:, None, None, :].to(torch.bool)
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

        attn = F.softmax(scores.float(), dim=-1).to(dtype=x.dtype)
        attn = self.dropout(attn)
        y = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        return self.out(y)


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig, causal: bool = True) -> None:
        super().__init__()
        self.norm_attn = RMSNorm(config.d_model)
        self.attn = SelfAttention(config.d_model, config.n_heads, config.dropout, causal=causal)
        self.norm_ffn = RMSNorm(config.d_model)
        self.ffn = SwiGLUFeedForward(config.d_model, config.d_ff, config.dropout)

    def forward(self, x: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        x = x + self.attn(self.norm_attn(x), attention_mask=attention_mask)
        return x + self.ffn(self.norm_ffn(x))


class DecoderBackbone(nn.Module):
    def __init__(self, config: ModelConfig, causal: bool = True) -> None:
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.blocks = nn.ModuleList(TransformerBlock(config, causal=causal) for _ in range(config.n_layers))
        self.norm = RMSNorm(config.d_model)

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        batch, seq_len = input_ids.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError(f"sequence length {seq_len} exceeds max_seq_len {self.config.max_seq_len}")
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch, seq_len)
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        for block in self.blocks:
            x = block(x, attention_mask=attention_mask)
        return self.norm(x)


class AutoregressiveLM(nn.Module):
    def __init__(self, config: ModelConfig, causal: bool = True) -> None:
        super().__init__()
        self.backbone = DecoderBackbone(config, causal=causal)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(self, input_ids: Tensor, labels: Tensor | None = None) -> dict[str, Tensor]:
        hidden = self.backbone(input_ids)
        logits = self.lm_head(hidden)
        output: dict[str, Tensor] = {"logits": logits, "hidden": hidden}
        if labels is not None:
            output["loss"] = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return output


class TopKRouter(nn.Module):
    def __init__(self, d_model: int, num_experts: int, top_k: int = 2) -> None:
        super().__init__()
        if top_k < 1 or top_k > num_experts:
            raise ValueError("top_k must be in [1, num_experts]")
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        logits = self.router(x)
        probs = F.softmax(logits.float(), dim=-1).to(dtype=x.dtype)
        weights, indices = torch.topk(probs, self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        density_proxy = probs.mean(dim=0)
        selected = F.one_hot(indices, num_classes=self.num_experts).to(dtype=probs.dtype).sum(dim=1)
        density = selected.mean(dim=0) / self.top_k
        aux_loss = self.num_experts * torch.sum(density_proxy * density)
        return indices, weights, aux_loss


class MoEFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, num_experts: int, top_k: int = 2, dropout: float = 0.0) -> None:
        super().__init__()
        self.router = TopKRouter(d_model, num_experts, top_k=top_k)
        self.experts = nn.ModuleList(
            SwiGLUFeedForward(d_model, d_ff, dropout=dropout) for _ in range(num_experts)
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        batch, seq_len, d_model = x.shape
        flat = x.reshape(batch * seq_len, d_model)
        indices, weights, aux_loss = self.router(flat)
        out = torch.zeros_like(flat)

        for slot in range(indices.size(1)):
            slot_indices = indices[:, slot]
            slot_weights = weights[:, slot]
            for expert_id, expert in enumerate(self.experts):
                mask = slot_indices == expert_id
                if mask.any():
                    out[mask] += expert(flat[mask]) * slot_weights[mask, None]
        return out.view(batch, seq_len, d_model), aux_loss


class MoETransformerBlock(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        num_experts: int = 8,
        top_k: int = 2,
        causal: bool = True,
    ) -> None:
        super().__init__()
        self.norm_attn = RMSNorm(config.d_model)
        self.attn = SelfAttention(config.d_model, config.n_heads, config.dropout, causal=causal)
        self.norm_moe = RMSNorm(config.d_model)
        self.moe = MoEFeedForward(config.d_model, config.d_ff, num_experts, top_k=top_k, dropout=config.dropout)

    def forward(self, x: Tensor, attention_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        x = x + self.attn(self.norm_attn(x), attention_mask=attention_mask)
        moe_out, aux_loss = self.moe(self.norm_moe(x))
        return x + moe_out, aux_loss


class SlidingWindowAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, window_size: int = 128, dropout: float = 0.0) -> None:
        super().__init__()
        if window_size < 1:
            raise ValueError("window_size must be positive")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim**-0.5
        self.window_size = window_size
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        batch, seq_len, _ = x.shape
        qkv = self.qkv(x).view(batch, seq_len, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        positions = torch.arange(seq_len, device=x.device)
        distance = positions[:, None] - positions[None, :]
        mask = (distance >= 0) & (distance < self.window_size)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        attn = F.softmax(scores.float(), dim=-1).to(dtype=x.dtype)
        attn = self.dropout(attn)
        y = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        return self.out(y)


class CausalLinearAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, eps: float = 1e-6) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        batch, seq_len, _ = x.shape
        qkv = self.qkv(x).view(batch, seq_len, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = F.elu(q.transpose(1, 2)) + 1.0
        k = F.elu(k.transpose(1, 2)) + 1.0
        v = v.transpose(1, 2)

        kv = torch.einsum("bhtd,bhte->bhtde", k, v)
        kv_prefix = kv.cumsum(dim=2)
        k_prefix = k.cumsum(dim=2)
        numerator = torch.einsum("bhtd,bhtde->bhte", q, kv_prefix)
        denominator = torch.einsum("bhtd,bhtd->bht", q, k_prefix).unsqueeze(-1).clamp_min(self.eps)
        y = numerator / denominator
        y = self.dropout(y).transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        return self.out(y)


class SparseLinearAttentionBlock(nn.Module):
    def __init__(self, config: ModelConfig, window_size: int = 128) -> None:
        super().__init__()
        self.norm = RMSNorm(config.d_model)
        self.local = SlidingWindowAttention(config.d_model, config.n_heads, window_size, config.dropout)
        self.linear = CausalLinearAttention(config.d_model, config.n_heads, config.dropout)
        self.mix_logit = nn.Parameter(torch.tensor(0.0))
        self.norm_ffn = RMSNorm(config.d_model)
        self.ffn = SwiGLUFeedForward(config.d_model, config.d_ff, config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        z = self.norm(x)
        gate = torch.sigmoid(self.mix_logit)
        x = x + gate * self.local(z) + (1.0 - gate) * self.linear(z)
        return x + self.ffn(self.norm_ffn(x))


class RecurrentTokenMixer(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.in_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.time_decay = nn.Parameter(torch.full((d_model,), -2.0))
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: Tensor, initial_state: Tensor | None = None) -> tuple[Tensor, Tensor]:
        batch, seq_len, d_model = x.shape
        state = x.new_zeros(batch, d_model) if initial_state is None else initial_state
        outputs = []
        gate, candidate, decay = self.in_proj(x).chunk(3, dim=-1)
        for step in range(seq_len):
            step_decay = torch.sigmoid(decay[:, step] + self.time_decay)
            step_candidate = torch.tanh(candidate[:, step])
            state = step_decay * state + (1.0 - step_decay) * step_candidate
            outputs.append(torch.sigmoid(gate[:, step]) * state)
        y = torch.stack(outputs, dim=1)
        return self.out_proj(y), state


class RNNBackboneBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.norm_rnn = RMSNorm(config.d_model)
        self.mixer = RecurrentTokenMixer(config.d_model)
        self.norm_ffn = RMSNorm(config.d_model)
        self.ffn = SwiGLUFeedForward(config.d_model, config.d_ff, config.dropout)

    def forward(self, x: Tensor, initial_state: Tensor | None = None) -> tuple[Tensor, Tensor]:
        mixed, state = self.mixer(self.norm_rnn(x), initial_state=initial_state)
        x = x + mixed
        return x + self.ffn(self.norm_ffn(x)), state


class HybridArchitectureModel(nn.Module):
    def __init__(self, config: ModelConfig, attention_every: int = 4) -> None:
        super().__init__()
        if attention_every < 1:
            raise ValueError("attention_every must be positive")
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        blocks: list[nn.Module] = []
        for layer_idx in range(config.n_layers):
            if layer_idx == 0 or (layer_idx + 1) % attention_every == 0:
                blocks.append(TransformerBlock(config, causal=True))
            else:
                blocks.append(RNNBackboneBlock(config))
        self.blocks = nn.ModuleList(blocks)
        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(self, input_ids: Tensor, labels: Tensor | None = None) -> dict[str, Any]:
        batch, seq_len = input_ids.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError(f"sequence length {seq_len} exceeds max_seq_len {self.config.max_seq_len}")
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch, seq_len)
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        recurrent_states: list[Tensor] = []
        for block in self.blocks:
            if isinstance(block, RNNBackboneBlock):
                x, state = block(x)
                recurrent_states.append(state)
            else:
                x = block(x)
        hidden = self.norm(x)
        logits = self.lm_head(hidden)
        output: dict[str, Any] = {"logits": logits, "hidden": hidden, "recurrent_states": recurrent_states}
        if labels is not None:
            output["loss"] = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return output


class MultiTokenPredictionModel(nn.Module):
    def __init__(self, config: ModelConfig, prediction_offsets: int = 4) -> None:
        super().__init__()
        if prediction_offsets < 1:
            raise ValueError("prediction_offsets must be positive")
        self.backbone = DecoderBackbone(config, causal=True)
        self.heads = nn.ModuleList(
            nn.Linear(config.d_model, config.vocab_size, bias=False) for _ in range(prediction_offsets)
        )

    def forward(self, input_ids: Tensor, labels: Tensor | None = None) -> dict[str, Any]:
        hidden = self.backbone(input_ids)
        logits_by_offset = {offset + 1: head(hidden) for offset, head in enumerate(self.heads)}
        output: dict[str, Any] = {
            "logits": logits_by_offset[1],
            "mtp_logits": logits_by_offset,
            "hidden": hidden,
        }
        if labels is not None:
            losses = []
            for offset, logits in logits_by_offset.items():
                if logits.size(1) <= offset:
                    continue
                losses.append(
                    F.cross_entropy(
                        logits[:, :-offset].reshape(-1, logits.size(-1)),
                        labels[:, offset:].reshape(-1),
                        ignore_index=-100,
                    )
                )
            if losses:
                output["loss"] = torch.stack(losses).mean()
        return output


class LatentReasoningModel(nn.Module):
    def __init__(self, config: ModelConfig, num_latents: int = 8) -> None:
        super().__init__()
        if num_latents < 1:
            raise ValueError("num_latents must be positive")
        self.config = config
        self.num_latents = num_latents
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len + num_latents, config.d_model)
        self.latents = nn.Parameter(torch.randn(num_latents, config.d_model) * 0.02)
        self.blocks = nn.ModuleList(TransformerBlock(config, causal=True) for _ in range(config.n_layers))
        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def _insert_latents(self, x: Tensor, latent_insert_index: int) -> tuple[Tensor, Tensor]:
        batch, seq_len, _ = x.shape
        if latent_insert_index < 0 or latent_insert_index > seq_len:
            raise ValueError("latent_insert_index must be within the token sequence")
        latents = self.latents.unsqueeze(0).expand(batch, -1, -1)
        x_with_latents = torch.cat(
            [x[:, :latent_insert_index], latents, x[:, latent_insert_index:]],
            dim=1,
        )
        visible = torch.ones(seq_len + self.num_latents, dtype=torch.bool, device=x.device)
        visible[latent_insert_index : latent_insert_index + self.num_latents] = False
        return x_with_latents, visible

    def forward(
        self,
        input_ids: Tensor,
        labels: Tensor | None = None,
        latent_insert_index: int | None = None,
    ) -> dict[str, Tensor]:
        batch, seq_len = input_ids.shape
        insert_index = seq_len if latent_insert_index is None else latent_insert_index
        token_x = self.token_emb(input_ids)
        x, visible = self._insert_latents(token_x, insert_index)
        if x.size(1) > self.config.max_seq_len + self.num_latents:
            raise ValueError("sequence plus latent tokens exceeds configured positional capacity")
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0).expand(batch, x.size(1))
        x = x + self.pos_emb(positions)
        for block in self.blocks:
            x = block(x)
        hidden_with_latents = self.norm(x)
        logits_with_latents = self.lm_head(hidden_with_latents)
        logits = logits_with_latents[:, visible]
        output = {
            "logits": logits,
            "hidden": hidden_with_latents[:, visible],
            "latent_hidden": hidden_with_latents[:, ~visible],
        }
        if labels is not None:
            output["loss"] = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return output


class DiscreteDiffusionLanguageModel(nn.Module):
    def __init__(self, config: ModelConfig, num_diffusion_steps: int = 1000, mask_token_id: int | None = None) -> None:
        super().__init__()
        if num_diffusion_steps < 1:
            raise ValueError("num_diffusion_steps must be positive")
        self.config = config
        self.num_diffusion_steps = num_diffusion_steps
        self.mask_token_id = config.vocab_size if mask_token_id is None else mask_token_id
        embed_size = max(config.vocab_size, self.mask_token_id + 1)
        self.token_emb = nn.Embedding(embed_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.time_emb = nn.Embedding(num_diffusion_steps, config.d_model)
        self.blocks = nn.ModuleList(TransformerBlock(config, causal=False) for _ in range(config.n_layers))
        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def q_sample(self, clean_ids: Tensor, timesteps: Tensor) -> tuple[Tensor, Tensor]:
        probs = (timesteps.float() + 1.0) / float(self.num_diffusion_steps)
        while probs.ndim < clean_ids.ndim:
            probs = probs.unsqueeze(-1)
        mask = torch.rand_like(clean_ids.float()) < probs
        noisy = clean_ids.masked_fill(mask, self.mask_token_id)
        return noisy, mask

    def forward(
        self,
        noisy_ids: Tensor,
        timesteps: Tensor,
        labels: Tensor | None = None,
        loss_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        batch, seq_len = noisy_ids.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError(f"sequence length {seq_len} exceeds max_seq_len {self.config.max_seq_len}")
        positions = torch.arange(seq_len, device=noisy_ids.device).unsqueeze(0).expand(batch, seq_len)
        time = self.time_emb(timesteps).unsqueeze(1)
        x = self.token_emb(noisy_ids) + self.pos_emb(positions) + time
        for block in self.blocks:
            x = block(x)
        hidden = self.norm(x)
        logits = self.lm_head(hidden)
        output = {"logits": logits, "hidden": hidden}
        if labels is not None:
            if loss_mask is None:
                loss_mask = torch.ones_like(labels, dtype=torch.bool)
            flat_logits = logits.reshape(-1, logits.size(-1))
            flat_labels = labels.reshape(-1)
            flat_mask = loss_mask.reshape(-1)
            output["loss"] = F.cross_entropy(flat_logits[flat_mask], flat_labels[flat_mask])
        return output

    def training_step(self, clean_ids: Tensor, timesteps: Tensor | None = None) -> dict[str, Tensor]:
        if timesteps is None:
            timesteps = torch.randint(
                0,
                self.num_diffusion_steps,
                (clean_ids.size(0),),
                dtype=torch.long,
                device=clean_ids.device,
            )
        noisy, mask = self.q_sample(clean_ids, timesteps)
        output = self.forward(noisy, timesteps, labels=clean_ids, loss_mask=mask)
        output["noisy_ids"] = noisy
        output["loss_mask"] = mask
        return output


class DifferentiableMemory(nn.Module):
    def __init__(self, d_model: int, memory_slots: int = 128) -> None:
        super().__init__()
        self.keys = nn.Parameter(torch.randn(memory_slots, d_model) * 0.02)
        self.values = nn.Parameter(torch.randn(memory_slots, d_model) * 0.02)
        self.query = nn.Linear(d_model, d_model, bias=False)
        self.scale = d_model**-0.5

    def forward(self, x: Tensor, external_memory: Tensor | None = None) -> tuple[Tensor, Tensor]:
        query = self.query(x)
        if external_memory is None:
            keys = self.keys.unsqueeze(0).expand(x.size(0), -1, -1)
            values = self.values.unsqueeze(0).expand(x.size(0), -1, -1)
        else:
            keys = external_memory
            values = external_memory
        scores = torch.matmul(query, keys.transpose(-2, -1)) * self.scale
        weights = F.softmax(scores.float(), dim=-1).to(dtype=x.dtype)
        return torch.matmul(weights, values), weights


class MemoryAugmentedBlock(nn.Module):
    def __init__(self, config: ModelConfig, memory_slots: int = 128) -> None:
        super().__init__()
        self.norm_attn = RMSNorm(config.d_model)
        self.attn = SelfAttention(config.d_model, config.n_heads, config.dropout, causal=True)
        self.norm_mem = RMSNorm(config.d_model)
        self.memory = DifferentiableMemory(config.d_model, memory_slots=memory_slots)
        self.memory_gate = nn.Linear(config.d_model, config.d_model)
        self.norm_ffn = RMSNorm(config.d_model)
        self.ffn = SwiGLUFeedForward(config.d_model, config.d_ff, config.dropout)

    def forward(self, x: Tensor, external_memory: Tensor | None = None) -> tuple[Tensor, Tensor]:
        x = x + self.attn(self.norm_attn(x))
        mem, weights = self.memory(self.norm_mem(x), external_memory=external_memory)
        x = x + torch.sigmoid(self.memory_gate(x)) * mem
        return x + self.ffn(self.norm_ffn(x)), weights


class MemoryAugmentedLM(nn.Module):
    def __init__(self, config: ModelConfig, memory_slots: int = 128) -> None:
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.blocks = nn.ModuleList(MemoryAugmentedBlock(config, memory_slots=memory_slots) for _ in range(config.n_layers))
        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: Tensor,
        labels: Tensor | None = None,
        external_memory: Tensor | None = None,
    ) -> dict[str, Any]:
        batch, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch, seq_len)
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        memory_weights = []
        for block in self.blocks:
            x, weights = block(x, external_memory=external_memory)
            memory_weights.append(weights)
        hidden = self.norm(x)
        logits = self.lm_head(hidden)
        output: dict[str, Any] = {"logits": logits, "hidden": hidden, "memory_weights": memory_weights}
        if labels is not None:
            output["loss"] = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return output


class OmniModalArchitecture(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        image_dim: int = 1024,
        video_dim: int = 1024,
        audio_dim: int = 768,
        action_vocab_size: int = 256,
    ) -> None:
        super().__init__()
        self.config = config
        self.text_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.action_emb = nn.Embedding(action_vocab_size, config.d_model)
        self.image_proj = nn.Linear(image_dim, config.d_model)
        self.video_proj = nn.Linear(video_dim, config.d_model)
        self.audio_proj = nn.Linear(audio_dim, config.d_model)
        self.modality_emb = nn.Embedding(5, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.blocks = nn.ModuleList(TransformerBlock(config, causal=True) for _ in range(config.n_layers))
        self.norm = RMSNorm(config.d_model)
        self.text_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.action_head = nn.Linear(config.d_model, action_vocab_size, bias=False)

    def _with_modality(self, x: Tensor, modality_id: int) -> Tensor:
        return x + self.modality_emb.weight[modality_id].view(1, 1, -1)

    def forward(
        self,
        input_ids: Tensor | None = None,
        image_features: Tensor | None = None,
        video_features: Tensor | None = None,
        audio_features: Tensor | None = None,
        action_ids: Tensor | None = None,
    ) -> dict[str, Tensor]:
        pieces = []
        if input_ids is not None:
            pieces.append(self._with_modality(self.text_emb(input_ids), 0))
        if image_features is not None:
            pieces.append(self._with_modality(self.image_proj(image_features), 1))
        if video_features is not None:
            pieces.append(self._with_modality(self.video_proj(video_features), 2))
        if audio_features is not None:
            pieces.append(self._with_modality(self.audio_proj(audio_features), 3))
        if action_ids is not None:
            pieces.append(self._with_modality(self.action_emb(action_ids), 4))
        if not pieces:
            raise ValueError("at least one modality input must be provided")

        batch = pieces[0].size(0)
        if any(piece.size(0) != batch for piece in pieces):
            raise ValueError("all modality inputs must share the same batch size")
        x = torch.cat(pieces, dim=1)
        if x.size(1) > self.config.max_seq_len:
            raise ValueError(f"packed multimodal length {x.size(1)} exceeds max_seq_len {self.config.max_seq_len}")
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0).expand(batch, x.size(1))
        x = x + self.pos_emb(positions)
        for block in self.blocks:
            x = block(x)
        hidden = self.norm(x)
        return {
            "hidden": hidden,
            "text_logits": self.text_head(hidden),
            "action_logits": self.action_head(hidden),
        }


class ReasoningNativeArchitecture(nn.Module):
    def __init__(self, config: ModelConfig, num_plan_tokens: int = 64) -> None:
        super().__init__()
        self.backbone = DecoderBackbone(config, causal=True)
        self.policy_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.verifier_head = nn.Linear(config.d_model, 1)
        self.planner_head = nn.Linear(config.d_model, num_plan_tokens)
        self.value_head = nn.Linear(config.d_model, 1)

    def forward(
        self,
        input_ids: Tensor,
        labels: Tensor | None = None,
        process_labels: Tensor | None = None,
        plan_labels: Tensor | None = None,
        loss_weights: dict[str, float] | None = None,
    ) -> dict[str, Tensor]:
        weights = {"policy": 1.0, "verifier": 0.2, "planner": 0.2}
        if loss_weights is not None:
            weights.update(loss_weights)

        hidden = self.backbone(input_ids)
        policy_logits = self.policy_head(hidden)
        verifier_logits = self.verifier_head(hidden).squeeze(-1)
        planner_logits = self.planner_head(hidden)
        values = self.value_head(hidden).squeeze(-1)
        output = {
            "logits": policy_logits,
            "verifier_logits": verifier_logits,
            "planner_logits": planner_logits,
            "values": values,
            "hidden": hidden,
        }

        losses = []
        if labels is not None:
            policy_loss = F.cross_entropy(
                policy_logits[:, :-1].reshape(-1, policy_logits.size(-1)),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
            output["policy_loss"] = policy_loss
            losses.append(weights["policy"] * policy_loss)
        if process_labels is not None:
            mask = process_labels != -100
            verifier_loss = F.binary_cross_entropy_with_logits(
                verifier_logits[mask],
                process_labels[mask].to(dtype=verifier_logits.dtype),
            )
            output["verifier_loss"] = verifier_loss
            losses.append(weights["verifier"] * verifier_loss)
        if plan_labels is not None:
            planner_loss = F.cross_entropy(
                planner_logits.reshape(-1, planner_logits.size(-1)),
                plan_labels.reshape(-1),
                ignore_index=-100,
            )
            output["planner_loss"] = planner_loss
            losses.append(weights["planner"] * planner_loss)
        if losses:
            output["loss"] = torch.stack(losses).sum()
        return output


REFERENCE_IMPLEMENTATIONS = {
    "MoE": MoETransformerBlock,
    "Sparse / Linear Attention": SparseLinearAttentionBlock,
    "RNN-like Backbone": RNNBackboneBlock,
    "Hybrid Architecture": HybridArchitectureModel,
    "MTP": MultiTokenPredictionModel,
    "Latent Reasoning": LatentReasoningModel,
    "dLLM": DiscreteDiffusionLanguageModel,
    "Memory-augmented LLM": MemoryAugmentedLM,
    "Omni-modal Architecture": OmniModalArchitecture,
    "Reasoning-native Architecture": ReasoningNativeArchitecture,
}


def build_reference_implementation(family: str, config: ModelConfig, **kwargs: Any) -> nn.Module:
    try:
        cls = REFERENCE_IMPLEMENTATIONS[family]
    except KeyError as exc:
        raise ValueError(f"unknown architecture family: {family}") from exc
    return cls(config, **kwargs)

