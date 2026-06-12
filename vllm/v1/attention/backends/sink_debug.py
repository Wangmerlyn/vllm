# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools
import os
from pathlib import Path

import regex as re
import torch

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)

_DUMP_COUNTER = itertools.count()
_WARNED_DUMP_FAILURE = False


def compute_attention_sink_mass(
    sinks: torch.Tensor,
    softmax_lse: torch.Tensor,
) -> torch.Tensor:
    sink_mass = torch.exp(sinks.float()[:, None] - softmax_lse.float())
    return sink_mass.transpose(0, 1).contiguous()


def attention_sink_dump_enabled() -> bool:
    return envs.VLLM_ATTENTION_SINK_DUMP_PATH is not None


def dump_attention_sink_mass(
    *,
    backend: str,
    layer_name: str,
    sinks: torch.Tensor,
    softmax_lse: torch.Tensor,
    num_actual_tokens: int,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    sliding_window: tuple[int, int],
) -> Path | None:
    dump_path = envs.VLLM_ATTENTION_SINK_DUMP_PATH
    if dump_path is None:
        return None

    global _WARNED_DUMP_FAILURE
    try:
        root = Path(dump_path)
        root.mkdir(parents=True, exist_ok=True)
        layer_stub = re.sub(r"[^A-Za-z0-9_.-]+", "_", layer_name)
        filename = (
            f"pid{os.getpid()}_{next(_DUMP_COUNTER):06d}_{backend}_{layer_stub}.pt"
        )
        path = root / filename
        payload = {
            "backend": backend,
            "layer_name": layer_name,
            "num_actual_tokens": int(num_actual_tokens),
            "sink_bias": sinks.detach().float().cpu(),
            "softmax_lse": softmax_lse.detach().float().cpu(),
            "sink_mass": compute_attention_sink_mass(sinks, softmax_lse)
            .detach()
            .float()
            .cpu(),
            "query_start_loc": query_start_loc.detach().cpu(),
            "seq_lens": seq_lens.detach().cpu(),
            "sliding_window": tuple(sliding_window),
        }
        torch.save(payload, path)
        return path
    except Exception:
        if not _WARNED_DUMP_FAILURE:
            logger.exception("Failed to dump attention sink mass.")
            _WARNED_DUMP_FAILURE = True
        return None
