# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib
from types import SimpleNamespace

import torch

import vllm.envs as envs
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends import flash_attn, flash_attn_diffkv, sink_debug


def test_compute_attention_sink_mass_transposes_lse():
    sinks = torch.tensor([0.0, 1.0], dtype=torch.float32)
    lse = torch.tensor([[0.0, 2.0, 4.0], [1.0, 3.0, 5.0]], dtype=torch.float32)

    mass = sink_debug.compute_attention_sink_mass(sinks, lse)

    expected = torch.exp(sinks[:, None] - lse).transpose(0, 1).contiguous()
    assert mass.shape == (3, 2)
    torch.testing.assert_close(mass, expected)


def test_dump_attention_sink_mass_disabled(monkeypatch, tmp_path):
    monkeypatch.delenv("VLLM_ATTENTION_SINK_DUMP_PATH", raising=False)
    importlib.reload(envs)
    importlib.reload(sink_debug)

    path = sink_debug.dump_attention_sink_mass(
        backend="FLASH_ATTN",
        layer_name="model.layers.0.self_attn.attn",
        sinks=torch.tensor([0.0], dtype=torch.float32),
        softmax_lse=torch.tensor([[0.0]], dtype=torch.float32),
        num_actual_tokens=1,
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        seq_lens=torch.tensor([1], dtype=torch.int32),
        sliding_window=(-1, -1),
    )

    assert path is None
    assert list(tmp_path.iterdir()) == []


def test_dump_attention_sink_mass_writes_file(monkeypatch, tmp_path):
    monkeypatch.setenv("VLLM_ATTENTION_SINK_DUMP_PATH", str(tmp_path))
    importlib.reload(envs)
    importlib.reload(sink_debug)

    sinks = torch.tensor([0.0, 1.0], dtype=torch.float32)
    lse = torch.tensor([[0.0, 2.0], [1.0, 3.0]], dtype=torch.float32)
    path = sink_debug.dump_attention_sink_mass(
        backend="FLASH_ATTN",
        layer_name="model.layers.0.self_attn.attn",
        sinks=sinks,
        softmax_lse=lse,
        num_actual_tokens=2,
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        seq_lens=torch.tensor([2], dtype=torch.int32),
        sliding_window=(127, 0),
    )

    assert path is not None
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["backend"] == "FLASH_ATTN"
    assert payload["layer_name"] == "model.layers.0.self_attn.attn"
    assert payload["num_actual_tokens"] == 2
    assert payload["sliding_window"] == (127, 0)
    torch.testing.assert_close(payload["sink_bias"], sinks)
    torch.testing.assert_close(payload["softmax_lse"], lse)
    torch.testing.assert_close(
        payload["sink_mass"], torch.exp(sinks[:, None] - lse).transpose(0, 1)
    )


def test_flash_attention_hook_requests_lse_when_dump_enabled(monkeypatch):
    records = _run_flash_attention_forward(
        monkeypatch,
        flash_attn,
        flash_attn.FlashAttentionImpl,
        dump_enabled=True,
    )

    assert records["kernel"]["return_softmax_lse"] is True
    assert records["kernel"]["s_aux"] is records["sinks"]
    assert records["dump"]["backend"] == "FLASH_ATTN"
    assert records["dump"]["layer_name"] == "layer.test.attn"
    assert records["dump"]["sinks"] is records["sinks"]
    torch.testing.assert_close(records["dump"]["softmax_lse"], records["lse"])
    assert records["dump"]["num_actual_tokens"] == 2
    assert records["dump"]["query_start_loc"] is records["metadata"].query_start_loc
    assert records["dump"]["seq_lens"] is records["metadata"].seq_lens
    assert records["dump"]["sliding_window"] == (127, 0)


def test_flash_attention_diffkv_hook_keeps_fast_path_when_dump_disabled(
    monkeypatch,
):
    records = _run_flash_attention_forward(
        monkeypatch,
        flash_attn_diffkv,
        flash_attn_diffkv.FlashAttentionDiffKVImpl,
        dump_enabled=False,
    )

    assert "return_softmax_lse" not in records["kernel"]
    assert records["kernel"]["s_aux"] is records["sinks"]
    assert "dump" not in records


def _run_flash_attention_forward(
    monkeypatch,
    backend_module,
    impl_cls,
    *,
    dump_enabled: bool,
):
    records = {}
    sinks = torch.tensor([0.0, 1.0], dtype=torch.float32)
    softmax_lse = torch.tensor([[0.0, 2.0], [1.0, 3.0]], dtype=torch.float32)

    def fake_flash_attn_varlen_func(**kwargs):
        records["kernel"] = kwargs
        kwargs["out"].fill_(3.0)
        if kwargs.get("return_softmax_lse"):
            return kwargs["out"], softmax_lse
        return kwargs["out"]

    def fake_dump_attention_sink_mass(**kwargs):
        records["dump"] = kwargs
        return None

    monkeypatch.setattr(
        backend_module,
        "attention_sink_dump_enabled",
        lambda: dump_enabled,
    )
    monkeypatch.setattr(
        backend_module,
        "dump_attention_sink_mass",
        fake_dump_attention_sink_mass,
    )
    monkeypatch.setattr(
        backend_module,
        "flash_attn_varlen_func",
        fake_flash_attn_varlen_func,
        raising=False,
    )

    impl = object.__new__(impl_cls)
    impl.num_heads = 2
    impl.head_size = 4
    impl.scale = 1.0
    impl.num_kv_heads = 2
    impl.alibi_slopes = None
    impl.sliding_window = (127, 0)
    impl.kv_cache_dtype = "auto"
    impl.logits_soft_cap = 0
    impl.attn_type = AttentionType.DECODER
    impl.vllm_flash_attn_version = 4
    impl.sinks = sinks
    impl.supports_quant_query_input = False
    impl.dcp_world_size = 1

    query = torch.zeros(2, 2, 4, dtype=torch.float32)
    key = torch.zeros(2, 2, 4, dtype=torch.float32)
    value = torch.zeros(2, 2, 4, dtype=torch.float32)
    kv_cache = torch.zeros(1, 2, 2, 2, 4, dtype=torch.float32)
    output = torch.zeros(2, 2, 4, dtype=torch.float32)
    metadata = flash_attn.FlashAttentionMetadata(
        num_actual_tokens=2,
        max_query_len=2,
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        max_seq_len=2,
        seq_lens=torch.tensor([2], dtype=torch.int32),
        block_table=torch.tensor([[0]], dtype=torch.int32),
        slot_mapping=torch.tensor([0, 1], dtype=torch.int64),
        use_cascade=False,
        common_prefix_len=0,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=None,
        max_num_splits=0,
        causal=True,
        mm_prefix_range_tensor=None,
    )
    layer = SimpleNamespace(
        layer_name="layer.test.attn",
        _q_scale=torch.ones(1, dtype=torch.float32),
        _k_scale=torch.ones(1, dtype=torch.float32),
        _v_scale=torch.ones(1, dtype=torch.float32),
    )

    returned = impl.forward(layer, query, key, value, kv_cache, metadata, output)

    assert returned is output
    assert torch.all(output == 3.0)
    records["sinks"] = sinks
    records["lse"] = softmax_lse
    records["metadata"] = metadata
    return records
