# test_mistral_small_3_2_24b_instruct_2506.py
# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
CPU vs Spyre comparison tests for selected torch operations used in
  mistralai/Mistral-Small-3.2-24B-Instruct-2506.


Running tests
-------------
All tests:
    pytest test_mistral_small_3_2_24b_instruct_2506.py

--- By op (umbrella marks) ---
    pytest test_mistral_small_3_2_24b_instruct_2506.py -m torch_cat
    pytest test_mistral_small_3_2_24b_instruct_2506.py -m torch_mul
    pytest test_mistral_small_3_2_24b_instruct_2506.py -m torch_neg
    pytest test_mistral_small_3_2_24b_instruct_2506.py -m torch_pow
    pytest test_mistral_small_3_2_24b_instruct_2506.py -m torch_rsqrt
    pytest test_mistral_small_3_2_24b_instruct_2506.py -m torch_unsqueeze
    pytest test_mistral_small_3_2_24b_instruct_2506.py -m torch_sdpa
    pytest test_mistral_small_3_2_24b_instruct_2506.py -m torch_reshape
    pytest test_mistral_small_3_2_24b_instruct_2506.py -m torch_nn_functional_silu

--- Combining marks with -k keyword filters ---
All eager cat tests:
    pytest test_mistral_small_3_2_24b_instruct_2506.py -m torch_cat -k "eager"

All compiled mul tests:
    pytest test_mistral_small_3_2_24b_instruct_2506.py -m torch_mul -k "compiled"

A specific pattern:
    pytest test_mistral_small_3_2_24b_instruct_2506.py -m torch_neg -k "pattern_000"

--- Boolean mark expressions ---
    pytest test_mistral_small_3_2_24b_instruct_2506.py -m "torch_cat or torch_mul"
    pytest test_mistral_small_3_2_24b_instruct_2506.py -m "torch_pow or torch_rsqrt"
"""

import math
import os
import sys
import unittest

import pytest
import torch
import torch.nn.functional as F

_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
if _MODEL_DIR not in sys.path:
    sys.path.insert(0, _MODEL_DIR)

from utils_mistral_small_3_2_24b_instruct_2506 import (
    # Infrastructure
    ParameterizedTestMeta,
    DEVICE,
    TOLERANCES,
 
    # Architecture constants
    NUM_Q_HEADS,
    NUM_KV_HEADS,
    HEAD_DIM,
    GQA_GROUPS,
    SCALE,
    SLIDING_WINDOW,
    NUM_LAYERS,
    DEFAULT_DTYPE,
    VOCAB_SIZE,
    ROPE_THETA,
    HIDDEN_SIZE,
    INTERMEDIATE_SIZE,
 
    # Shorthand dtypes
    BF16,
    F16,
    F32,
    I64,
 
    # SDPA pre-built param dicts
    PREFILL_PARAMS,
    DECODE_PARAMS,
    DTYPE_PARAMS,
    NUMERIC_COVERAGE_PARAMS,
    GROWING_KV_PARAMS,
 
    # Tensor factories
    make_qkv,
    make_tensor,
    _t,
     _W,
    cached_randn,
    expand_kv,
 
    # Mask builders
    causal_mask,
    sdpa_fn,
 
    # Comparison helpers
    compare_sdpa,
    HIDDEN_SIZE,
    NUM_Q_HEADS, 
    HEAD_DIM,
    compare_with_cpu,
)

# Native dtype for all attention / FFN tensors in this model
BF16 = torch.bfloat16
# RMSNorm intermediate computations are promoted to float32
F32  = torch.float32
S = slice        # S(None, 32)  →  slice(None, 32)  →  [:32]
_ = slice(None) 
# ═════════════════════════════════════════════════════════════════════════════
#  TestSDPA  —  torch.nn.functional.scaled_dot_product_attention
# ═════════════════════════════════════════════════════════════════════════════
 
class TestSDPA(unittest.TestCase, metaclass=ParameterizedTestMeta):
    """
    Eager and compiled CPU vs Spyre SDPA comparison tests for
    Mistral-Small-3.2-24B-Instruct-2506.
 
    Architecture specifics
    ----------------------
    q  : [B, 32, S_q,  128]   (NUM_Q_HEADS=32,   HEAD_DIM=128)
    k  : [B,  8, S_kv, 128]   (NUM_KV_HEADS=8,   GQA groups=4)
    v  : [B,  8, S_kv, 128]
    scale = 1 / sqrt(128) ≈ 0.0884
    No sliding window attention (SLIDING_WINDOW=None).
    Native dtype: float16.
 
    Target decode shape: q=[1,32,1,128], k=v=[1,8,38,128]
    (1 new token attending over 38-token KV cache).
 
    Execution modes
    ---------------
    eager    → compare_sdpa()                 (no torch.compile, dtype-aware tols)
    compiled → compare_with_cpu(...compiled=True)  (wider COMPILED_ATOL/RTOL)
    """
 
    pytestmark = pytest.mark.torch_sdpa
 
    torch.manual_seed(0xBEEF_2506)
 
    # ── Helper: build eager+compiled param_sets from a {name: (q,k,v)} dict ──
    @staticmethod
    def _ec(d):
        """Return {name_eager: (*qkv, False), name_compiled: (*qkv, True)} for d."""
        out = {}
        for k, v in d.items():
            out[f"{k}_eager"]    = (*v, False)
            out[f"{k}_compiled"] = (*v, True)
        return out
 
    # ── PARAMS ───────────────────────────────────────────────────────────────
 
    PARAMS = {
 
        # ── Target shape: exact [1,32,1,128] float16 decode ─────────────────────
        ("test_target_shape_decode", "test_sdpa_decode"): {
            "param_sets": {
                "bs1_kv38_fp16_eager":    (*make_qkv(1, 1, 38, diff="target"), False),
                "bs1_kv38_fp16_compiled": (*make_qkv(1, 1, 38, diff="target"), True),
                "bs1_kv1_fp16_eager":     (*make_qkv(1, 1,  1, diff="target1"), False),
                "bs1_kv1_fp16_compiled":  (*make_qkv(1, 1,  1, diff="target1"), True),
            },
        },
 
        # ── Prefill causal (float16) ─────────────────────────────────────────
        ("test_prefill_causal_fp16", "test_sdpa_prefill_causal"): {
            "param_sets": {
                **{f"{k}_eager":    (*v, False) for k, v in PREFILL_PARAMS.items()},
                **{f"{k}_compiled": (*v, True)  for k, v in PREFILL_PARAMS.items()},
            },
        },
 
        # ── Decode (no mask, float16) ────────────────────────────────────────
        ("test_decode_fp16", "test_sdpa_decode"): {
            "param_sets": {
                **{f"{k}_eager":    (*v, False) for k, v in DECODE_PARAMS.items()},
                **{f"{k}_compiled": (*v, True)  for k, v in DECODE_PARAMS.items()},
            },
        },
 
        # ── Multi-dtype prefill (causal) ──────────────────────────────────────
        ("test_prefill_causal_multidtype", "test_sdpa_prefill_causal"): {
            "param_sets": {
                **{f"{k}_eager":    (*v, False) for k, v in DTYPE_PARAMS.items()},
                **{f"{k}_compiled": (*v, True)  for k, v in DTYPE_PARAMS.items()},
            },
        },
 
        # ── is_causal=True ≡ explicit causal mask ────────────────────────────
        ("test_causal_flag_vs_mask", "test_sdpa_causal_flag_vs_mask"): {
            "param_sets": {
                "bs1_seq38_fp16_eager":    (*make_qkv(1, 38, 38, diff="causal_flag"), False),
                "bs1_seq38_fp16_compiled": (*make_qkv(1, 38, 38, diff="causal_flag"), True),
                "bs1_seq128_fp16_eager":   (*make_qkv(1, 128, 128, diff="causal_flag128"), False),
                "bs1_seq128_fp16_compiled":(*make_qkv(1, 128, 128, diff="causal_flag128"), True),
            },
        },
 
        # ── Attention-weight rows sum to 1 ────────────────────────────────────
        ("test_attn_weights_sum_to_one", "test_sdpa_weights_sum_to_one"): {
            "param_sets": {
                "bs1_seq38_fp16_eager":    (*make_qkv(1, 38, 38, diff="wsum"), False),
                "bs1_seq38_fp16_compiled": (*make_qkv(1, 38, 38, diff="wsum"), True),
            },
        },
 
        # ── GQA expansion shape ───────────────────────────────────────────────
        ("test_gqa_expansion_shape", "test_sdpa_gqa_shape"): {
            "param_sets": {
                "bs2_seq38_fp16_eager":    (*make_qkv(2, 38, 38, diff="gqa"), False),
                "bs2_seq38_fp16_compiled": (*make_qkv(2, 38, 38, diff="gqa"), True),
            },
        },
 
        # ── Batch consistency ─────────────────────────────────────────────────
        ("test_batch_consistency", "test_sdpa_batch_consistency"): {
            "param_sets": {
                "bs4_seq38_fp16_eager":    (*make_qkv(4, 38, 38, diff="batch_cons"), False),
                "bs4_seq38_fp16_compiled": (*make_qkv(4, 38, 38, diff="batch_cons"), True),
            },
        },
 
        # ── Numerical coverage: 40 seeds across short/mid/long prefill ────────
        ("test_numeric_coverage_prefill", "test_sdpa_prefill_causal"): {
            "param_sets": {
                **{f"{k}_eager":    (*v, False) for k, v in NUMERIC_COVERAGE_PARAMS.items()},
                **{f"{k}_compiled": (*v, True)  for k, v in NUMERIC_COVERAGE_PARAMS.items()},
            },
        },
 
        # ── Growing KV cache (autoregressive decode) ──────────────────────────
        ("test_growing_kvcache", "test_sdpa_decode"): {
            "param_sets": {
                **{f"{k}_eager":    (*v, False) for k, v in GROWING_KV_PARAMS.items()},
                **{f"{k}_compiled": (*v, True)  for k, v in GROWING_KV_PARAMS.items()},
            },
        },
 
        # ── Gradient flow (eager only) ────────────────────────────────────────
        ("test_gradient_flow", "test_sdpa_gradient_flow"): {
            "param_sets": {
                "bs1_seq38_fp16_eager": (*make_qkv(1, 38, 38, diff="grad"), False),
            },
        },
 
        # ── Determinism ───────────────────────────────────────────────────────
        ("test_determinism", "test_sdpa_determinism"): {
            "param_sets": {
                "bs1_seq38_fp16_eager":    (*make_qkv(1, 38, 38, diff="det"), False),
                "bs1_seq38_fp16_compiled": (*make_qkv(1, 38, 38, diff="det"), True),
            },
        },
 
        # ── Padding mask blocks future tokens ─────────────────────────────────
        ("test_padding_mask", "test_sdpa_padding_mask"): {
            "param_sets": {
                "bs1_seq38_fp16_eager":    (*make_qkv(1, 38, 38, diff="pad"), False),
                "bs1_seq38_fp16_compiled": (*make_qkv(1, 38, 38, diff="pad"), True),
            },
        },
    }
 
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
 
    # ── Base test methods ─────────────────────────────────────────────────────
 
    def test_sdpa_prefill_causal(self, q, k, v, compiled):
        """Prefill: is_causal=True — eager or compiled CPU vs Spyre."""
        fn = lambda q, k, v: sdpa_fn(q, k, v, is_causal=True)
        if compiled:
            compare_with_cpu(fn, q, k, v, compiled=True)
        else:
            compare_sdpa(fn, q, k, v, dtype=q.dtype)
 
    def test_sdpa_decode(self, q, k, v, compiled):
        """Decode (seq_q=1): no mask — eager or compiled CPU vs Spyre."""
        fn = lambda q, k, v: sdpa_fn(q, k, v, is_causal=False)
        if compiled:
            compare_with_cpu(fn, q, k, v, compiled=True)
        else:
            compare_sdpa(fn, q, k, v, dtype=q.dtype)
 
    def test_sdpa_causal_flag_vs_mask(self, q, k, v, compiled):
        """is_causal=True flag and explicit causal mask must agree on both devices."""
        seq_len  = q.shape[2]
        fn_flag  = lambda q, k, v: sdpa_fn(q, k, v, is_causal=True)
        fn_mask  = lambda q, k, v: sdpa_fn(
            q, k, v,
            attn_mask=causal_mask(seq_len, q.dtype, q.device),
            is_causal=False,
        )
        if compiled:
            compare_with_cpu(fn_flag, q, k, v, compiled=True)
            compare_with_cpu(fn_mask, q, k, v, compiled=True)
        else:
            compare_sdpa(fn_flag, q, k, v, dtype=q.dtype)
            compare_sdpa(fn_mask, q, k, v, dtype=q.dtype)
 
    def test_sdpa_weights_sum_to_one(self, q, k, v, compiled):
        """Each attention-weight row must sum to 1 (softmax in fp32)."""
        seq_len = q.shape[2]
 
        def fn(q, k, v):
            k_exp, v_exp = expand_kv(k, v)
            scores  = torch.matmul(q, k_exp.transpose(-2, -1)) * SCALE
            scores += causal_mask(seq_len, q.dtype, q.device)
            return torch.softmax(scores.float(), dim=-1).sum(dim=-1)
 
        if compiled:
            compare_with_cpu(fn, q, k, v, compiled=True)
        else:
            compare_sdpa(fn, q, k, v, dtype=q.dtype)
 
    def test_sdpa_gqa_shape(self, q, k, v, compiled):
        """After GQA head expansion K and V shapes must equal Q shape."""
        def fn(q, k, v):
            k_exp, v_exp = expand_kv(k, v)
            assert k_exp.shape == q.shape, (
                f"K shape mismatch: {tuple(k_exp.shape)} != {tuple(q.shape)}"
            )
            assert v_exp.shape == q.shape, (
                f"V shape mismatch: {tuple(v_exp.shape)} != {tuple(q.shape)}"
            )
            return sdpa_fn(q, k, v, is_causal=True)
 
        if compiled:
            compare_with_cpu(fn, q, k, v, compiled=True)
        else:
            compare_sdpa(fn, q, k, v, dtype=q.dtype)
 
    def test_sdpa_batch_consistency(self, q, k, v, compiled):
        """All batch items from the same tensor must produce identical outputs."""
        tol = TOLERANCES[q.dtype]
        B   = q.shape[0]
 
        def fn(q, k, v):
            q_b = q[0:1].expand(B, -1, -1, -1)
            k_b = k[0:1].expand(B, -1, -1, -1)
            v_b = v[0:1].expand(B, -1, -1, -1)
            out = sdpa_fn(q_b, k_b, v_b, is_causal=True)
            for i in range(1, B):
                torch.testing.assert_close(
                    out[0], out[i],
                    atol=tol["atol"], rtol=tol["rtol"],
                    msg=f"Batch item {i} differs from item 0",
                )
            return out
 
        if compiled:
            compare_with_cpu(fn, q, k, v, compiled=True)
        else:
            compare_sdpa(fn, q, k, v, dtype=q.dtype)
 
    def test_sdpa_gradient_flow(self, q, k, v, compiled):
        """Q gradient back-propagates through SDPA without NaNs (eager only)."""
        def fn_q_only(q, k, v):
            q   = q.detach().requires_grad_(True)
            k   = k.detach()
            v   = v.detach()
            out = sdpa_fn(q, k, v, is_causal=True)
            out.sum().backward()
            assert q.grad is not None, "Gradient for Q is None"
            assert not torch.isnan(q.grad).any(), (
                f"NaN in Q gradient: "
                f"{torch.isnan(q.grad).sum().item()} / {q.grad.numel()}"
            )
            return out
 
        compare_sdpa(fn_q_only, q, k, v, dtype=q.dtype)
 
        q_c = q.detach().requires_grad_(True)
        k_c = k.detach().requires_grad_(True)
        v_c = v.detach().requires_grad_(True)
        sdpa_fn(q_c, k_c, v_c, is_causal=True).sum().backward()
        for name, t in (("K", k_c), ("V", v_c)):
            assert t.grad is not None, f"CPU gradient for {name} is None"
            assert not torch.isnan(t.grad).any(), (
                f"NaN in CPU {name} gradient: "
                f"{torch.isnan(t.grad).sum().item()} / {t.grad.numel()}"
            )
 
    def test_sdpa_determinism(self, q, k, v, compiled):
        """Two consecutive identical SDPA calls must return the same output."""
        def fn(q, k, v):
            out1 = sdpa_fn(q, k, v, is_causal=True)
            out2 = sdpa_fn(q, k, v, is_causal=True)
            torch.testing.assert_close(
                out1, out2,
                msg=(
                    f"Two identical SDPA calls returned different results. "
                    f"Max |\u0394|: {(out1 - out2).abs().max().item():.6f}"
                ),
            )
            return out1
 
        if compiled:
            compare_with_cpu(fn, q, k, v, compiled=True)
        else:
            compare_sdpa(fn, q, k, v, dtype=q.dtype)
 
    def test_sdpa_padding_mask(self, q, k, v, compiled):
        """Positions masked with -inf must receive near-zero attention weight."""
        S = q.shape[2]
 
        def fn(q, k, v):
            k_exp, v_exp = expand_kv(k, v)
            pad_mask = torch.zeros(1, 1, S, S, dtype=q.dtype, device=q.device)
            pad_mask[:, :, :, S // 2:] = float("-inf")
            scores  = torch.matmul(q, k_exp.transpose(-2, -1)) * SCALE
            scores += pad_mask
            weights = torch.softmax(scores.float(), dim=-1)
            max_masked = weights[:, :, :, S // 2:].abs().max()
            assert max_masked < 1e-5, (
                f"Masked positions got non-zero attention: "
                f"{max_masked.item():.2e} (threshold 1e-5)"
            )
            return weights
 
        if compiled:
            compare_with_cpu(fn, q, k, v, compiled=True)
        else:
            compare_sdpa(fn, q, k, v, dtype=q.dtype)
 
    # ── Non-parameterized sanity checks ───────────────────────────────────────
 
    def test_model_config_constants(self):
        """Sanity-check architecture constants sourced from config.json."""
        assert NUM_Q_HEADS        == 32
        assert NUM_KV_HEADS       == 8
        assert GQA_GROUPS         == 4
        assert HEAD_DIM           == 128,   f"HEAD_DIM should be 128, got {HEAD_DIM}"
        assert NUM_LAYERS         == 40
        assert SLIDING_WINDOW     is None,  "This model has no sliding window"
        assert HIDDEN_SIZE        == 5120,  f"HIDDEN_SIZE should be 5120, got {HIDDEN_SIZE}"
        assert INTERMEDIATE_SIZE  == 32768, f"INTERMEDIATE_SIZE should be 32768, got {INTERMEDIATE_SIZE}"
        assert VOCAB_SIZE         == 131_072
        assert abs(SCALE - 1.0 / math.sqrt(128)) < 1e-9
        assert DEFAULT_DTYPE      == torch.float16
 
    def test_target_tensor_shape_and_dtype(self):
        """
        make_qkv must produce the exact target shapes and dtype:
          q : [1, 32,   1, 128]  float16   (decode query)
          k : [1,  8,  38, 128]  float16   (KV cache)
          v : [1,  8,  38, 128]  float16
        """
        q, k, v = make_qkv(1, 1, 38, dtype=DEFAULT_DTYPE, diff="shape_check")
 
        assert q.shape == (1, NUM_Q_HEADS,    1, HEAD_DIM), f"q shape: {tuple(q.shape)}"
        assert k.shape == (1, NUM_KV_HEADS,  38, HEAD_DIM), f"k shape: {tuple(k.shape)}"
        assert v.shape == (1, NUM_KV_HEADS,  38, HEAD_DIM), f"v shape: {tuple(v.shape)}"
        assert q.dtype == torch.float16, f"q dtype: {q.dtype}"
        assert k.dtype == torch.float16, f"k dtype: {k.dtype}"
        assert v.dtype == torch.float16, f"v dtype: {v.dtype}"
 
    def test_no_sliding_window(self):
        """SLIDING_WINDOW must be None for this model."""
        assert SLIDING_WINDOW is None, (
            f"Mistral-Small-3.2-24B has no sliding window; got {SLIDING_WINDOW}"
        )
 
    def test_head_dim_is_explicit_not_derived(self):
        """HEAD_DIM is explicitly 128 in config, NOT hidden_size // num_q_heads."""
        derived = HIDDEN_SIZE // NUM_Q_HEADS   # 5120 // 32 = 160
        assert HEAD_DIM == 128
        assert HEAD_DIM != derived, (
            "HEAD_DIM coincidentally equals hidden_size // num_q_heads — "
            "this test documents that the config sets it explicitly."
        )
 
    def test_gqa_groups_correct(self):
        """GQA_GROUPS = NUM_Q_HEADS / NUM_KV_HEADS = 4."""
        assert GQA_GROUPS == 4
        assert NUM_Q_HEADS == GQA_GROUPS * NUM_KV_HEADS
# ═════════════════════════════════════════════════════════════════════════════
#  TestReshape  —  torch.reshape
# ═════════════════════════════════════════════════════════════════════════════
 
class TestReshape(unittest.TestCase, metaclass=ParameterizedTestMeta):
    """
    CPU vs Spyre tests for torch.reshape —
    Mistral-Small-3.2-24B-Instruct-2506.
 
    Source op call-sites in the model (modeling_mistral.py)
    -------------------------------------------------------
    A) attn_output.reshape(*input_shape, -1).contiguous()       — Line 173
       Attention heads merged back: [B, S, 32, 128] → [B, S, 4096]
       Note: 4096 = NUM_Q_HEADS * HEAD_DIM (32 × 128), not HIDDEN_SIZE.
 
    B) Grouped KV reshape for GQA key/value stacking:
       (B, 8, 4, S_kv, 128) → (B, 32, S_kv, 128)
 
    C) FFN intermediate reshape: (B, S, 32768) → (B*S, 32768) and back.
 
    D) Hidden-state flat reshape: (B, S, 5120) → (-1, 5120).
 
    Group naming mirrors the 24B test file:
      A — attention output (heads merged)
      X — grouped KV
      L — non-contiguous (transpose → reshape)
      M — reshape().contiguous() chain
      P — multi-dtype
      H — hidden-state / FFN shapes
      S — CPU-only contiguity assertion
    """
 
    pytestmark = pytest.mark.torch_reshape
    torch.manual_seed(0)
 
    # Helper: 4096 = NUM_Q_HEADS * HEAD_DIM
    _ATTN_OUT_DIM = NUM_Q_HEADS * HEAD_DIM   # 4096
 
    PARAMS = {
 
        # ── GROUP A: attn output [B,S,32,128] → [B,S,4096] ──────────────────
        ("test_torch_reshape_A000", "_run_reshape_test"): {
            "param_sets": {
                "decode_1x1x32x128_eager":    (_t((1,  1, 32, 128)), (1,  1, -1), False),
                "decode_1x1x32x128_compiled": (_t((1,  1, 32, 128)), (1,  1, -1), True),
            }
        },
        ("test_torch_reshape_A001", "_run_reshape_test"): {
            "param_sets": {
                "prefill_1x38x32x128_eager":    (_t((1, 38, 32, 128)), (1, 38, -1), False),
                "prefill_1x38x32x128_compiled": (_t((1, 38, 32, 128)), (1, 38, -1), True),
            }
        },
        ("test_torch_reshape_A002", "_run_reshape_test"): {
            "param_sets": {
                "prefill_1x128x32x128_eager":    (_t((1, 128, 32, 128)), (1, 128, -1), False),
                "prefill_1x128x32x128_compiled": (_t((1, 128, 32, 128)), (1, 128, -1), True),
            }
        },
        ("test_torch_reshape_A003", "_run_reshape_test"): {
            "param_sets": {
                "batch4_decode_4x1x32x128_eager":    (_t((4,  1, 32, 128)), (4,  1, -1), False),
                "batch4_decode_4x1x32x128_compiled": (_t((4,  1, 32, 128)), (4,  1, -1), True),
            }
        },
        ("test_torch_reshape_A004", "_run_reshape_test"): {
            "param_sets": {
                "batch2_prefill_2x38x32x128_eager":    (_t((2, 38, 32, 128)), (2, 38, -1), False),
                "batch2_prefill_2x38x32x128_compiled": (_t((2, 38, 32, 128)), (2, 38, -1), True),
            }
        },
        # KV head: [1,1,8,128] → [1,1,1024]
        ("test_torch_reshape_A005", "_run_reshape_test"): {
            "param_sets": {
                "kv_decode_1x1x8x128_eager":    (_t((1,  1, 8, 128)), (1,  1, -1), False),
                "kv_decode_1x1x8x128_compiled": (_t((1,  1, 8, 128)), (1,  1, -1), True),
            }
        },
 
        # ── GROUP X: Grouped KV reshape (B,8,4,S,128) → (B,32,S,128) ────────
        ("test_torch_reshape_X000", "_run_reshape_test"): {
            "param_sets": {
                "grouped_kv_prefill_1x8x4x38x128_eager":    (_t((1, 8, 4, 38, 128)),  (1, 32, 38,  128), False),
                "grouped_kv_prefill_1x8x4x38x128_compiled": (_t((1, 8, 4, 38, 128)),  (1, 32, 38,  128), True),
            }
        },
        ("test_torch_reshape_X001", "_run_reshape_test"): {
            "param_sets": {
                "grouped_kv_prefill_1x8x4x128x128_eager":    (_t((1, 8, 4, 128, 128)), (1, 32, 128, 128), False),
                "grouped_kv_prefill_1x8x4x128x128_compiled": (_t((1, 8, 4, 128, 128)), (1, 32, 128, 128), True),
            }
        },
        ("test_torch_reshape_X002", "_run_reshape_test"): {
            "param_sets": {
                "grouped_kv_decode_1x8x4x1x128_eager":    (_t((1, 8, 4, 1, 128)), (1, 32, 1, 128), False),
                "grouped_kv_decode_1x8x4x1x128_compiled": (_t((1, 8, 4, 1, 128)), (1, 32, 1, 128), True),
            }
        },
 
        # ── GROUP L: non-contiguous (transpose → reshape) ─────────────────────
        ("test_torch_reshape_L000", "_run_reshape_after_transpose_test"): {
            "param_sets": {
                "noncontig_q_decode_eager":    (_t((1, 32,  1, 128)), 1, 2, (1,  1, -1), False),
                "noncontig_q_decode_compiled": (_t((1, 32,  1, 128)), 1, 2, (1,  1, -1), True),
            }
        },
        ("test_torch_reshape_L001", "_run_reshape_after_transpose_test"): {
            "param_sets": {
                "noncontig_q_prefill38_eager":    (_t((1, 32, 38, 128)), 1, 2, (1, 38, -1), False),
                "noncontig_q_prefill38_compiled": (_t((1, 32, 38, 128)), 1, 2, (1, 38, -1), True),
            }
        },
        ("test_torch_reshape_L002", "_run_reshape_after_transpose_test"): {
            "param_sets": {
                "noncontig_q_prefill128_eager":    (_t((1, 32, 128, 128)), 1, 2, (1, 128, -1), False),
                "noncontig_q_prefill128_compiled": (_t((1, 32, 128, 128)), 1, 2, (1, 128, -1), True),
            }
        },
        ("test_torch_reshape_L003", "_run_reshape_after_transpose_test"): {
            "param_sets": {
                "noncontig_kv_decode_eager":    (_t((1,  8,  1, 128)), 1, 2, (1,  1, -1), False),
                "noncontig_kv_decode_compiled": (_t((1,  8,  1, 128)), 1, 2, (1,  1, -1), True),
            }
        },
        ("test_torch_reshape_L004", "_run_reshape_after_transpose_test"): {
            "param_sets": {
                "noncontig_kv_prefill38_eager":    (_t((1,  8, 38, 128)), 1, 2, (1, 38, -1), False),
                "noncontig_kv_prefill38_compiled": (_t((1,  8, 38, 128)), 1, 2, (1, 38, -1), True),
            }
        },
        ("test_torch_reshape_L005", "_run_reshape_after_transpose_test"): {
            "param_sets": {
                "noncontig_batch2_q_prefill_eager":    (_t((2, 32, 38, 128)), 1, 2, (2, 38, -1), False),
                "noncontig_batch2_q_prefill_compiled": (_t((2, 32, 38, 128)), 1, 2, (2, 38, -1), True),
            }
        },
 
        # ── GROUP M: .reshape().contiguous() — full model op chain ────────────
        ("test_torch_reshape_M000", "_run_reshape_contiguous_test"): {
            "param_sets": {
                "chain_decode_eager":    (_t((1,  1, 32, 128)), (1,  1, -1), False),
                "chain_decode_compiled": (_t((1,  1, 32, 128)), (1,  1, -1), True),
            }
        },
        ("test_torch_reshape_M001", "_run_reshape_contiguous_test"): {
            "param_sets": {
                "chain_prefill38_eager":    (_t((1, 38, 32, 128)), (1, 38, -1), False),
                "chain_prefill38_compiled": (_t((1, 38, 32, 128)), (1, 38, -1), True),
            }
        },
        ("test_torch_reshape_M002", "_run_reshape_contiguous_test"): {
            "param_sets": {
                "chain_batch2_prefill_eager":    (_t((2, 38, 32, 128)), (2, 38, -1), False),
                "chain_batch2_prefill_compiled": (_t((2, 38, 32, 128)), (2, 38, -1), True),
            }
        },
 
        # ── GROUP P: multi-dtype ───────────────────────────────────────────────
        ("test_torch_reshape_P000", "_run_reshape_test"): {
            "param_sets": {
                "bf16_decode_eager":    (_t((1,  1, 32, 128), torch.bfloat16), (1,  1, -1), False),
                "bf16_decode_compiled": (_t((1,  1, 32, 128), torch.bfloat16), (1,  1, -1), True),
            }
        },
        ("test_torch_reshape_P001", "_run_reshape_test"): {
            "param_sets": {
                "bf16_prefill_eager":    (_t((1, 38, 32, 128), torch.bfloat16), (1, 38, -1), False),
                "bf16_prefill_compiled": (_t((1, 38, 32, 128), torch.bfloat16), (1, 38, -1), True),
            }
        },
        ("test_torch_reshape_P002", "_run_reshape_test"): {
            "param_sets": {
                "fp16_decode_eager":    (_t((1,  1, 32, 128), torch.float16), (1,  1, -1), False),
                "fp16_decode_compiled": (_t((1,  1, 32, 128), torch.float16), (1,  1, -1), True),
            }
        },
        ("test_torch_reshape_P003", "_run_reshape_test"): {
            "param_sets": {
                "fp32_decode_eager":    (_t((1,  1, 32, 128), torch.float32), (1,  1, -1), False),
                "fp32_decode_compiled": (_t((1,  1, 32, 128), torch.float32), (1,  1, -1), True),
            }
        },
        ("test_torch_reshape_P004", "_run_reshape_test"): {
            "param_sets": {
                "int32_decode_eager":    (_t((1,  1, 32, 128), torch.int32), (1,  1, -1), False),
                "int32_decode_compiled": (_t((1,  1, 32, 128), torch.int32), (1,  1, -1), True),
            }
        },
 
        # ── GROUP H: hidden-state / FFN shapes ────────────────────────────────
        # HIDDEN_SIZE = 5120, INTERMEDIATE_SIZE = 32768
        ("test_torch_reshape_H000", "_run_reshape_test"): {
            "param_sets": {
                "hidden_decode_flat_eager":    (_t((1,  1, 5120)), (-1, 5120), False),
                "hidden_decode_flat_compiled": (_t((1,  1, 5120)), (-1, 5120), True),
            }
        },
        ("test_torch_reshape_H001", "_run_reshape_test"): {
            "param_sets": {
                "hidden_prefill38_flat_eager":    (_t((1, 38, 5120)), (-1, 5120), False),
                "hidden_prefill38_flat_compiled": (_t((1, 38, 5120)), (-1, 5120), True),
            }
        },
        ("test_torch_reshape_H002", "_run_reshape_test"): {
            "param_sets": {
                "ffn_intermediate_flat_eager":    (_t((1, 38, 32768)), (38, -1), False),
                "ffn_intermediate_flat_compiled": (_t((1, 38, 32768)), (38, -1), True),
            }
        },
        # FFN: (B, S, intermediate) → (B*S, intermediate)
        ("test_torch_reshape_H003", "_run_reshape_test"): {
            "param_sets": {
                "ffn_batch2_flat_eager":    (_t((2, 38, 32768)), (76, -1), False),
                "ffn_batch2_flat_compiled": (_t((2, 38, 32768)), (76, -1), True),
            }
        },
        # Restore: (B*S, intermediate) → (B, S, intermediate)
        ("test_torch_reshape_H004", "_run_reshape_test"): {
            "param_sets": {
                "ffn_restore_eager":    (_t((38, 32768)), (1, 38, -1), False),
                "ffn_restore_compiled": (_t((38, 32768)), (1, 38, -1), True),
            }
        },
 
        # ── GROUP S: CPU-only contiguity structural assertion ─────────────────
        ("test_torch_reshape_S000", "_run_reshape_contiguity_test"): {
            "param_sets": {
                "contiguity_decode":  (_t((1,  1, 32, 128)), (1,  1, -1)),
                "contiguity_prefill": (_t((1, 38, 32, 128)), (1, 38, -1)),
                "contiguity_hidden":  (_t((1, 38, 5120)),    (-1, 5120)),
            }
        },
    }
 
    def _run_reshape_test(self, tensor, target_shape, compiled):
        """tensor.reshape(*target_shape) — eager or compiled."""
        compare_with_cpu(lambda t: t.reshape(*target_shape), tensor, compiled=compiled)
 
    def _run_reshape_contiguous_test(self, tensor, target_shape, compiled):
        """.reshape().contiguous() — full model op chain."""
        compare_with_cpu(
            lambda t: t.reshape(*target_shape).contiguous(),
            tensor, compiled=compiled,
        )
 
    def _run_reshape_after_transpose_test(self, tensor, d0, d1, target_shape, compiled):
        """tensor.transpose(d0,d1).reshape(*target_shape) — non-contiguous input."""
        compare_with_cpu(
            lambda t: t.transpose(d0, d1).reshape(*target_shape),
            tensor, compiled=compiled,
        )
 
    def _run_reshape_contiguity_test(self, tensor, target_shape):
        """CPU-only: contiguous input → contiguous output, same numel."""
        t = tensor.cpu()
        assert t.is_contiguous()
        result = t.reshape(*target_shape)
        assert result.is_contiguous(), (
            f"reshape {tuple(t.shape)} → {target_shape} is not contiguous"
        )
        assert result.numel() == t.numel()
 
    # ── Non-parameterized sanity checks ───────────────────────────────────────
 
    def test_reshape_numel_preserved(self):
        """reshape must never change element count."""
        for shape, new_shape in [
            ((1,  1, 32, 128), (1,  1, 4096)),
            ((1, 38, 32, 128), (1, 38, 4096)),
            ((1, 38, 5120),    (38, 5120)),
            ((1, 38, 32768),   (38, 32768)),
        ]:
            t = _t(shape)
            r = t.reshape(*new_shape)
            assert r.numel() == t.numel(), (
                f"numel changed: {t.numel()} → {r.numel()} "
                f"for {shape} → {new_shape}"
            )
 
    def test_reshape_attn_out_dim(self):
        """Attention output dim = NUM_Q_HEADS * HEAD_DIM = 32*128 = 4096."""
        assert NUM_Q_HEADS * HEAD_DIM == 4096
        t = _t((1, 38, NUM_Q_HEADS, HEAD_DIM))
        r = t.reshape(1, 38, -1)
        assert r.shape == (1, 38, 4096)
 
 
# ═════════════════════════════════════════════════════════════════════════════
#  TestFunctionalSilu  —  torch.nn.functional.silu
# ═════════════════════════════════════════════════════════════════════════════
 
class TestFunctionalSilu(unittest.TestCase, metaclass=ParameterizedTestMeta):
    """
    CPU vs Spyre tests for torch.nn.functional.silu —
    Mistral-Small-3.2-24B-Instruct-2506.
 
    SiLU is used in every FFN block as the SwiGLU gate activation:
        F.silu(gate_proj(x)) * up_proj(x)
 
    Model dimensions (from config.json)
    ------------------------------------
    INTERMEDIATE_SIZE : 32768   (gate/up projection output)
    HIDDEN_SIZE       : 5120    (hidden-state dimension)
 
    Sequence lengths from traced ops: 38 (prefill), 1 (decode).
 
    Group layout (mirrors 14B test file)
    -------------------------------------
      A — FFN gate decode       [B,1,INTERMEDIATE]
      B — FFN gate prefill      [B,S,INTERMEDIATE]
      C — hidden gate           [B,S,HIDDEN]
      D — SwiGLU product        F.silu(gate) * up
      E — multi-dtype
      F — non-contiguous input  (transpose → silu)
      G — CPU-only IEEE 754 special values
      H — CPU-only identity     F.silu(x) == x*sigmoid(x)
    """
 
    pytestmark = pytest.mark.torch_nn_functional_silu
    torch.manual_seed(0xCAFE_2506)
 
    _INTERMEDIATE = INTERMEDIATE_SIZE   # 32768
    _HIDDEN       = HIDDEN_SIZE         # 5120
 
    PARAMS = {
 
        # ── GROUP A: FFN gate decode [B,1,INTERMEDIATE] ──────────────────────
        ("test_silu_A000", "_run_silu_ffn_gate_test"): {
            "param_sets": {
                "decode_1x1x32768_eager":    (_t((1, 1, 32768)), False),
                "decode_1x1x32768_compiled": (_t((1, 1, 32768)), True),
            }
        },
        ("test_silu_A001", "_run_silu_ffn_gate_test"): {
            "param_sets": {
                "decode_batch2_2x1x32768_eager":    (_t((2, 1, 32768)), False),
                "decode_batch2_2x1x32768_compiled": (_t((2, 1, 32768)), True),
            }
        },
        ("test_silu_A002", "_run_silu_ffn_gate_test"): {
            "param_sets": {
                "decode_batch4_4x1x32768_eager":    (_t((4, 1, 32768)), False),
                "decode_batch4_4x1x32768_compiled": (_t((4, 1, 32768)), True),
            }
        },
        ("test_silu_A003", "_run_silu_ffn_gate_test"): {
            "param_sets": {
                "decode_batch8_8x1x32768_eager":    (_t((8, 1, 32768)), False),
                "decode_batch8_8x1x32768_compiled": (_t((8, 1, 32768)), True),
            }
        },
 
        # ── GROUP B: FFN gate prefill [B,S,INTERMEDIATE] ─────────────────────
        ("test_silu_B000", "_run_silu_ffn_gate_test"): {
            "param_sets": {
                "prefill_1x38x32768_eager":    (_t((1,  38, 32768)), False),
                "prefill_1x38x32768_compiled": (_t((1,  38, 32768)), True),
            }
        },
        ("test_silu_B001", "_run_silu_ffn_gate_test"): {
            "param_sets": {
                "prefill_1x128x32768_eager":    (_t((1, 128, 32768)), False),
                "prefill_1x128x32768_compiled": (_t((1, 128, 32768)), True),
            }
        },
        ("test_silu_B002", "_run_silu_ffn_gate_test"): {
            "param_sets": {
                "prefill_1x512x32768_eager":    (_t((1, 512, 32768)), False),
                "prefill_1x512x32768_compiled": (_t((1, 512, 32768)), True),
            }
        },
        ("test_silu_B003", "_run_silu_ffn_gate_test"): {
            "param_sets": {
                "prefill_batch2_2x38x32768_eager":    (_t((2, 38, 32768)), False),
                "prefill_batch2_2x38x32768_compiled": (_t((2, 38, 32768)), True),
            }
        },
 
        # ── GROUP C: hidden-state gate [B,S,HIDDEN] ───────────────────────────
        ("test_silu_C000", "_run_silu_ffn_gate_test"): {
            "param_sets": {
                "hidden_decode_1x1x5120_eager":    (_t((1,  1, 5120)), False),
                "hidden_decode_1x1x5120_compiled": (_t((1,  1, 5120)), True),
            }
        },
        ("test_silu_C001", "_run_silu_ffn_gate_test"): {
            "param_sets": {
                "hidden_prefill_1x38x5120_eager":    (_t((1, 38, 5120)), False),
                "hidden_prefill_1x38x5120_compiled": (_t((1, 38, 5120)), True),
            }
        },
 
        # ── GROUP D: SwiGLU product F.silu(gate) * up ─────────────────────────
        ("test_silu_D000", "_run_silu_swiglu_test"): {
            "param_sets": {
                "swiglu_decode_1x1x32768_eager":    (_t((1,  1, 32768)), _t((1,  1, 32768)), False),
                "swiglu_decode_1x1x32768_compiled": (_t((1,  1, 32768)), _t((1,  1, 32768)), True),
            }
        },
        ("test_silu_D001", "_run_silu_swiglu_test"): {
            "param_sets": {
                "swiglu_prefill_1x38x32768_eager":    (_t((1, 38, 32768)), _t((1, 38, 32768)), False),
                "swiglu_prefill_1x38x32768_compiled": (_t((1, 38, 32768)), _t((1, 38, 32768)), True),
            }
        },
        ("test_silu_D002", "_run_silu_swiglu_test"): {
            "param_sets": {
                "swiglu_prefill_1x128x32768_eager":    (_t((1, 128, 32768)), _t((1, 128, 32768)), False),
                "swiglu_prefill_1x128x32768_compiled": (_t((1, 128, 32768)), _t((1, 128, 32768)), True),
            }
        },
 
        # ── GROUP E: multi-dtype ───────────────────────────────────────────────
        ("test_silu_E000", "_run_silu_ffn_gate_test"): {
            "param_sets": {
                "bf16_decode_1x1x32768_eager":    (_t((1, 1, 32768), torch.bfloat16), False),
                "bf16_decode_1x1x32768_compiled": (_t((1, 1, 32768), torch.bfloat16), True),
            }
        },
        ("test_silu_E001", "_run_silu_ffn_gate_test"): {
            "param_sets": {
                "fp16_decode_1x1x32768_eager":    (_t((1, 1, 32768), torch.float16), False),
                "fp16_decode_1x1x32768_compiled": (_t((1, 1, 32768), torch.float16), True),
            }
        },
        ("test_silu_E002", "_run_silu_ffn_gate_test"): {
            "param_sets": {
                "fp32_decode_1x1x32768_eager":    (_t((1, 1, 32768), torch.float32), False),
                "fp32_decode_1x1x32768_compiled": (_t((1, 1, 32768), torch.float32), True),
            }
        },
        ("test_silu_E003", "_run_silu_swiglu_test"): {
            "param_sets": {
                "bf16_swiglu_1x38x32768_eager":    (
                    _t((1, 38, 32768), torch.bfloat16),
                    _t((1, 38, 32768), torch.bfloat16),
                    False,
                ),
                "bf16_swiglu_1x38x32768_compiled": (
                    _t((1, 38, 32768), torch.bfloat16),
                    _t((1, 38, 32768), torch.bfloat16),
                    True,
                ),
            }
        },
 
        # ── GROUP F: non-contiguous input (transpose → silu) ─────────────────
        ("test_silu_F000", "_run_silu_noncontig_test"): {
            "param_sets": {
                "noncontig_gate_decode_eager":    (_t((1, 32768, 1)), 0, 1, False),
                "noncontig_gate_decode_compiled": (_t((1, 32768, 1)), 0, 1, True),
            }
        },
        ("test_silu_F001", "_run_silu_noncontig_test"): {
            "param_sets": {
                "noncontig_hidden_prefill_eager":    (_t((1, 5120, 38)), 1, 2, False),
                "noncontig_hidden_prefill_compiled": (_t((1, 5120, 38)), 1, 2, True),
            }
        },
 
        # ── GROUP G: CPU-only special IEEE 754 values ─────────────────────────
        ("test_silu_G000", "_run_silu_special_values_test"): {
            "param_sets": {
                "special_mixed_fp32": (
                    torch.tensor(
                        [0.0, 1.0, -1.0, float("inf"), float("-inf"),
                         float("nan"), 10.0, -10.0, 65504.0, -65504.0],
                        dtype=torch.float32,
                    ),
                ),
            }
        },
 
        # ── GROUP H: CPU-only numerical identity F.silu(x) == x*sigmoid(x) ───
        ("test_silu_H000", "_run_silu_identity_check_test"): {
            "param_sets": {
                "identity_gate_decode_fp32":  (_t((1,  1, 32768), torch.float32),),
                "identity_gate_prefill_fp32": (_t((1, 38, 32768), torch.float32),),
                "identity_hidden_fp32":       (_t((1, 38,  5120), torch.float32),),
            }
        },
    }
 
    # ── Base test methods ─────────────────────────────────────────────────────
 
    def _run_silu_ffn_gate_test(self, tensor, compiled):
        """F.silu(gate) — FFN gate activation, eager or compiled."""
        compare_with_cpu(
            lambda t: torch.nn.functional.silu(t),
            tensor, compiled=compiled,
        )
 
    def _run_silu_swiglu_test(self, gate, up, compiled):
        """F.silu(gate) * up — full SwiGLU product, eager or compiled."""
        compare_with_cpu(
            lambda g, u: torch.nn.functional.silu(g) * u,
            gate, up, compiled=compiled,
        )
 
    def _run_silu_noncontig_test(self, tensor, d0, d1, compiled):
        """F.silu on a non-contiguous view produced by transpose."""
        compare_with_cpu(
            lambda t: torch.nn.functional.silu(t.transpose(d0, d1).contiguous()),
            tensor, compiled=compiled,
        )
 
    def _run_silu_special_values_test(self, tensor):
        """CPU-only: IEEE 754 special-value behaviour of F.silu."""
        t      = tensor.cpu().float()
        result = torch.nn.functional.silu(t).float()
        for idx in range(t.numel()):
            raw = t.view(-1)[idx].item()
            got = result.view(-1)[idx].item()
            if math.isnan(raw):
                assert math.isnan(got), f"silu(NaN) should be NaN, got {got}"
            elif raw == float("inf"):
                assert got == float("inf"), f"silu(+inf) should be +inf, got {got}"
            elif raw == float("-inf"):
                assert math.isnan(got), f"silu(-inf) should be NaN (IEEE 754), got {got}"
            elif raw == 0.0:
                assert got == 0.0, f"silu(0) should be 0.0, got {got}"
            else:
                if got == 0.0:
                    pass   # signed-zero underflow is valid IEEE 754 behaviour
                else:
                    assert (raw >= 0) == (got >= 0), \
                        f"silu({raw}) sign wrong: got {got}"
 
    def _run_silu_identity_check_test(self, tensor):
        """CPU-only: F.silu(x) == x * sigmoid(x) element-wise (fp32)."""
        t = tensor.cpu().float()
        torch.testing.assert_close(
            torch.nn.functional.silu(t),
            t * torch.sigmoid(t),
            atol=1e-5, rtol=1e-5,
            msg=lambda msg: f"F.silu(x) != x*sigmoid(x) on {tuple(t.shape)}\n\n{msg}\n",
        )
 
    # ── Non-parameterized sanity checks ───────────────────────────────────────
 
    def test_silu_zero_fixed_point(self):
        """silu(0) == 0 for bf16, fp16, fp32."""
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            assert torch.nn.functional.silu(torch.zeros(1, dtype=dtype)).item() == 0.0
 
    def test_silu_shape_preserved_model_shapes(self):
        """F.silu must not alter shape for canonical model shapes."""
        for shape in [
            (1,  1, 32768), (1, 38, 32768), (1, 128, 32768),
            (1,  1,  5120), (1, 38,  5120),
        ]:
            t = _t(shape)
            assert torch.nn.functional.silu(t).shape == t.shape
 
    def test_silu_swiglu_matches_decomposition(self):
        """F.silu(gate)*up == (gate*sigmoid(gate))*up in fp32."""
        gate = _t((1, 38, 32768), torch.float32)
        up   = _t((1, 38, 32768), torch.float32)
        torch.testing.assert_close(
            torch.nn.functional.silu(gate) * up,
            (gate * torch.sigmoid(gate)) * up,
            atol=1e-5, rtol=1e-5,
        )
 
    def test_silu_intermediate_size_correct(self):
        """INTERMEDIATE_SIZE must be 32768 for this model (not 16384 from 14B)."""
        assert INTERMEDIATE_SIZE == 32768, (
            f"Wrong INTERMEDIATE_SIZE: expected 32768, got {INTERMEDIATE_SIZE}. "
            "Check that utils_mistral_small_3_2_24b.py is imported, not utils.py."
        )
 
# ─────────────────────────────────────────────────────────────────────────────
# TestCat
# ─────────────────────────────────────────────────────────────────────────────

class TestCat(unittest.TestCase, metaclass=ParameterizedTestMeta):
    """
    Tests for torch.cat patterns observed in
    mistralai/Mistral-Small-3.2-24B-Instruct-2506.

    Five distinct call-sites appear in the traced model:

      cat.1  RoPE freqs concat (decode)   : [(1,1,64),  (1,1,64)]  → (1,1,128)   float32  dim=-1
      cat.2  rotate_half q (decode)       : [(1,32,1,64),(1,32,1,64)] → (1,32,1,128) bf16  dim=-1
      cat.3  rotate_half k (decode)       : [(1,8,1,64), (1,8,1,64)]  → (1,8,1,128)  bf16  dim=-1
      cat.4  KV-cache append (decode)     : [(1,8,38,128),(1,8,1,128)]→ (1,8,39,128) bf16  dim=-2
      cat.5  RoPE freqs concat (prefill)  : [(1,38,64), (1,38,64)]  → (1,38,128)  float32  dim=-1
      cat.6  rotate_half q (prefill)      : [(1,32,38,64),(1,32,38,64)]→(1,32,38,128) bf16 dim=-1
      cat.7  rotate_half k (prefill)      : [(1,8,38,64),(1,8,38,64)] → (1,8,38,128) bf16 dim=-1

    Source lines (modeling_mistral.py):
      Line 320 : emb = torch.cat((freqs, freqs), dim=-1)
      Line 55  : return torch.cat((-x2, x1), dim=-1)   [rotate_half q & k]
      cache_utils.py:119 : self.keys = torch.cat([self.keys, key_states], dim=-2)
    """

    pytestmark = pytest.mark.torch_cat
    torch.manual_seed(0)

    PARAMS = {

        # ── cat.1  RoPE freqs concat — DECODE ────────────────────────────────
        # [(1,1,64), (1,1,64)] dim=-1 → (1,1,128)   float32
        ("test_torch_cat_pattern_000", "_run_cat_test"): {
            "param_sets": {
                "rope_freqs_decode_eager": (
                    [torch.randn(1, 1, 64, dtype=F32),
                     torch.randn(1, 1, 64, dtype=F32)],
                    -1, False,
                ),
                "rope_freqs_decode_compiled": (
                    [torch.randn(1, 1, 64, dtype=F32),
                     torch.randn(1, 1, 64, dtype=F32)],
                    -1, True,
                ),
            }
        },

        # ── cat.2  rotate_half q — DECODE ────────────────────────────────────
        # [(1,32,1,64), (1,32,1,64)] dim=-1 → (1,32,1,128)   bf16
        ("test_torch_cat_pattern_001", "_run_cat_test"): {
            "param_sets": {
                "rotate_half_q_decode_eager": (
                    [torch.randn(1, 32, 1, 64, dtype=BF16),
                     torch.randn(1, 32, 1, 64, dtype=BF16)],
                    -1, False,
                ),
                "rotate_half_q_decode_compiled": (
                    [torch.randn(1, 32, 1, 64, dtype=BF16),
                     torch.randn(1, 32, 1, 64, dtype=BF16)],
                    -1, True,
                ),
            }
        },

        # ── cat.3  rotate_half k — DECODE ────────────────────────────────────
        # [(1,8,1,64), (1,8,1,64)] dim=-1 → (1,8,1,128)   bf16
        ("test_torch_cat_pattern_002", "_run_cat_test"): {
            "param_sets": {
                "rotate_half_k_decode_eager": (
                    [torch.randn(1, 8, 1, 64, dtype=BF16),
                     torch.randn(1, 8, 1, 64, dtype=BF16)],
                    -1, False,
                ),
                "rotate_half_k_decode_compiled": (
                    [torch.randn(1, 8, 1, 64, dtype=BF16),
                     torch.randn(1, 8, 1, 64, dtype=BF16)],
                    -1, True,
                ),
            }
        },

        # ── cat.4  KV-cache append — DECODE ──────────────────────────────────
        # [(1,8,38,128), (1,8,1,128)] dim=-2 → (1,8,39,128)   bf16
        # (past KV cache of 38 tokens + 1 new token)
        ("test_torch_cat_pattern_003", "_run_cat_test"): {
            "param_sets": {
                "kvcache_append_decode_eager": (
                    [torch.randn(1, 8, 38, 128, dtype=BF16),
                     torch.randn(1, 8,  1, 128, dtype=BF16)],
                    -2, False,
                ),
                "kvcache_append_decode_compiled": (
                    [torch.randn(1, 8, 38, 128, dtype=BF16),
                     torch.randn(1, 8,  1, 128, dtype=BF16)],
                    -2, True,
                ),
            }
        },

        # ── cat.5  RoPE freqs concat — PREFILL ───────────────────────────────
        # [(1,38,64), (1,38,64)] dim=-1 → (1,38,128)   float32
        ("test_torch_cat_pattern_004", "_run_cat_test"): {
            "param_sets": {
                "rope_freqs_prefill_eager": (
                    [torch.randn(1, 38, 64, dtype=F32),
                     torch.randn(1, 38, 64, dtype=F32)],
                    -1, False,
                ),
                "rope_freqs_prefill_compiled": (
                    [torch.randn(1, 38, 64, dtype=F32),
                     torch.randn(1, 38, 64, dtype=F32)],
                    -1, True,
                ),
            }
        },

        # ── cat.6  rotate_half q — PREFILL ───────────────────────────────────
        # [(1,32,38,64), (1,32,38,64)] dim=-1 → (1,32,38,128)   bf16
        ("test_torch_cat_pattern_005", "_run_cat_test"): {
            "param_sets": {
                "rotate_half_q_prefill_eager": (
                    [torch.randn(1, 32, 38, 64, dtype=BF16),
                     torch.randn(1, 32, 38, 64, dtype=BF16)],
                    -1, False,
                ),
                "rotate_half_q_prefill_compiled": (
                    [torch.randn(1, 32, 38, 64, dtype=BF16),
                     torch.randn(1, 32, 38, 64, dtype=BF16)],
                    -1, True,
                ),
            }
        },

        # ── cat.7  rotate_half k — PREFILL ───────────────────────────────────
        # [(1,8,38,64), (1,8,38,64)] dim=-1 → (1,8,38,128)   bf16
        ("test_torch_cat_pattern_006", "_run_cat_test"): {
            "param_sets": {
                "rotate_half_k_prefill_eager": (
                    [torch.randn(1, 8, 38, 64, dtype=BF16),
                     torch.randn(1, 8, 38, 64, dtype=BF16)],
                    -1, False,
                ),
                "rotate_half_k_prefill_compiled": (
                    [torch.randn(1, 8, 38, 64, dtype=BF16),
                     torch.randn(1, 8, 38, 64, dtype=BF16)],
                    -1, True,
                ),
            }
        },

        # ── Extra coverage: varying KV-cache lengths ──────────────────────────
        # Simulate a smaller initial cache (e.g. first decode step, past=0)
        ("test_torch_cat_pattern_007", "_run_cat_test"): {
            "param_sets": {
                "kvcache_empty_past_eager": (
                    [torch.zeros(0, dtype=BF16),
                     torch.randn(1, 8, 1, 128, dtype=BF16)],
                    -2, False,
                ),
                "kvcache_empty_past_compiled": (
                    [torch.zeros(0, dtype=BF16),
                     torch.randn(1, 8, 1, 128, dtype=BF16)],
                    -2, True,
                ),
            }
        },
    }

    def _run_cat_test(self, tensors, dim, compiled):
        """torch.cat(tensors, dim=dim) — eager or compiled."""
        def cat_fn(*ts):
            return torch.cat(list(ts), dim=dim)

        compare_with_cpu(cat_fn, *tensors, compiled=compiled)


# ─────────────────────────────────────────────────────────────────────────────
# TestMul
# ─────────────────────────────────────────────────────────────────────────────

class TestMul(unittest.TestCase, metaclass=ParameterizedTestMeta):
    """
    Tests for torch.mul patterns observed in
    mistralai/Mistral-Small-3.2-24B-Instruct-2506.

    All call-sites extracted from the flat tables in mistral_ops_decode.txt
    and mistral_ops_prefill.txt.  Three signatures appear:

      binary       torch.mul(tensor, tensor)   — same or broadcast shapes
      scalar_right torch.mul(tensor, scalar)   — attention_scaling

    DECODE ops covered:
      mul.D1  [1,1,128]    * scalar             float32  attention_scaling (RoPE cos)
      mul.D2  [1,1,5120]   * [1,1,1]            float32  rsqrt normalisation
      mul.D3  [5120]       * [1,1,5120]          bf16    weight * hidden (broadcast)
      mul.D4  [1,32,1,128] * [1,1,1,128]         bf16    q * cos (broadcast)
      mul.D5  [1,8,1,128]  * [1,1,1,128]         bf16    k * cos (broadcast)
      mul.D6  [1,1,32768]  * [1,1,32768]         bf16    gate * up (elementwise)

    PREFILL ops covered:
      mul.P1  [1,38,128]   * scalar             float32  attention_scaling (RoPE cos)
      mul.P2  [1,38,5120]  * [1,38,1]           float32  rsqrt normalisation
      mul.P3  [5120]       * [1,38,5120]         bf16    weight * hidden (broadcast)
      mul.P4  [1,32,38,128]* [1,1,38,128]        bf16    q * cos (broadcast)
      mul.P5  [1,8,38,128] * [1,1,38,128]        bf16    k * cos (broadcast)
      mul.P6  [1,38,32768] * [1,38,32768]        bf16    gate * up (elementwise)

    Source lines:
      modeling_mistral.py:321  cos = emb.cos() * self.attention_scaling
      modeling_mistral.py:195  hidden_states * torch.rsqrt(variance + eps)
      modeling_mistral.py:196  self.weight * hidden_states.to(input_dtype)
      modeling_mistral.py:79   q_embed = (q * cos) + (rotate_half(q) * sin)
      modeling_mistral.py:80   k_embed = (k * cos) + (rotate_half(k) * sin)
      modeling_mistral.py:47   self.act_fn(self.gate_proj(x)) * self.up_proj(x)
    """

    pytestmark = pytest.mark.torch_mul
    torch.manual_seed(0)

    PARAMS = {

        # ── DECODE patterns ───────────────────────────────────────────────────

        # mul.D1  [1,1,128] * scalar  — attention_scaling, RoPE cos decode
        ("test_torch_mul_pattern_000", "_run_mul_scalar_right"): {
            "param_sets": {
                "scalar_right_1x1x128_decode_eager": (
                    torch.randn(1, 1, 128, dtype=F32), 1.0, False,
                ),
                "scalar_right_1x1x128_decode_compiled": (
                    torch.randn(1, 1, 128, dtype=F32), 1.0, True,
                ),
            }
        },

        # mul.D2  [1,1,5120] * [1,1,1]  — rsqrt normalisation decode
        ("test_torch_mul_pattern_001", "_run_mul_binary"): {
            "param_sets": {
                "binary_1x1x5120_1x1x1_decode_eager": (
                    torch.randn(1, 1, 5120, dtype=F32),
                    torch.randn(1, 1,    1, dtype=F32),
                    False,
                ),
                "binary_1x1x5120_1x1x1_decode_compiled": (
                    torch.randn(1, 1, 5120, dtype=F32),
                    torch.randn(1, 1,    1, dtype=F32),
                    True,
                ),
            }
        },

        # mul.D3  [5120] * [1,1,5120]  — weight * hidden decode (broadcast)
        ("test_torch_mul_pattern_002", "_run_mul_binary"): {
            "param_sets": {
                "binary_5120_1x1x5120_decode_eager": (
                    torch.randn(5120,        dtype=BF16),
                    torch.randn(1, 1, 5120,  dtype=BF16),
                    False,
                ),
                "binary_5120_1x1x5120_decode_compiled": (
                    torch.randn(5120,        dtype=BF16),
                    torch.randn(1, 1, 5120,  dtype=BF16),
                    True,
                ),
            }
        },

        # mul.D4  [1,32,1,128] * [1,1,1,128]  — q * cos decode (broadcast)
        ("test_torch_mul_pattern_003", "_run_mul_binary"): {
            "param_sets": {
                "binary_1x32x1x128_1x1x1x128_decode_eager": (
                    torch.randn(1, 32, 1, 128, dtype=BF16),
                    torch.randn(1,  1, 1, 128, dtype=BF16),
                    False,
                ),
                "binary_1x32x1x128_1x1x1x128_decode_compiled": (
                    torch.randn(1, 32, 1, 128, dtype=BF16),
                    torch.randn(1,  1, 1, 128, dtype=BF16),
                    True,
                ),
            }
        },

        # mul.D5  [1,8,1,128] * [1,1,1,128]  — k * cos decode (broadcast)
        ("test_torch_mul_pattern_004", "_run_mul_binary"): {
            "param_sets": {
                "binary_1x8x1x128_1x1x1x128_decode_eager": (
                    torch.randn(1,  8, 1, 128, dtype=BF16),
                    torch.randn(1,  1, 1, 128, dtype=BF16),
                    False,
                ),
                "binary_1x8x1x128_1x1x1x128_decode_compiled": (
                    torch.randn(1,  8, 1, 128, dtype=BF16),
                    torch.randn(1,  1, 1, 128, dtype=BF16),
                    True,
                ),
            }
        },

        # mul.D6  [1,1,32768] * [1,1,32768]  — gate * up decode (elementwise)
        ("test_torch_mul_pattern_005", "_run_mul_binary"): {
            "param_sets": {
                "binary_1x1x32768_decode_eager": (
                    torch.randn(1, 1, 32768, dtype=BF16),
                    torch.randn(1, 1, 32768, dtype=BF16),
                    False,
                ),
                "binary_1x1x32768_decode_compiled": (
                    torch.randn(1, 1, 32768, dtype=BF16),
                    torch.randn(1, 1, 32768, dtype=BF16),
                    True,
                ),
            }
        },

        # ── PREFILL patterns ──────────────────────────────────────────────────

        # mul.P1  [1,38,128] * scalar  — attention_scaling, RoPE cos prefill
        ("test_torch_mul_pattern_006", "_run_mul_scalar_right"): {
            "param_sets": {
                "scalar_right_1x38x128_prefill_eager": (
                    torch.randn(1, 38, 128, dtype=F32), 1.0, False,
                ),
                "scalar_right_1x38x128_prefill_compiled": (
                    torch.randn(1, 38, 128, dtype=F32), 1.0, True,
                ),
            }
        },

        # mul.P2  [1,38,5120] * [1,38,1]  — rsqrt normalisation prefill
        ("test_torch_mul_pattern_007", "_run_mul_binary"): {
            "param_sets": {
                "binary_1x38x5120_1x38x1_prefill_eager": (
                    torch.randn(1, 38, 5120, dtype=F32),
                    torch.randn(1, 38,    1, dtype=F32),
                    False,
                ),
                "binary_1x38x5120_1x38x1_prefill_compiled": (
                    torch.randn(1, 38, 5120, dtype=F32),
                    torch.randn(1, 38,    1, dtype=F32),
                    True,
                ),
            }
        },

        # mul.P3  [5120] * [1,38,5120]  — weight * hidden prefill (broadcast)
        ("test_torch_mul_pattern_008", "_run_mul_binary"): {
            "param_sets": {
                "binary_5120_1x38x5120_prefill_eager": (
                    torch.randn(5120,         dtype=BF16),
                    torch.randn(1, 38, 5120,  dtype=BF16),
                    False,
                ),
                "binary_5120_1x38x5120_prefill_compiled": (
                    torch.randn(5120,         dtype=BF16),
                    torch.randn(1, 38, 5120,  dtype=BF16),
                    True,
                ),
            }
        },

        # mul.P4  [1,32,38,128] * [1,1,38,128]  — q * cos prefill (broadcast)
        ("test_torch_mul_pattern_009", "_run_mul_binary"): {
            "param_sets": {
                "binary_1x32x38x128_1x1x38x128_prefill_eager": (
                    torch.randn(1, 32, 38, 128, dtype=BF16),
                    torch.randn(1,  1, 38, 128, dtype=BF16),
                    False,
                ),
                "binary_1x32x38x128_1x1x38x128_prefill_compiled": (
                    torch.randn(1, 32, 38, 128, dtype=BF16),
                    torch.randn(1,  1, 38, 128, dtype=BF16),
                    True,
                ),
            }
        },

        # mul.P5  [1,8,38,128] * [1,1,38,128]  — k * cos prefill (broadcast)
        ("test_torch_mul_pattern_010", "_run_mul_binary"): {
            "param_sets": {
                "binary_1x8x38x128_1x1x38x128_prefill_eager": (
                    torch.randn(1,  8, 38, 128, dtype=BF16),
                    torch.randn(1,  1, 38, 128, dtype=BF16),
                    False,
                ),
                "binary_1x8x38x128_1x1x38x128_prefill_compiled": (
                    torch.randn(1,  8, 38, 128, dtype=BF16),
                    torch.randn(1,  1, 38, 128, dtype=BF16),
                    True,
                ),
            }
        },

        # mul.P6  [1,38,32768] * [1,38,32768]  — gate * up prefill (elementwise)
        ("test_torch_mul_pattern_011", "_run_mul_binary"): {
            "param_sets": {
                "binary_1x38x32768_prefill_eager": (
                    torch.randn(1, 38, 32768, dtype=BF16),
                    torch.randn(1, 38, 32768, dtype=BF16),
                    False,
                ),
                "binary_1x38x32768_prefill_compiled": (
                    torch.randn(1, 38, 32768, dtype=BF16),
                    torch.randn(1, 38, 32768, dtype=BF16),
                    True,
                ),
            }
        },
    }

    # ------------------------------------------------------------------
    # Base test methods
    # ------------------------------------------------------------------

    def _run_mul_binary(self, a, b, compiled):
        """torch.mul(tensor, tensor) — both operands are tensors."""
        compare_with_cpu(torch.mul, a, b, compiled=compiled)

    def _run_mul_scalar_right(self, a, scalar, compiled):
        """torch.mul(tensor, scalar) — tensor on left, scalar on right."""
        compare_with_cpu(
            lambda x: torch.mul(x, scalar),
            a,
            compiled=compiled,
        )


# ─────────────────────────────────────────────────────────────────────────────
# TestNeg
# ─────────────────────────────────────────────────────────────────────────────

class TestNeg(unittest.TestCase, metaclass=ParameterizedTestMeta):
    """
    Tests for torch.neg patterns observed in
    mistralai/Mistral-Small-3.2-24B-Instruct-2506.

    neg appears in rotate_half(x) which negates the second half of the last
    dimension and concatenates it with the first half:
        return torch.cat((-x2, x1), dim=-1)   (modeling_mistral.py:55)

    Four unique call-sites:
      neg.1  [1,32,1,64]   — rotate_half q DECODE    bf16
      neg.2  [1,8,1,64]    — rotate_half k DECODE    bf16
      neg.3  [1,32,38,64]  — rotate_half q PREFILL   bf16
      neg.4  [1,8,38,64]   — rotate_half k PREFILL   bf16
    """

    pytestmark = pytest.mark.torch_neg
    torch.manual_seed(0)

    PARAMS = {

        # ── DECODE patterns ───────────────────────────────────────────────────

        # neg.1  [1,32,1,64]  — rotate_half q decode
        ("test_torch_neg_pattern_000", "_run_neg_test"): {
            "param_sets": {
                "neg_1x32x1x64_decode_eager": (
                    torch.randn(1, 32, 1, 64, dtype=BF16), False,
                ),
                "neg_1x32x1x64_decode_compiled": (
                    torch.randn(1, 32, 1, 64, dtype=BF16), True,
                ),
            }
        },

        # neg.2  [1,8,1,64]  — rotate_half k decode
        ("test_torch_neg_pattern_001", "_run_neg_test"): {
            "param_sets": {
                "neg_1x8x1x64_decode_eager": (
                    torch.randn(1, 8, 1, 64, dtype=BF16), False,
                ),
                "neg_1x8x1x64_decode_compiled": (
                    torch.randn(1, 8, 1, 64, dtype=BF16), True,
                ),
            }
        },

        # ── PREFILL patterns ──────────────────────────────────────────────────

        # neg.3  [1,32,38,64]  — rotate_half q prefill
        ("test_torch_neg_pattern_002", "_run_neg_test"): {
            "param_sets": {
                "neg_1x32x38x64_prefill_eager": (
                    torch.randn(1, 32, 38, 64, dtype=BF16), False,
                ),
                "neg_1x32x38x64_prefill_compiled": (
                    torch.randn(1, 32, 38, 64, dtype=BF16), True,
                ),
            }
        },

        # neg.4  [1,8,38,64]  — rotate_half k prefill
        ("test_torch_neg_pattern_003", "_run_neg_test"): {
            "param_sets": {
                "neg_1x8x38x64_prefill_eager": (
                    torch.randn(1, 8, 38, 64, dtype=BF16), False,
                ),
                "neg_1x8x38x64_prefill_compiled": (
                    torch.randn(1, 8, 38, 64, dtype=BF16), True,
                ),
            }
        },
    }

    def _run_neg_test(self, a, compiled):
        """torch.neg(tensor) — elementwise negation."""
        compare_with_cpu(torch.neg, a, compiled=compiled)


# ─────────────────────────────────────────────────────────────────────────────
# TestPow
# ─────────────────────────────────────────────────────────────────────────────

class TestPow(unittest.TestCase, metaclass=ParameterizedTestMeta):
    """
    Tests for torch.pow patterns observed in
    mistralai/Mistral-Small-3.2-24B-Instruct-2506.

    pow appears in the RMSNorm variance computation:
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
                                              (modeling_mistral.py:194)

    Inputs are first cast to float32 before squaring.
    The exponent is always the integer 2.

    Two unique shapes:
      pow.1  [1,1,5120]   float32   DECODE  — variance decode
      pow.2  [1,38,5120]  float32   PREFILL — variance prefill
    """

    pytestmark = pytest.mark.torch_pow
    torch.manual_seed(0)

    PARAMS = {

        # pow.1  [1,1,5120] ** 2  — variance decode
        ("test_torch_pow_pattern_000", "_run_pow_test"): {
            "param_sets": {
                "pow_1x1x5120_exp2_decode_eager": (
                    torch.randn(1, 1, 5120, dtype=F32), 2, False,
                ),
                "pow_1x1x5120_exp2_decode_compiled": (
                    torch.randn(1, 1, 5120, dtype=F32), 2, True,
                ),
            }
        },

        # pow.2  [1,38,5120] ** 2  — variance prefill
        ("test_torch_pow_pattern_001", "_run_pow_test"): {
            "param_sets": {
                "pow_1x38x5120_exp2_prefill_eager": (
                    torch.randn(1, 38, 5120, dtype=F32), 2, False,
                ),
                "pow_1x38x5120_exp2_prefill_compiled": (
                    torch.randn(1, 38, 5120, dtype=F32), 2, True,
                ),
            }
        },

        # ── Extra coverage: float exponent (0.5) ─────────────────────────────
        # Not traced directly in the model but tests the general pow kernel
        # with the same hidden shapes using a non-integer exponent.
        ("test_torch_pow_pattern_002", "_run_pow_test"): {
            "param_sets": {
                "pow_1x1x5120_exp0p5_decode_eager": (
                    torch.abs(torch.randn(1, 1, 5120, dtype=F32)) + 1e-6, 0.5, False,
                ),
                "pow_1x1x5120_exp0p5_decode_compiled": (
                    torch.abs(torch.randn(1, 1, 5120, dtype=F32)) + 1e-6, 0.5, True,
                ),
            }
        },
    }

    def _run_pow_test(self, a, exponent, compiled):
        """torch.pow(tensor, exponent) — elementwise power with scalar exponent."""
        compare_with_cpu(
            lambda x: torch.pow(x, exponent),
            a,
            compiled=compiled,
        )


# ─────────────────────────────────────────────────────────────────────────────
# TestRsqrt
# ─────────────────────────────────────────────────────────────────────────────

class TestRsqrt(unittest.TestCase, metaclass=ParameterizedTestMeta):
    """
    Tests for torch.rsqrt patterns observed in
    mistralai/Mistral-Small-3.2-24B-Instruct-2506.

    rsqrt is called in every RMSNorm block after computing the variance:
        hidden_states * torch.rsqrt(variance + self.variance_epsilon)
                                              (modeling_mistral.py:195)

    The input is variance + epsilon (float32), always strictly positive.
    Inputs are constructed as abs(randn) + eps to guarantee positive values.

    Two unique shapes:
      rsqrt.1  [1,1,1]    float32   DECODE  — variance + eps decode
      rsqrt.2  [1,38,1]   float32   PREFILL — variance + eps prefill
    """

    pytestmark = pytest.mark.torch_rsqrt
    torch.manual_seed(0)

    # Match the model's variance_epsilon (1e-5); use float32 eps as lower bound
    _EPS = torch.finfo(torch.float32).eps  # ~1.19e-7, safely above zero

    PARAMS = {

        # rsqrt.1  [1,1,1]  — variance + eps decode
        ("test_torch_rsqrt_pattern_000", "_run_rsqrt_test"): {
            "param_sets": {
                "rsqrt_1x1x1_decode_eager": (
                    torch.abs(torch.randn(1, 1, 1, dtype=F32)) + _EPS,
                    False,
                ),
                "rsqrt_1x1x1_decode_compiled": (
                    torch.abs(torch.randn(1, 1, 1, dtype=F32)) + _EPS,
                    True,
                ),
            }
        },

        # rsqrt.2  [1,38,1]  — variance + eps prefill
        ("test_torch_rsqrt_pattern_001", "_run_rsqrt_test"): {
            "param_sets": {
                "rsqrt_1x38x1_prefill_eager": (
                    torch.abs(torch.randn(1, 38, 1, dtype=F32)) + _EPS,
                    False,
                ),
                "rsqrt_1x38x1_prefill_compiled": (
                    torch.abs(torch.randn(1, 38, 1, dtype=F32)) + _EPS,
                    True,
                ),
            }
        },

        # ── Extra coverage: batch-size-2 prefill ──────────────────────────────
        # Sanity-check that rsqrt scales correctly to larger batch.
        ("test_torch_rsqrt_pattern_002", "_run_rsqrt_test"): {
            "param_sets": {
                "rsqrt_2x38x1_prefill_batch2_eager": (
                    torch.abs(torch.randn(2, 38, 1, dtype=F32)) + _EPS,
                    False,
                ),
                "rsqrt_2x38x1_prefill_batch2_compiled": (
                    torch.abs(torch.randn(2, 38, 1, dtype=F32)) + _EPS,
                    True,
                ),
            }
        },
    }

    def _run_rsqrt_test(self, a, compiled):
        """torch.rsqrt(tensor) — elementwise reciprocal square root."""
        compare_with_cpu(torch.rsqrt, a, compiled=compiled)


# ─────────────────────────────────────────────────────────────────────────────
# TestUnsqueeze
# ─────────────────────────────────────────────────────────────────────────────

class TestUnsqueeze(unittest.TestCase, metaclass=ParameterizedTestMeta):
    """
    Tests for torch.unsqueeze patterns observed in
    mistralai/Mistral-Small-3.2-24B-Instruct-2506.

    unsqueeze appears twice per attention layer — once for query RoPE,
    once for key RoPE — via:
        cos = cos.unsqueeze(unsqueeze_dim)   (modeling_mistral.py:77)

    The unsqueeze_dim is 1 (insert a head dimension), converting the
    [batch, seq, head_dim] cos/sin tensor into [batch, 1, seq, head_dim]
    for broadcasting against [batch, heads, seq, head_dim] q/k tensors.

    Four unique call-sites:
      unsqueeze.1  [1,1,128]   dim=1   bf16   DECODE  cos/sin decode
      unsqueeze.2  [1,38,128]  dim=1   bf16   PREFILL cos/sin prefill
      (Position IDs unsqueeze at line 370 also traced — integer path)
      unsqueeze.3  [1,]        dim=0   int64  DECODE  position_ids.unsqueeze(0)
      unsqueeze.4  [38,]       dim=0   int64  PREFILL position_ids.unsqueeze(0)
    """

    pytestmark = pytest.mark.torch_unsqueeze
    torch.manual_seed(0)

    PARAMS = {

        # ── DECODE patterns ───────────────────────────────────────────────────

        # unsqueeze.1  [1,1,128] dim=1  — cos/sin decode   bf16
        # Input : (1,1,128) → output : (1,1,1,128)
        ("test_torch_unsqueeze_pattern_000", "_run_unsqueeze_test"): {
            "param_sets": {
                "unsqueeze_1x1x128_dim1_decode_eager": (
                    torch.randn(1, 1, 128, dtype=BF16), 1, False,
                ),
                "unsqueeze_1x1x128_dim1_decode_compiled": (
                    torch.randn(1, 1, 128, dtype=BF16), 1, True,
                ),
            }
        },

        # unsqueeze.3  [1,] dim=0  — position_ids decode   int64
        # Input : (1,) → output : (1,1)
        ("test_torch_unsqueeze_pattern_001", "_run_unsqueeze_test"): {
            "param_sets": {
                "unsqueeze_1_dim0_decode_eager": (
                    torch.randint(0, 2048, (1,), dtype=torch.int64), 0, False,
                ),
                "unsqueeze_1_dim0_decode_compiled": (
                    torch.randint(0, 2048, (1,), dtype=torch.int64), 0, True,
                ),
            }
        },

        # ── PREFILL patterns ──────────────────────────────────────────────────

        # unsqueeze.2  [1,38,128] dim=1  — cos/sin prefill   bf16
        # Input : (1,38,128) → output : (1,1,38,128)
        ("test_torch_unsqueeze_pattern_002", "_run_unsqueeze_test"): {
            "param_sets": {
                "unsqueeze_1x38x128_dim1_prefill_eager": (
                    torch.randn(1, 38, 128, dtype=BF16), 1, False,
                ),
                "unsqueeze_1x38x128_dim1_prefill_compiled": (
                    torch.randn(1, 38, 128, dtype=BF16), 1, True,
                ),
            }
        },

        # unsqueeze.4  [38,] dim=0  — position_ids prefill   int64
        # Input : (38,) → output : (1,38)
        ("test_torch_unsqueeze_pattern_003", "_run_unsqueeze_test"): {
            "param_sets": {
                "unsqueeze_38_dim0_prefill_eager": (
                    torch.randint(0, 2048, (38,), dtype=torch.int64), 0, False,
                ),
                "unsqueeze_38_dim0_prefill_compiled": (
                    torch.randint(0, 2048, (38,), dtype=torch.int64), 0, True,
                ),
            }
        },

        # ── Extra coverage: negative dim ──────────────────────────────────────
        # dim=-1 is commonly used in scaling operations; exercise it on a
        # representative hidden-state slice.
        ("test_torch_unsqueeze_pattern_004", "_run_unsqueeze_test"): {
            "param_sets": {
                "unsqueeze_1x38x5120_dimneg1_eager": (
                    torch.randn(1, 38, 5120, dtype=BF16), -1, False,
                ),
                "unsqueeze_1x38x5120_dimneg1_compiled": (
                    torch.randn(1, 38, 5120, dtype=BF16), -1, True,
                ),
            }
        },
    }

    def _run_unsqueeze_test(self, a, dim, compiled):
        """torch.unsqueeze(tensor, dim) — insert a dimension of size 1 at dim."""
        compare_with_cpu(
            lambda x: torch.unsqueeze(x, dim),
            a,
            compiled=compiled,
        )


# ─────────────────────────────────────────────────────────────────────────────
# TestAdd
# ─────────────────────────────────────────────────────────────────────────────

class TestAdd(unittest.TestCase, metaclass=ParameterizedTestMeta):
    """
    Tests for torch.add patterns observed in Mistral-Small-3.2-24B-Instruct-2506.

    Shapes are sourced directly from Mistral-Small-3.2-24B-Instruct-2506_spyre.yaml.
    Three call signatures appear in the model:

      binary      torch.add(tensor, tensor)
      scalar      torch.add(tensor, scalar)  or  torch.add(scalar, tensor)
      alpha       torch.add(tensor, tensor, alpha=value)
      inplace     tensor.add_(tensor)

    Input shapes, dtypes and semantic roles:
        (38,)            int64     position ids prefill
        (1,)             int64     position ids decode
        (1, 38, 1)       float32   variance epsilon prefill
        (1,  1, 1)       float32   variance epsilon decode
        (1, 32, 38, 128) bfloat16  q_embed rotary prefill
        (1, 32,  1, 128) bfloat16  q_embed rotary decode
        (1,  8, 38, 128) bfloat16  k_embed rotary prefill
        (1,  8,  1, 128) bfloat16  k_embed rotary decode
        (1, 38, 5120)    bfloat16  residual + hidden prefill
        (1,  1, 5120)    bfloat16  residual + hidden decode

    pytestmark stamps every generated method with @pytest.mark.torch_add.
    """

    pytestmark = pytest.mark.torch_add

    torch.manual_seed(0)

    PARAMS = {
        # ------------------------------------------------------------------
        # Binary tensor + tensor patterns
        # ------------------------------------------------------------------

        # add  q_embed rotary prefill: [1,32,38,128] + [1,32,38,128]
        ("test_torch_add_pattern_000", "_run_add_binary"): {
            "param_sets": {
                "binary_1x32x38x128_eager": (
                    torch.randn(1, 32, 38, 128, dtype=torch.bfloat16),
                    torch.randn(1, 32, 38, 128, dtype=torch.bfloat16),
                    False,
                ),
                "binary_1x32x38x128_compiled": (
                    torch.randn(1, 32, 38, 128, dtype=torch.bfloat16),
                    torch.randn(1, 32, 38, 128, dtype=torch.bfloat16),
                    True,
                ),
            }
        },
        # add  k_embed rotary prefill: [1,8,38,128] + [1,8,38,128]
        ("test_torch_add_pattern_001", "_run_add_binary"): {
            "param_sets": {
                "binary_1x8x38x128_eager": (
                    torch.randn(1, 8, 38, 128, dtype=torch.bfloat16),
                    torch.randn(1, 8, 38, 128, dtype=torch.bfloat16),
                    False,
                ),
                "binary_1x8x38x128_compiled": (
                    torch.randn(1, 8, 38, 128, dtype=torch.bfloat16),
                    torch.randn(1, 8, 38, 128, dtype=torch.bfloat16),
                    True,
                ),
            }
        },
        # add  residual + hidden prefill: [1,38,5120] + [1,38,5120]
        ("test_torch_add_pattern_002", "_run_add_binary"): {
            "param_sets": {
                "binary_1x38x5120_eager": (
                    torch.randn(1, 38, 5120, dtype=torch.bfloat16),
                    torch.randn(1, 38, 5120, dtype=torch.bfloat16),
                    False,
                ),
                "binary_1x38x5120_compiled": (
                    torch.randn(1, 38, 5120, dtype=torch.bfloat16),
                    torch.randn(1, 38, 5120, dtype=torch.bfloat16),
                    True,
                ),
            }
        },
        # add  q_embed rotary decode: [1,32,1,128] + [1,32,1,128]
        ("test_torch_add_pattern_003", "_run_add_binary"): {
            "param_sets": {
                "binary_1x32x1x128_eager": (
                    torch.randn(1, 32, 1, 128, dtype=torch.bfloat16),
                    torch.randn(1, 32, 1, 128, dtype=torch.bfloat16),
                    False,
                ),
                "binary_1x32x1x128_compiled": (
                    torch.randn(1, 32, 1, 128, dtype=torch.bfloat16),
                    torch.randn(1, 32, 1, 128, dtype=torch.bfloat16),
                    True,
                ),
            }
        },
        # add  k_embed rotary decode: [1,8,1,128] + [1,8,1,128]
        ("test_torch_add_pattern_004", "_run_add_binary"): {
            "param_sets": {
                "binary_1x8x1x128_eager": (
                    torch.randn(1, 8, 1, 128, dtype=torch.bfloat16),
                    torch.randn(1, 8, 1, 128, dtype=torch.bfloat16),
                    False,
                ),
                "binary_1x8x1x128_compiled": (
                    torch.randn(1, 8, 1, 128, dtype=torch.bfloat16),
                    torch.randn(1, 8, 1, 128, dtype=torch.bfloat16),
                    True,
                ),
            }
        },
        # add  residual + hidden decode: [1,1,5120] + [1,1,5120]
        ("test_torch_add_pattern_005", "_run_add_binary"): {
            "param_sets": {
                "binary_1x1x5120_eager": (
                    torch.randn(1, 1, 5120, dtype=torch.bfloat16),
                    torch.randn(1, 1, 5120, dtype=torch.bfloat16),
                    False,
                ),
                "binary_1x1x5120_compiled": (
                    torch.randn(1, 1, 5120, dtype=torch.bfloat16),
                    torch.randn(1, 1, 5120, dtype=torch.bfloat16),
                    True,
                ),
            }
        },
        # add  position ids binary prefill: [38] + [38]  (int64)
        ("test_torch_add_pattern_006", "_run_add_binary"): {
            "param_sets": {
                "binary_38_int64_eager": (
                    torch.randint(0, 512, (38,), dtype=torch.int64),
                    torch.randint(0, 512, (38,), dtype=torch.int64),
                    False,
                ),
                "binary_38_int64_compiled": (
                    torch.randint(0, 512, (38,), dtype=torch.int64),
                    torch.randint(0, 512, (38,), dtype=torch.int64),
                    True,
                ),
            }
        },
        # add  position ids binary decode: [1] + [1]  (int64)
        ("test_torch_add_pattern_007", "_run_add_binary"): {
            "param_sets": {
                "binary_1_int64_eager": (
                    torch.randint(0, 512, (1,), dtype=torch.int64),
                    torch.randint(0, 512, (1,), dtype=torch.int64),
                    False,
                ),
                "binary_1_int64_compiled": (
                    torch.randint(0, 512, (1,), dtype=torch.int64),
                    torch.randint(0, 512, (1,), dtype=torch.int64),
                    True,
                ),
            }
        },

        # ------------------------------------------------------------------
        # Scalar patterns  (tensor + scalar  or  scalar + tensor)
        # ------------------------------------------------------------------

        # add  variance epsilon prefill: [1,38,1] + 1e-5  (float32)
        ("test_torch_add_pattern_008", "_run_add_scalar"): {
            "param_sets": {
                "scalar_1x38x1_eps_eager": (
                    torch.randn(1, 38, 1, dtype=torch.float32),
                    1e-5,
                    False,
                ),
                "scalar_1x38x1_eps_compiled": (
                    torch.randn(1, 38, 1, dtype=torch.float32),
                    1e-5,
                    True,
                ),
            }
        },
        # add  variance epsilon decode: [1,1,1] + 1e-5  (float32)
        ("test_torch_add_pattern_009", "_run_add_scalar"): {
            "param_sets": {
                "scalar_1x1x1_eps_eager": (
                    torch.randn(1, 1, 1, dtype=torch.float32),
                    1e-5,
                    False,
                ),
                "scalar_1x1x1_eps_compiled": (
                    torch.randn(1, 1, 1, dtype=torch.float32),
                    1e-5,
                    True,
                ),
            }
        },
        # add  position ids scalar offset prefill: [38] + scalar  (int64)
        ("test_torch_add_pattern_010", "_run_add_scalar"): {
            "param_sets": {
                "scalar_38_int64_eager": (
                    torch.randint(0, 512, (38,), dtype=torch.int64),
                    1,
                    False,
                ),
                "scalar_38_int64_compiled": (
                    torch.randint(0, 512, (38,), dtype=torch.int64),
                    1,
                    True,
                ),
            }
        },
        # add  position ids scalar offset decode: [1] + scalar  (int64)
        ("test_torch_add_pattern_011", "_run_add_scalar"): {
            "param_sets": {
                "scalar_1_int64_eager": (
                    torch.randint(0, 512, (1,), dtype=torch.int64),
                    1,
                    False,
                ),
                "scalar_1_int64_compiled": (
                    torch.randint(0, 512, (1,), dtype=torch.int64),
                    1,
                    True,
                ),
            }
        },
        # add  attn scale prefill: 1 + tensor[38]  (scalar on left, bfloat16)
        ("test_torch_add_pattern_012", "_run_add_scalar_left"): {
            "param_sets": {
                "scalar_left_1_plus_38_eager": (
                    1,
                    torch.randn(38, dtype=torch.bfloat16),
                    False,
                ),
                "scalar_left_1_plus_38_compiled": (
                    1,
                    torch.randn(38, dtype=torch.bfloat16),
                    True,
                ),
            }
        },
        # add  attn scale decode: 1 + tensor[1]  (scalar on left, bfloat16)
        ("test_torch_add_pattern_013", "_run_add_scalar_left"): {
            "param_sets": {
                "scalar_left_1_plus_1_eager": (
                    1,
                    torch.randn(1, dtype=torch.bfloat16),
                    False,
                ),
                "scalar_left_1_plus_1_compiled": (
                    1,
                    torch.randn(1, dtype=torch.bfloat16),
                    True,
                ),
            }
        },

        # ------------------------------------------------------------------
        # Alpha  torch.add(a, b, alpha=value)
        # ------------------------------------------------------------------

        # Representative shape from rotary prefill with non-unit alpha (bfloat16)
        ("test_torch_add_pattern_014", "_run_add_alpha"): {
            "param_sets": {
                "alpha_2_1x32x38x128_eager": (
                    torch.randn(1, 32, 38, 128, dtype=torch.bfloat16),
                    torch.randn(1, 32, 38, 128, dtype=torch.bfloat16),
                    2.0,
                    False,
                ),
                "alpha_2_1x32x38x128_compiled": (
                    torch.randn(1, 32, 38, 128, dtype=torch.bfloat16),
                    torch.randn(1, 32, 38, 128, dtype=torch.bfloat16),
                    2.0,
                    True,
                ),
                "alpha_0_1x32x38x128_eager": (
                    torch.randn(1, 32, 38, 128, dtype=torch.bfloat16),
                    torch.randn(1, 32, 38, 128, dtype=torch.bfloat16),
                    0.0,
                    False,
                ),
            }
        },

        # ------------------------------------------------------------------
        # In-place  tensor.add_(tensor)
        # ------------------------------------------------------------------

        # Representative shape from residual update (bfloat16)
        ("test_torch_add_pattern_015", "_run_add_inplace"): {
            "param_sets": {
                "inplace_1x38x5120_eager": (
                    torch.zeros(1, 38, 5120, dtype=torch.bfloat16),
                    torch.randn(1, 38, 5120, dtype=torch.bfloat16),
                    False,
                ),
                "inplace_1x38x5120_compiled": (
                    torch.zeros(1, 38, 5120, dtype=torch.bfloat16),
                    torch.randn(1, 38, 5120, dtype=torch.bfloat16),
                    True,
                ),
                # decode shape
                "inplace_1x1x5120_eager": (
                    torch.zeros(1, 1, 5120, dtype=torch.bfloat16),
                    torch.randn(1, 1, 5120, dtype=torch.bfloat16),
                    False,
                ),
            }
        },
    }

    # ------------------------------------------------------------------
    # Base test methods
    # ------------------------------------------------------------------

    def _run_add_binary(self, a, b, compiled):
        """torch.add(tensor, tensor) — both operands are tensors."""
        compare_with_cpu(torch.add, a, b, compiled=compiled)

    def _run_add_scalar(self, a, scalar, compiled):
        """torch.add(tensor, scalar) — tensor on left, scalar on right."""
        compare_with_cpu(
            lambda x: torch.add(x, scalar),
            a,
            compiled=compiled,
        )

    def _run_add_scalar_left(self, scalar, b, compiled):
        """torch.add(scalar, tensor) — scalar on left, tensor on right."""
        compare_with_cpu(
            lambda x: torch.add(scalar, x),
            b,
            compiled=compiled,
        )

    def _run_add_alpha(self, a, b, alpha, compiled):
        """torch.add(tensor, tensor, alpha=value) — scaled second operand."""
        compare_with_cpu(
            lambda x, y: torch.add(x, y, alpha=alpha),
            a, b,
            compiled=compiled,
        )

    def _run_add_inplace(self, dst, src, compiled):
        """tensor.add_(tensor) — in-place addition; return value must be same object."""
        def fn(d, s):
            d = d.clone()
            out = d.add_(s)
            assert out.data_ptr() == d.data_ptr(), (
                "add_: return value is not the same tensor as dst"
            )
            return out

        compare_with_cpu(fn, dst, src, compiled=compiled)


# ─────────────────────────────────────────────────────────────────────────────
# TestSub
# ─────────────────────────────────────────────────────────────────────────────

class TestSub(unittest.TestCase, metaclass=ParameterizedTestMeta):
    """
    Tests for torch.sub patterns observed in Mistral-Small-3.2-24B-Instruct-2506.

    Shapes are sourced directly from Mistral-Small-3.2-24B-Instruct-2506_spyre.yaml.
    Three call signatures appear in the model:

      binary      torch.sub(tensor, tensor)
      scalar      torch.sub(tensor, scalar)  or  torch.sub(scalar, tensor)
      alpha       torch.sub(tensor, tensor, alpha=value)
      inplace     tensor.sub_(tensor)

    Input shapes, dtypes and semantic roles:
        (1, 1)  int64  position / sequence-length offset (decode)

    pytestmark stamps every generated method with @pytest.mark.torch_sub.
    """

    pytestmark = pytest.mark.torch_sub

    torch.manual_seed(0)

    PARAMS = {
        # ------------------------------------------------------------------
        # Binary tensor - tensor patterns
        # ------------------------------------------------------------------

        # sub  position offset decode: [1,1] - [1,1]  (int64)
        ("test_torch_sub_pattern_000", "_run_sub_binary"): {
            "param_sets": {
                "binary_1x1_int64_eager": (
                    torch.randint(1, 512, (1, 1), dtype=torch.int64),
                    torch.randint(0, 256, (1, 1), dtype=torch.int64),
                    False,
                ),
                "binary_1x1_int64_compiled": (
                    torch.randint(1, 512, (1, 1), dtype=torch.int64),
                    torch.randint(0, 256, (1, 1), dtype=torch.int64),
                    True,
                ),
            }
        },
        # sub  zero result  [1,1] - [1,1]  same values -> 0  (int64)
        ("test_torch_sub_pattern_001", "_run_sub_binary"): {
            "param_sets": {
                "binary_1x1_int64_zero_result_eager": (
                    torch.full((1, 1), 42, dtype=torch.int64),
                    torch.full((1, 1), 42, dtype=torch.int64),
                    False,
                ),
                "binary_1x1_int64_zero_result_compiled": (
                    torch.full((1, 1), 42, dtype=torch.int64),
                    torch.full((1, 1), 42, dtype=torch.int64),
                    True,
                ),
            }
        },

        # ------------------------------------------------------------------
        # Scalar patterns  (tensor - scalar  or  scalar - tensor)
        # ------------------------------------------------------------------

        # sub  position ids scalar offset decode: [1,1] - 1  (int64)
        ("test_torch_sub_pattern_002", "_run_sub_scalar"): {
            "param_sets": {
                "scalar_1x1_int64_minus1_eager": (
                    torch.randint(1, 512, (1, 1), dtype=torch.int64),
                    1,
                    False,
                ),
                "scalar_1x1_int64_minus1_compiled": (
                    torch.randint(1, 512, (1, 1), dtype=torch.int64),
                    1,
                    True,
                ),
            }
        },
        # sub  position ids scalar offset decode: [1,1] - 0  (int64, no-op)
        ("test_torch_sub_pattern_003", "_run_sub_scalar"): {
            "param_sets": {
                "scalar_1x1_int64_minus0_eager": (
                    torch.randint(0, 512, (1, 1), dtype=torch.int64),
                    0,
                    False,
                ),
                "scalar_1x1_int64_minus0_compiled": (
                    torch.randint(0, 512, (1, 1), dtype=torch.int64),
                    0,
                    True,
                ),
            }
        },
        # sub  scalar - tensor decode: scalar - [1,1]  (int64, scalar on left)
        ("test_torch_sub_pattern_004", "_run_sub_scalar_left"): {
            "param_sets": {
                "scalar_left_512_minus_1x1_eager": (
                    512,
                    torch.randint(0, 512, (1, 1), dtype=torch.int64),
                    False,
                ),
                "scalar_left_512_minus_1x1_compiled": (
                    512,
                    torch.randint(0, 512, (1, 1), dtype=torch.int64),
                    True,
                ),
            }
        },

        # ------------------------------------------------------------------
        # Alpha  torch.sub(a, b, alpha=value)
        # ------------------------------------------------------------------

        # sub  alpha=2 decode: [1,1] - 2*[1,1]  (int64)
        ("test_torch_sub_pattern_005", "_run_sub_alpha"): {
            "param_sets": {
                "alpha_2_1x1_int64_eager": (
                    torch.randint(1, 512, (1, 1), dtype=torch.int64),
                    torch.randint(0, 64,  (1, 1), dtype=torch.int64),
                    2,
                    False,
                ),
                "alpha_2_1x1_int64_compiled": (
                    torch.randint(1, 512, (1, 1), dtype=torch.int64),
                    torch.randint(0, 64,  (1, 1), dtype=torch.int64),
                    2,
                    True,
                ),
                "alpha_0_1x1_int64_eager": (
                    torch.randint(1, 512, (1, 1), dtype=torch.int64),
                    torch.randint(0, 64,  (1, 1), dtype=torch.int64),
                    0,
                    False,
                ),
            }
        },

        # ------------------------------------------------------------------
        # In-place  tensor.sub_(tensor)
        # ------------------------------------------------------------------

        # sub_  in-place position offset decode: [1,1] -= [1,1]  (int64)
        ("test_torch_sub_pattern_006", "_run_sub_inplace"): {
            "param_sets": {
                "inplace_1x1_int64_eager": (
                    torch.randint(1, 512, (1, 1), dtype=torch.int64),
                    torch.randint(0, 256, (1, 1), dtype=torch.int64),
                    False,
                ),
                "inplace_1x1_int64_compiled": (
                    torch.randint(1, 512, (1, 1), dtype=torch.int64),
                    torch.randint(0, 256, (1, 1), dtype=torch.int64),
                    True,
                ),
            }
        },
    }

    # ------------------------------------------------------------------
    # Base test methods
    # ------------------------------------------------------------------

    def _run_sub_binary(self, a, b, compiled):
        """torch.sub(tensor, tensor) — both operands are tensors."""
        compare_with_cpu(torch.sub, a, b, compiled=compiled)

    def _run_sub_scalar(self, a, scalar, compiled):
        """torch.sub(tensor, scalar) — tensor on left, scalar on right."""
        compare_with_cpu(
            lambda x: torch.sub(x, scalar),
            a,
            compiled=compiled,
        )

    def _run_sub_scalar_left(self, scalar, b, compiled):
        """torch.sub(scalar, tensor) — scalar on left, tensor on right."""
        compare_with_cpu(
            lambda x: torch.sub(scalar, x),
            b,
            compiled=compiled,
        )

    def _run_sub_alpha(self, a, b, alpha, compiled):
        """torch.sub(tensor, tensor, alpha=value) — scaled subtrahend."""
        compare_with_cpu(
            lambda x, y: torch.sub(x, y, alpha=alpha),
            a, b,
            compiled=compiled,
        )

    def _run_sub_inplace(self, dst, src, compiled):
        """tensor.sub_(tensor) — in-place subtraction; return value must be same object."""
        def fn(d, s):
            d = d.clone()
            out = d.sub_(s)
            assert out.data_ptr() == d.data_ptr(), (
                "sub_: return value is not the same tensor as dst"
            )
            return out

        compare_with_cpu(fn, dst, src, compiled=compiled)


# ─────────────────────────────────────────────────────────────────────────────
# TestMatmul
# ─────────────────────────────────────────────────────────────────────────────

class TestMatmul(unittest.TestCase, metaclass=ParameterizedTestMeta):
    """
    Tests for torch.matmul patterns observed in Mistral-Small-3.2-24B-Instruct-2506.

    The PARAMS dict drives ParameterizedTestMeta to expand ``_run_matmul_test``
    into one concrete test method per (pattern, eager/compiled) combination.
    - pytestmark = torch_matmul selects the entire class with  pytest -m torch_matmul.
    - Each generated method is also individually stamped with @pytest.mark.torch_matmul
      by the metaclass (derived from _run_matmul_test -> "matmul" -> torch_matmul).

    Input shapes and output shapes:
        A: (1, 64,  1)  B: (1, 1, 38) -> out: (1, 64, 38)   attention score prefill
        A: (1, 64,  1)  B: (1, 1,  1) -> out: (1, 64,  1)   attention score decode
    dtype : torch.float32

    Each param_set entry is a 4-tuple:
        (a: Tensor, b: Tensor, op: callable, compiled: bool)
    Exception: _run_matmul_special_values_test takes only (a, b) — CPU-only.
    """

    pytestmark = pytest.mark.torch_matmul

    torch.manual_seed(0)

    PARAMS = {
        # ------------------------------------------------------------------
        # pattern_000  basic matmul  A:(1,64,1) @ B:(1,1,38) -> (1,64,38)
        # ------------------------------------------------------------------
        ("test_torch_matmul_pattern_000", "_run_matmul_test"): {
            "param_sets": {
                "1x64x1_1x1x38_eager":    (torch.randn(1, 64, 1, dtype=torch.float32), torch.randn(1, 1, 38, dtype=torch.float32), lambda a, b: torch.matmul(a, b), False),
                "1x64x1_1x1x38_compiled": (torch.randn(1, 64, 1, dtype=torch.float32), torch.randn(1, 1, 38, dtype=torch.float32), lambda a, b: torch.matmul(a, b), True),
            }
        },
        # ------------------------------------------------------------------
        # pattern_001  basic matmul  A:(1,64,1) @ B:(1,1,1) -> (1,64,1)
        # ------------------------------------------------------------------
        ("test_torch_matmul_pattern_001", "_run_matmul_test"): {
            "param_sets": {
                "1x64x1_1x1x1_eager":    (torch.randn(1, 64, 1, dtype=torch.float32), torch.randn(1, 1,  1, dtype=torch.float32), lambda a, b: torch.matmul(a, b), False),
                "1x64x1_1x1x1_compiled": (torch.randn(1, 64, 1, dtype=torch.float32), torch.randn(1, 1,  1, dtype=torch.float32), lambda a, b: torch.matmul(a, b), True),
            }
        },
        # ------------------------------------------------------------------
        # pattern_002  method alias  A:(1,64,1) @ B:(1,1,38) -> (1,64,38)
        # a.matmul(b) == torch.matmul(a, b)
        # ------------------------------------------------------------------
        ("test_torch_matmul_pattern_002", "_run_matmul_test"): {
            "param_sets": {
                "method_1x64x1_1x1x38_eager":    (torch.randn(1, 64, 1, dtype=torch.float32), torch.randn(1, 1, 38, dtype=torch.float32), lambda a, b: a.matmul(b), False),
                "method_1x64x1_1x1x38_compiled": (torch.randn(1, 64, 1, dtype=torch.float32), torch.randn(1, 1, 38, dtype=torch.float32), lambda a, b: a.matmul(b), True),
            }
        },
        # ------------------------------------------------------------------
        # pattern_003  method alias  A:(1,64,1) @ B:(1,1,1) -> (1,64,1)
        # ------------------------------------------------------------------
        ("test_torch_matmul_pattern_003", "_run_matmul_test"): {
            "param_sets": {
                "method_1x64x1_1x1x1_eager":    (torch.randn(1, 64, 1, dtype=torch.float32), torch.randn(1, 1,  1, dtype=torch.float32), lambda a, b: a.matmul(b), False),
                "method_1x64x1_1x1x1_compiled": (torch.randn(1, 64, 1, dtype=torch.float32), torch.randn(1, 1,  1, dtype=torch.float32), lambda a, b: a.matmul(b), True),
            }
        },
        # ------------------------------------------------------------------
        # pattern_004  @ operator  A:(1,64,1) @ B:(1,1,38) -> (1,64,38)
        # a @ b == torch.matmul(a, b)
        # ------------------------------------------------------------------
        ("test_torch_matmul_pattern_004", "_run_matmul_test"): {
            "param_sets": {
                "bmm_op_1x64x1_1x1x38_eager":    (torch.randn(1, 64, 1, dtype=torch.float32), torch.randn(1, 1, 38, dtype=torch.float32), lambda a, b: a @ b, False),
                "bmm_op_1x64x1_1x1x38_compiled": (torch.randn(1, 64, 1, dtype=torch.float32), torch.randn(1, 1, 38, dtype=torch.float32), lambda a, b: a @ b, True),
            }
        },
        # ------------------------------------------------------------------
        # pattern_005  @ operator  A:(1,64,1) @ B:(1,1,1) -> (1,64,1)
        # ------------------------------------------------------------------
        ("test_torch_matmul_pattern_005", "_run_matmul_test"): {
            "param_sets": {
                "bmm_op_1x64x1_1x1x1_eager":    (torch.randn(1, 64, 1, dtype=torch.float32), torch.randn(1, 1,  1, dtype=torch.float32), lambda a, b: a @ b, False),
                "bmm_op_1x64x1_1x1x1_compiled": (torch.randn(1, 64, 1, dtype=torch.float32), torch.randn(1, 1,  1, dtype=torch.float32), lambda a, b: a @ b, True),
            }
        },
        # ------------------------------------------------------------------
        # pattern_006  all-zeros A  A:(1,64,1) @ B:(1,1,38) -> all-zeros out
        # matmul with zero matrix must produce zero output.
        # ------------------------------------------------------------------
        ("test_torch_matmul_pattern_006", "_run_matmul_test"): {
            "param_sets": {
                "zeros_a_1x64x1_1x1x38_eager":    (torch.zeros(1, 64, 1, dtype=torch.float32), torch.randn(1, 1, 38, dtype=torch.float32), lambda a, b: torch.matmul(a, b), False),
                "zeros_a_1x64x1_1x1x38_compiled": (torch.zeros(1, 64, 1, dtype=torch.float32), torch.randn(1, 1, 38, dtype=torch.float32), lambda a, b: torch.matmul(a, b), True),
            }
        },
        # ------------------------------------------------------------------
        # pattern_007  all-zeros B  A:(1,64,1) @ B:(1,1,1) -> all-zeros out
        # ------------------------------------------------------------------
        ("test_torch_matmul_pattern_007", "_run_matmul_test"): {
            "param_sets": {
                "zeros_b_1x64x1_1x1x1_eager":    (torch.randn(1, 64, 1, dtype=torch.float32), torch.zeros(1, 1,  1, dtype=torch.float32), lambda a, b: torch.matmul(a, b), False),
                "zeros_b_1x64x1_1x1x1_compiled": (torch.randn(1, 64, 1, dtype=torch.float32), torch.zeros(1, 1,  1, dtype=torch.float32), lambda a, b: torch.matmul(a, b), True),
            }
        },
        # ------------------------------------------------------------------
        # pattern_008  all-ones inputs  A:(1,64,1) @ B:(1,1,38) -> all-ones * 1
        # matmul of ones: each output element == 1.0 (inner dim = 1).
        # ------------------------------------------------------------------
        ("test_torch_matmul_pattern_008", "_run_matmul_test"): {
            "param_sets": {
                "ones_1x64x1_1x1x38_eager":    (torch.ones(1, 64, 1, dtype=torch.float32), torch.ones(1, 1, 38, dtype=torch.float32), lambda a, b: torch.matmul(a, b), False),
                "ones_1x64x1_1x1x38_compiled": (torch.ones(1, 64, 1, dtype=torch.float32), torch.ones(1, 1, 38, dtype=torch.float32), lambda a, b: torch.matmul(a, b), True),
            }
        },
        # ------------------------------------------------------------------
        # pattern_009  all-ones inputs  A:(1,64,1) @ B:(1,1,1) -> 1.0
        # ------------------------------------------------------------------
        ("test_torch_matmul_pattern_009", "_run_matmul_test"): {
            "param_sets": {
                "ones_1x64x1_1x1x1_eager":    (torch.ones(1, 64, 1, dtype=torch.float32), torch.ones(1, 1,  1, dtype=torch.float32), lambda a, b: torch.matmul(a, b), False),
                "ones_1x64x1_1x1x1_compiled": (torch.ones(1, 64, 1, dtype=torch.float32), torch.ones(1, 1,  1, dtype=torch.float32), lambda a, b: torch.matmul(a, b), True),
            }
        },
        # ------------------------------------------------------------------
        # pattern_010  matmul -> add  A:(1,64,1) @ B:(1,1,38) + bias
        # Simulates linear projection with bias addition (prefill).
        # ------------------------------------------------------------------
        ("test_torch_matmul_pattern_010", "_run_matmul_test"): {
            "param_sets": {
                "add_bias_1x64x1_1x1x38_eager":    (torch.randn(1, 64, 1, dtype=torch.float32), torch.randn(1, 1, 38, dtype=torch.float32), lambda a, b: torch.matmul(a, b) + torch.ones(1, 64, 38, dtype=torch.float32), False),
                "add_bias_1x64x1_1x1x38_compiled": (torch.randn(1, 64, 1, dtype=torch.float32), torch.randn(1, 1, 38, dtype=torch.float32), lambda a, b: torch.matmul(a, b) + torch.ones(1, 64, 38, dtype=torch.float32), True),
            }
        },
        # ------------------------------------------------------------------
        # pattern_011  matmul -> add  A:(1,64,1) @ B:(1,1,1) + bias
        # Simulates linear projection with bias addition (decode).
        # ------------------------------------------------------------------
        ("test_torch_matmul_pattern_011", "_run_matmul_test"): {
            "param_sets": {
                "add_bias_1x64x1_1x1x1_eager":    (torch.randn(1, 64, 1, dtype=torch.float32), torch.randn(1, 1,  1, dtype=torch.float32), lambda a, b: torch.matmul(a, b) + torch.ones(1, 64,  1, dtype=torch.float32), False),
                "add_bias_1x64x1_1x1x1_compiled": (torch.randn(1, 64, 1, dtype=torch.float32), torch.randn(1, 1,  1, dtype=torch.float32), lambda a, b: torch.matmul(a, b) + torch.ones(1, 64,  1, dtype=torch.float32), True),
            }
        },
        # ------------------------------------------------------------------
        # pattern_012  special values  A:(1,64,1) @ B:(1,1,38)
        # CPU-only: +inf in A propagates to output.
        # ------------------------------------------------------------------
        ("test_torch_matmul_pattern_012", "_run_matmul_special_values_test"): {
            "param_sets": {
                "special_inf_a_1x64x1_1x1x38": (
                    torch.full((1, 64,  1), float("inf"), dtype=torch.float32),
                    torch.ones( (1,  1, 38),               dtype=torch.float32),
                ),
            }
        },
        # ------------------------------------------------------------------
        # pattern_013  special values  A:(1,64,1) @ B:(1,1,1)
        # CPU-only: NaN in A must propagate to all output elements.
        # ------------------------------------------------------------------
        ("test_torch_matmul_pattern_013", "_run_matmul_special_values_test"): {
            "param_sets": {
                "special_nan_a_1x64x1_1x1x1": (
                    torch.full((1, 64, 1), float("nan"), dtype=torch.float32),
                    torch.ones( (1,  1, 1),               dtype=torch.float32),
                ),
            }
        },
    }

    # ------------------------------------------------------------------
    # Base test methods — expanded by ParameterizedTestMeta.
    # ------------------------------------------------------------------

    def _run_matmul_test(self, a, b, op, compiled):
        """
        Shared body for all torch.matmul pattern tests.

        Wraps the given op in compare_with_cpu so that CPU and Spyre
        outputs are compared for both eager and compiled paths.

        Args:
            a:        float32 tensor, first matmul operand.
            b:        float32 tensor, second matmul operand.
            op:       callable applied to (a, b), e.g.:
                        lambda a, b: torch.matmul(a, b)
                        lambda a, b: a.matmul(b)
                        lambda a, b: a @ b
                        lambda a, b: torch.matmul(a, b) + bias
            compiled: True  -> torch.compile path
                      False -> eager path
        """
        compare_with_cpu(op, a, b, compiled=compiled)

    def _run_matmul_special_values_test(self, a, b):
        """
        CPU-only structural check for matmul with special IEEE 754 values.

        Verifies that +inf in an input propagates to +inf in the output,
        and NaN in an input poisons all output elements to NaN.

        Args:
            a: float32 tensor, first matmul operand (may contain inf/NaN).
            b: float32 tensor, second matmul operand.
        """
        result = torch.matmul(a.cpu(), b.cpu())

        if torch.isinf(a).any() or torch.isinf(b).any():
            assert torch.isinf(result).any() or torch.isnan(result).any(), (
                f"Expected inf/nan in output when input contains inf, "
                f"got: {result}"
            )
        if torch.isnan(a).any() or torch.isnan(b).any():
            assert torch.isnan(result).any(), (
                f"Expected NaN in output when input contains NaN, "
                f"got: {result}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# TestMean
# ─────────────────────────────────────────────────────────────────────────────

class TestMean(unittest.TestCase, metaclass=ParameterizedTestMeta):
    """
    Tests for torch.mean patterns observed in Mistral-Small-3.2-24B-Instruct-2506.

    The PARAMS dict drives ParameterizedTestMeta to expand ``_run_mean_test``
    into one concrete test method per (pattern, eager/compiled) combination.
    - pytestmark = torch_mean selects the entire class with  pytest -m torch_mean.
    - Each generated method is also individually stamped with @pytest.mark.torch_mean
      by the metaclass (derived from _run_mean_test -> "mean" -> torch_mean).

    Input shapes and reduction outputs:
        (1,  1, 5120)  dim=-1 -> (1,  1,    1)   hidden-state mean (decode)
        (1, 38, 5120)  dim=-1 -> (1, 38,    1)   hidden-state mean (prefill)
        (1,  1, 5120)  dim=0  -> (1,  1, 5120)   batch mean (decode)
        (1, 38, 5120)  dim=0  -> (1, 38, 5120)   batch mean (prefill)
    dtype : torch.float32

    Each param_set entry is a 3-tuple:
        (tensor: Tensor, op: callable, compiled: bool)
    Exception: _run_mean_special_values_test takes only (tensor,) — CPU-only.
    """

    pytestmark = pytest.mark.torch_mean

    torch.manual_seed(0)

    PARAMS = {
        # ------------------------------------------------------------------
        # pattern_000  basic mean  (1,1,5120)  no dim — global mean
        # ------------------------------------------------------------------
        ("test_torch_mean_pattern_000", "_run_mean_test"): {
            "param_sets": {
                "1x1x5120_eager":    (torch.randn(1,  1, 5120, dtype=torch.float32), lambda t: torch.mean(t), False),
                "1x1x5120_compiled": (torch.randn(1,  1, 5120, dtype=torch.float32), lambda t: torch.mean(t), True),
            }
        },
        # ------------------------------------------------------------------
        # pattern_001  basic mean  (1,38,5120)  no dim — global mean
        # ------------------------------------------------------------------
        ("test_torch_mean_pattern_001", "_run_mean_test"): {
            "param_sets": {
                "1x38x5120_eager":    (torch.randn(1, 38, 5120, dtype=torch.float32), lambda t: torch.mean(t), False),
                "1x38x5120_compiled": (torch.randn(1, 38, 5120, dtype=torch.float32), lambda t: torch.mean(t), True),
            }
        },
        # ------------------------------------------------------------------
        # pattern_002  mean over last dim  (1,1,5120)  dim=-1 -> (1,1,1)
        # Reduces hidden dimension — most common path in layer norm pre-step.
        # ------------------------------------------------------------------
        ("test_torch_mean_pattern_002", "_run_mean_test"): {
            "param_sets": {
                "1x1x5120_dim-1_eager":    (torch.randn(1,  1, 5120, dtype=torch.float32), lambda t: torch.mean(t, dim=-1), False),
                "1x1x5120_dim-1_compiled": (torch.randn(1,  1, 5120, dtype=torch.float32), lambda t: torch.mean(t, dim=-1), True),
            }
        },
        # ------------------------------------------------------------------
        # pattern_003  mean over last dim  (1,38,5120)  dim=-1 -> (1,38,1)
        # ------------------------------------------------------------------
        ("test_torch_mean_pattern_003", "_run_mean_test"): {
            "param_sets": {
                "1x38x5120_dim-1_eager":    (torch.randn(1, 38, 5120, dtype=torch.float32), lambda t: torch.mean(t, dim=-1), False),
                "1x38x5120_dim-1_compiled": (torch.randn(1, 38, 5120, dtype=torch.float32), lambda t: torch.mean(t, dim=-1), True),
            }
        },
        # ------------------------------------------------------------------
        # pattern_004  mean over last dim keepdim  (1,1,5120)  -> (1,1,1)
        # keepdim=True preserves shape for downstream broadcast.
        # ------------------------------------------------------------------
        ("test_torch_mean_pattern_004", "_run_mean_test"): {
            "param_sets": {
                "1x1x5120_keepdim_eager":    (torch.randn(1,  1, 5120, dtype=torch.float32), lambda t: torch.mean(t, dim=-1, keepdim=True), False),
                "1x1x5120_keepdim_compiled": (torch.randn(1,  1, 5120, dtype=torch.float32), lambda t: torch.mean(t, dim=-1, keepdim=True), True),
            }
        },
        # ------------------------------------------------------------------
        # pattern_005  mean over last dim keepdim  (1,38,5120)  -> (1,38,1)
        # ------------------------------------------------------------------
        ("test_torch_mean_pattern_005", "_run_mean_test"): {
            "param_sets": {
                "1x38x5120_keepdim_eager":    (torch.randn(1, 38, 5120, dtype=torch.float32), lambda t: torch.mean(t, dim=-1, keepdim=True), False),
                "1x38x5120_keepdim_compiled": (torch.randn(1, 38, 5120, dtype=torch.float32), lambda t: torch.mean(t, dim=-1, keepdim=True), True),
            }
        },
        # ------------------------------------------------------------------
        # pattern_006  mean over seq dim  (1,38,5120)  dim=1 -> (1,1,5120)
        # Reduces sequence dimension — pooling across tokens.
        # ------------------------------------------------------------------
        ("test_torch_mean_pattern_006", "_run_mean_test"): {
            "param_sets": {
                "1x38x5120_dim1_eager":    (torch.randn(1, 38, 5120, dtype=torch.float32), lambda t: torch.mean(t, dim=1), False),
                "1x38x5120_dim1_compiled": (torch.randn(1, 38, 5120, dtype=torch.float32), lambda t: torch.mean(t, dim=1), True),
            }
        },
        # ------------------------------------------------------------------
        # pattern_007  mean over batch dim  (1,1,5120)  dim=0 -> (1,5120)
        # ------------------------------------------------------------------
        ("test_torch_mean_pattern_007", "_run_mean_test"): {
            "param_sets": {
                "1x1x5120_dim0_eager":    (torch.randn(1,  1, 5120, dtype=torch.float32), lambda t: torch.mean(t, dim=0), False),
                "1x1x5120_dim0_compiled": (torch.randn(1,  1, 5120, dtype=torch.float32), lambda t: torch.mean(t, dim=0), True),
            }
        },
        # ------------------------------------------------------------------
        # pattern_008  method alias  (1,1,5120)  t.mean(dim=-1)
        # t.mean() == torch.mean(t) — both call sites appear in the model.
        # ------------------------------------------------------------------
        ("test_torch_mean_pattern_008", "_run_mean_test"): {
            "param_sets": {
                "method_1x1x5120_eager":    (torch.randn(1,  1, 5120, dtype=torch.float32), lambda t: t.mean(dim=-1), False),
                "method_1x1x5120_compiled": (torch.randn(1,  1, 5120, dtype=torch.float32), lambda t: t.mean(dim=-1), True),
            }
        },
        # ------------------------------------------------------------------
        # pattern_009  method alias  (1,38,5120)  t.mean(dim=-1)
        # ------------------------------------------------------------------
        ("test_torch_mean_pattern_009", "_run_mean_test"): {
            "param_sets": {
                "method_1x38x5120_eager":    (torch.randn(1, 38, 5120, dtype=torch.float32), lambda t: t.mean(dim=-1), False),
                "method_1x38x5120_compiled": (torch.randn(1, 38, 5120, dtype=torch.float32), lambda t: t.mean(dim=-1), True),
            }
        },
        # ------------------------------------------------------------------
        # pattern_010  all-zeros input  (1,1,5120)  mean must be 0.0
        # ------------------------------------------------------------------
        ("test_torch_mean_pattern_010", "_run_mean_test"): {
            "param_sets": {
                "zeros_1x1x5120_eager":    (torch.zeros(1,  1, 5120, dtype=torch.float32), lambda t: torch.mean(t, dim=-1), False),
                "zeros_1x1x5120_compiled": (torch.zeros(1,  1, 5120, dtype=torch.float32), lambda t: torch.mean(t, dim=-1), True),
            }
        },
        # ------------------------------------------------------------------
        # pattern_011  all-zeros input  (1,38,5120)  mean must be 0.0
        # ------------------------------------------------------------------
        ("test_torch_mean_pattern_011", "_run_mean_test"): {
            "param_sets": {
                "zeros_1x38x5120_eager":    (torch.zeros(1, 38, 5120, dtype=torch.float32), lambda t: torch.mean(t, dim=-1), False),
                "zeros_1x38x5120_compiled": (torch.zeros(1, 38, 5120, dtype=torch.float32), lambda t: torch.mean(t, dim=-1), True),
            }
        },
        # ------------------------------------------------------------------
        # pattern_012  all-ones input  (1,1,5120)  mean must be 1.0
        # ------------------------------------------------------------------
        ("test_torch_mean_pattern_012", "_run_mean_test"): {
            "param_sets": {
                "ones_1x1x5120_eager":    (torch.ones(1,  1, 5120, dtype=torch.float32), lambda t: torch.mean(t, dim=-1), False),
                "ones_1x1x5120_compiled": (torch.ones(1,  1, 5120, dtype=torch.float32), lambda t: torch.mean(t, dim=-1), True),
            }
        },
        # ------------------------------------------------------------------
        # pattern_013  all-ones input  (1,38,5120)  mean must be 1.0
        # ------------------------------------------------------------------
        ("test_torch_mean_pattern_013", "_run_mean_test"): {
            "param_sets": {
                "ones_1x38x5120_eager":    (torch.ones(1, 38, 5120, dtype=torch.float32), lambda t: torch.mean(t, dim=-1), False),
                "ones_1x38x5120_compiled": (torch.ones(1, 38, 5120, dtype=torch.float32), lambda t: torch.mean(t, dim=-1), True),
            }
        },
        # ------------------------------------------------------------------
        # pattern_014  mean -> cast to float16  (1,1,5120)
        # Simulates precision downcast after mean for mixed-precision computation.
        # ------------------------------------------------------------------
        ("test_torch_mean_pattern_014", "_run_mean_test"): {
            "param_sets": {
                "cast_fp16_1x1x5120_eager":    (torch.randn(1,  1, 5120, dtype=torch.float32), lambda t: torch.mean(t, dim=-1).to(torch.float16), False),
                "cast_fp16_1x1x5120_compiled": (torch.randn(1,  1, 5120, dtype=torch.float32), lambda t: torch.mean(t, dim=-1).to(torch.float16), True),
            }
        },
        # ------------------------------------------------------------------
        # pattern_015  mean -> cast to float16  (1,38,5120)
        # ------------------------------------------------------------------
        ("test_torch_mean_pattern_015", "_run_mean_test"): {
            "param_sets": {
                "cast_fp16_1x38x5120_eager":    (torch.randn(1, 38, 5120, dtype=torch.float32), lambda t: torch.mean(t, dim=-1).to(torch.float16), False),
                "cast_fp16_1x38x5120_compiled": (torch.randn(1, 38, 5120, dtype=torch.float32), lambda t: torch.mean(t, dim=-1).to(torch.float16), True),
            }
        },
        # ------------------------------------------------------------------
        # pattern_016  special values  (1,1,5120)  NaN in input poisons mean
        # CPU-only structural check — no compiled variant needed.
        # ------------------------------------------------------------------
        ("test_torch_mean_pattern_016", "_run_mean_special_values_test"): {
            "param_sets": {
                "special_nan_1x1x5120": (
                    torch.cat([
                        torch.tensor([float("nan")], dtype=torch.float32),
                        torch.ones(5119, dtype=torch.float32),
                    ]).reshape(1, 1, 5120),
                ),
            }
        },
        # ------------------------------------------------------------------
        # pattern_017  special values  (1,38,5120)  +inf in input -> +inf mean
        # CPU-only structural check — no compiled variant needed.
        # ------------------------------------------------------------------
        ("test_torch_mean_pattern_017", "_run_mean_special_values_test"): {
            "param_sets": {
                "special_inf_1x38x5120": (
                    torch.cat([
                        torch.tensor([float("inf")], dtype=torch.float32),
                        torch.ones(38 * 5120 - 1, dtype=torch.float32),
                    ]).reshape(1, 38, 5120),
                ),
            }
        },
    }

    # ------------------------------------------------------------------
    # Base test methods — expanded by ParameterizedTestMeta.
    # ------------------------------------------------------------------

    def _run_mean_test(self, tensor, op, compiled):
        """
        Shared body for all torch.mean pattern tests.

        Wraps the given op in compare_with_cpu so that CPU and Spyre
        outputs are compared for both eager and compiled paths.

        Args:
            tensor:   float32 input tensor of shape (1,1,5120) or (1,38,5120).
            op:       callable applied to the tensor, e.g.:
                        lambda t: torch.mean(t)
                        lambda t: torch.mean(t, dim=-1)
                        lambda t: torch.mean(t, dim=-1, keepdim=True)
                        lambda t: t.mean(dim=-1)
                        lambda t: torch.mean(t, dim=-1).to(torch.float16)
            compiled: True  -> torch.compile path
                      False -> eager path
        """
        compare_with_cpu(op, tensor, compiled=compiled)

    def _run_mean_special_values_test(self, tensor):
        """
        CPU-only structural check for mean with special IEEE 754 values.

        Verifies that NaN in any element poisons the mean to NaN, and that
        +inf in any element produces +inf or NaN in the mean output.

        Args:
            tensor: float32 tensor of shape (1,1,5120) or (1,38,5120)
                    containing at least one special value.
        """
        result = torch.mean(tensor.cpu(), dim=-1)

        if torch.isnan(tensor).any():
            assert torch.isnan(result).any(), (
                f"Expected NaN in mean output when input contains NaN, "
                f"got: {result}"
            )
        if torch.isinf(tensor).any():
            assert torch.isinf(result).any() or torch.isnan(result).any(), (
                f"Expected inf/nan in mean output when input contains inf, "
                f"got: {result}"
            )

# ─────────────────────────────────────────────────────────────────────────────
# TestNe
# ─────────────────────────────────────────────────────────────────────────────

class TestNe(unittest.TestCase, metaclass=ParameterizedTestMeta):
    """
    Tests for torch.ne patterns observed in
    mistralai/Mistral-Small-3.2-24B-Instruct-2506.

    torch.ne appears exclusively in the PREFILL scenario as part of the
    causal-mask construction pipeline (modeling_mistral.py:373):

        position_ids_delta = torch.diff(position_ids, prepend=first_ids)
        non_contiguous_mask = position_ids_delta.ne(1)
        ...
        has_non_contiguous = non_contiguous_mask.cumsum(dim=-1)

    The ne call tests each element for inequality with the scalar 1, producing
    a bool tensor of the same shape as the input.

    One unique call-site:
      ne.1  [1,38]  int64  != 1  → [1,38]  bool   PREFILL
            source: modeling_mistral.py:373
            input[0] : shape=(1, 38)  dtype=int64
            output  : shape=(1, 38)  dtype=bool

    Note: this op does NOT appear in the DECODE trace (seq_len=1 path skips
    the non-contiguous position-id check).
    """

    pytestmark = pytest.mark.torch_ne
    torch.manual_seed(0)

    PARAMS = {

        # ── ne.1  [1,38] int64 != 1  — non-contiguous position-id mask ────────
        # The typical case: position_ids are contiguous [0..37], so the delta
        # is all 1s and ne(1) returns all False.
        ("test_torch_ne_pattern_000", "_run_ne_scalar_test"): {
            "param_sets": {
                "ne_1x38_int64_contiguous_eager": (
                    torch.arange(38, dtype=torch.int64).unsqueeze(0),  # [1,38]
                    1, False,
                ),
                "ne_1x38_int64_contiguous_compiled": (
                    torch.arange(38, dtype=torch.int64).unsqueeze(0),
                    1, True,
                ),
            }
        },

        # Non-contiguous position IDs (e.g. padding / chunked prefill):
        # some deltas will be != 1, so the result has True entries.
        ("test_torch_ne_pattern_001", "_run_ne_scalar_test"): {
            "param_sets": {
                "ne_1x38_int64_noncontig_eager": (
                    # Introduce gaps at positions 10 and 25
                    torch.tensor(
                        [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                          11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                          21, 22, 23, 24, 25, 27, 28, 29, 30, 31,
                          32, 33, 34, 35, 36, 37, 38, 39]],
                        dtype=torch.int64,
                    ),
                    1, False,
                ),
                "ne_1x38_int64_noncontig_compiled": (
                    torch.tensor(
                        [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                          11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                          21, 22, 23, 24, 25, 27, 28, 29, 30, 31,
                          32, 33, 34, 35, 36, 37, 38, 39]],
                        dtype=torch.int64,
                    ),
                    1, True,
                ),
            }
        },

        # ── Extra coverage: ne against tensor (tensor != tensor) ──────────────
        # While the model only calls ne(scalar=1), exercising the tensor-vs-
        # tensor path validates the kernel more broadly.
        ("test_torch_ne_pattern_002", "_run_ne_tensor_test"): {
            "param_sets": {
                "ne_1x38_int64_tensor_eager": (
                    torch.randint(0, 5, (1, 38), dtype=torch.int64),
                    torch.randint(0, 5, (1, 38), dtype=torch.int64),
                    False,
                ),
                "ne_1x38_int64_tensor_compiled": (
                    torch.randint(0, 5, (1, 38), dtype=torch.int64),
                    torch.randint(0, 5, (1, 38), dtype=torch.int64),
                    True,
                ),
            }
        },

        # ── Extra coverage: ne on a bool output with float32 input ────────────
        # Confirms that ne produces the correct bool dtype for float inputs.
        ("test_torch_ne_pattern_003", "_run_ne_scalar_test"): {
            "param_sets": {
                "ne_1x38_f32_scalar0_eager": (
                    torch.randn(1, 38, dtype=torch.float32),
                    0.0, False,
                ),
                "ne_1x38_f32_scalar0_compiled": (
                    torch.randn(1, 38, dtype=torch.float32),
                    0.0, True,
                ),
            }
        },
    }

    # ------------------------------------------------------------------
    # Base test methods
    # ------------------------------------------------------------------

    def _run_ne_scalar_test(self, a, scalar, compiled):
        """torch.ne(tensor, scalar) — elementwise inequality against a scalar."""
        compare_with_cpu(
            lambda x: torch.ne(x, scalar),
            a,
            compiled=compiled,
        )

    def _run_ne_tensor_test(self, a, b, compiled):
        """torch.ne(tensor, tensor) — elementwise inequality between two tensors."""
        compare_with_cpu(torch.ne, a, b, compiled=compiled)


# ─────────────────────────────────────────────────────────────────────────────
# TestCumsum
# ─────────────────────────────────────────────────────────────────────────────

class TestCumsum(unittest.TestCase, metaclass=ParameterizedTestMeta):
    """
    Tests for torch.cumsum patterns observed in
    mistralai/Mistral-Small-3.2-24B-Instruct-2506.

    torch.cumsum appears exclusively in the PREFILL scenario as part of the
    causal-mask construction pipeline (modeling_mistral.py:373), immediately
    after the torch.ne call:

        non_contiguous_mask = position_ids_delta.ne(1)   # [1,38] bool
        has_non_contiguous  = non_contiguous_mask.cumsum(dim=-1)   # [1,38] int64

    The cumulative sum converts the bool mask into a running count of
    non-contiguous position steps, used to select the correct causal-mask row.

    One unique call-site:
      cumsum.1  [1,38]  bool  dim=-1  → [1,38]  int64   PREFILL
               source: modeling_mistral.py:373
               input[0] : shape=(1, 38)  dtype=bool
               output  : shape=(1, 38)  dtype=int64

    Note: this op does NOT appear in the DECODE trace (seq_len=1 path skips
    the non-contiguous position-id check, same as torch.ne).
    """

    pytestmark = pytest.mark.torch_cumsum
    torch.manual_seed(0)

    PARAMS = {

        # ── cumsum.1  [1,38] bool dim=-1  — non-contiguous mask running count ─
        # All-False input (fully contiguous position IDs) — cumsum stays 0.
        ("test_torch_cumsum_pattern_000", "_run_cumsum_test"): {
            "param_sets": {
                "cumsum_1x38_bool_allfalse_eager": (
                    torch.zeros(1, 38, dtype=torch.bool), -1, False,
                ),
                "cumsum_1x38_bool_allfalse_compiled": (
                    torch.zeros(1, 38, dtype=torch.bool), -1, True,
                ),
            }
        },

        # Mixed True/False — represents non-contiguous position IDs.
        ("test_torch_cumsum_pattern_001", "_run_cumsum_test"): {
            "param_sets": {
                "cumsum_1x38_bool_mixed_eager": (
                    # Two gaps: at indices 10 and 25
                    torch.tensor(
                        [[False]*10 + [True] + [False]*14 + [True] + [False]*12],
                        dtype=torch.bool,
                    ),
                    -1, False,
                ),
                "cumsum_1x38_bool_mixed_compiled": (
                    torch.tensor(
                        [[False]*10 + [True] + [False]*14 + [True] + [False]*12],
                        dtype=torch.bool,
                    ),
                    -1, True,
                ),
            }
        },

        # All-True input — worst case, every step is non-contiguous.
        ("test_torch_cumsum_pattern_002", "_run_cumsum_test"): {
            "param_sets": {
                "cumsum_1x38_bool_alltrue_eager": (
                    torch.ones(1, 38, dtype=torch.bool), -1, False,
                ),
                "cumsum_1x38_bool_alltrue_compiled": (
                    torch.ones(1, 38, dtype=torch.bool), -1, True,
                ),
            }
        },

        # ── Extra coverage: int64 input (cumsum over the delta tensor itself) ─
        # The model only passes bool, but testing int64 validates the kernel
        # path used for the position-ids delta before the ne call.
        ("test_torch_cumsum_pattern_003", "_run_cumsum_test"): {
            "param_sets": {
                "cumsum_1x38_int64_dim1_eager": (
                    torch.randint(0, 4, (1, 38), dtype=torch.int64), -1, False,
                ),
                "cumsum_1x38_int64_dim1_compiled": (
                    torch.randint(0, 4, (1, 38), dtype=torch.int64), -1, True,
                ),
            }
        },

        # ── Extra coverage: dim=0 (column-wise cumsum) ────────────────────────
        # Exercises the less-common axis for completeness.
        ("test_torch_cumsum_pattern_004", "_run_cumsum_test"): {
            "param_sets": {
                "cumsum_1x38_bool_dim0_eager": (
                    torch.randint(0, 2, (1, 38), dtype=torch.bool), 0, False,
                ),
                "cumsum_1x38_bool_dim0_compiled": (
                    torch.randint(0, 2, (1, 38), dtype=torch.bool), 0, True,
                ),
            }
        },
    }

    # ------------------------------------------------------------------
    # Base test methods
    # ------------------------------------------------------------------

    def _run_cumsum_test(self, a, dim, compiled):
        """torch.cumsum(tensor, dim) — cumulative sum along the given dimension."""
        compare_with_cpu(
            lambda x: torch.cumsum(x, dim),
            a,
            compiled=compiled,
        )


# ===========================================================================
# TestSin
# ===========================================================================

class TestSin(unittest.TestCase, metaclass=ParameterizedTestMeta):
    """
    Tests for torch.sin patterns observed inMistral-Small-3.2-24B-Instruct-2506.

    ``torch.sin`` is a unary element-wise op — it takes an input tensor and
    returns a new tensor with the sine of each element.

    Shapes sourced from theMistral model's rotary embedding paths:
      [1, 38, 128]  — prefill
      [1, 1, 128]   — single-token decode
    """

    pytestmark = pytest.mark.torch_sin

    torch.manual_seed(0)

    PARAMS = {
        # ------------------------------------------------------------------
        # pattern_000  prefill shape [1, 38, 128], float16
        # ------------------------------------------------------------------
        ("test_torch_sin_pattern_000", "_run_sin_test"): {
            "param_sets": {
                "1x38x128_fp16_eager": (
                    torch.randn(1, 38, 128, dtype=torch.float16),
                    False,
                ),
                "1x38x128_fp16_compiled": (
                    torch.randn(1, 38, 128, dtype=torch.float16),
                    True,
                ),
            },
        },
        # ------------------------------------------------------------------
        # pattern_001  decode shape [1, 1, 128], float16
        # ------------------------------------------------------------------
        ("test_torch_sin_pattern_001", "_run_sin_test"): {
            "param_sets": {
                "1x1x128_fp16_eager": (
                    torch.randn(1, 1, 128, dtype=torch.float16),
                    False,
                ),
                "1x1x128_fp16_compiled": (
                    torch.randn(1, 1, 128, dtype=torch.float16),
                    True,
                ),
            },
        },
        # ------------------------------------------------------------------
        # pattern_002  out= parameter, prefill shape [1, 38, 128]
        # ------------------------------------------------------------------
        ("test_torch_sin_pattern_002", "_run_sin_out_test"): {
            "param_sets": {
                "1x38x128_out_eager": (
                    torch.randn(1, 38, 128, dtype=torch.float16),
                    False,
                ),
                "1x38x128_out_compiled": (
                    torch.randn(1, 38, 128, dtype=torch.float16),
                    True,
                ),
            },
        },
        # ------------------------------------------------------------------
        # pattern_003  out= parameter, decode shape [1, 1, 128]
        # ------------------------------------------------------------------
        ("test_torch_sin_pattern_003", "_run_sin_out_test"): {
            "param_sets": {
                "1x1x128_out_eager": (
                    torch.randn(1, 1, 128, dtype=torch.float16),
                    False,
                ),
                "1x1x128_out_compiled": (
                    torch.randn(1, 1, 128, dtype=torch.float16),
                    True,
                ),
            },
        },
    }

    def _run_sin_test(self, input_tensor, compiled):
        """torch.sin(tensor) — unary element-wise sine."""
        compare_with_cpu(torch.sin, input_tensor, compiled=compiled)

    def _run_sin_out_test(self, input_tensor, compiled):
        """Verify out= writes into a pre-allocated tensor."""
        shape = input_tensor.shape
        dtype = input_tensor.dtype

        def sin_out_fn(x):
            out = torch.empty(shape, dtype=dtype, device=x.device)
            result = torch.sin(x, out=out)
            assert result is out, f"{x.device}: out= did not return the same tensor"
            return result

        compare_with_cpu(sin_out_fn, input_tensor, compiled=compiled)


# ===========================================================================
# TestExpand
# ===========================================================================

class TestExpand(unittest.TestCase, metaclass=ParameterizedTestMeta):
    """
    Tests for torch.expand patterns observed inMistral-Small-3.2-24B-Instruct-2506.

    ``torch.expand`` is a view op — it returns a new view of the input
    tensor with singleton dimensions expanded to a larger size.

    Shape sourced from theMistral model:
      [1, 64, 1] -> [1, 64, 1]  — attention mask broadcast expand
    """

    pytestmark = pytest.mark.torch_expand

    torch.manual_seed(0)

    PARAMS = {
        # ------------------------------------------------------------------
        # pattern_000  [1,64,1] -> [1,64,1] with explicit sizes
        # ------------------------------------------------------------------
        ("test_torch_expand_pattern_000", "_run_expand_test"): {
            "param_sets": {
                "1x1x1_to_1x64x1_eager": (
                    torch.randn(1, 64, 1, dtype=torch.float16),
                    (1, 64, 1),
                    False,
                ),
                "1x1x1_to_1x64x1_compiled": (
                    torch.randn(1, 64, 1, dtype=torch.float16),
                    (1, 64, 1),
                    True,
                ),
            },
        },
        # ------------------------------------------------------------------
        # pattern_001  [1,64,1] -> [1,64,1] using -1
        # ------------------------------------------------------------------
        ("test_torch_expand_pattern_001", "_run_expand_test"): {
            "param_sets": {
                "1x1x1_neg1_eager": (
                    torch.randn(1, 64, 1, dtype=torch.float16),
                    (-1, 64, -1),
                    False,
                ),
                "1x1x1_neg1_compiled": (
                    torch.randn(1, 64, 1, dtype=torch.float16),
                    (-1, 64, -1),
                    True,
                ),
            },
        },
    }

    def _run_expand_test(self, input_tensor, target_size, compiled):
        """Tensor.expand(*sizes) — expand singleton dims to target size."""
        def expand_fn(x):
            return x.expand(*target_size)

        compare_with_cpu(expand_fn, input_tensor, compiled=compiled)


# ===========================================================================
# TestEq
# ===========================================================================

class TestEq(unittest.TestCase, metaclass=ParameterizedTestMeta):
    """
    Tests for torch.eq patterns observed inMistral-Small-3.2-24B-Instruct-2506.

    ``torch.eq(input, other)`` computes element-wise equality and returns
    a boolean tensor.  The second argument can be a number or a tensor.

    Shape sourced from theMistral model:
      [1]  — scalar-like comparison
    """

    pytestmark = pytest.mark.torch_eq

    torch.manual_seed(0)

    PARAMS = {
        # ------------------------------------------------------------------
        # pattern_000  tensor vs tensor: [1] == [1]
        # ------------------------------------------------------------------
        ("test_torch_eq_pattern_000", "_run_eq_tensor_test"): {
            "param_sets": {
                "1_tensor_eager": (
                    torch.randn(1, dtype=torch.float16),
                    torch.randn(1, dtype=torch.float16),
                    False,
                ),
                "1_tensor_compiled": (
                    torch.randn(1, dtype=torch.float16),
                    torch.randn(1, dtype=torch.float16),
                    True,
                ),
            },
        },
        # ------------------------------------------------------------------
        # pattern_001  tensor vs scalar: [1] == random float
        # ------------------------------------------------------------------
        ("test_torch_eq_pattern_001", "_run_eq_scalar_test"): {
            "param_sets": {
                "1_scalar_eager": (
                    torch.randn(1, dtype=torch.float16),
                    torch.randn(1).item(),
                    False,
                ),
                "1_scalar_compiled": (
                    torch.randn(1, dtype=torch.float16),
                    torch.randn(1).item(),
                    True,
                ),
            },
        },
        # ------------------------------------------------------------------
        # pattern_002  out= parameter: [1] == [1]
        # ------------------------------------------------------------------
        ("test_torch_eq_pattern_002", "_run_eq_out_test"): {
            "param_sets": {
                "1_out_eager": (
                    torch.randn(1, dtype=torch.float16),
                    torch.randn(1, dtype=torch.float16),
                    False,
                ),
                "1_out_compiled": (
                    torch.randn(1, dtype=torch.float16),
                    torch.randn(1, dtype=torch.float16),
                    True,
                ),
            },
        },
    }

    def _run_eq_tensor_test(self, input_tensor, other_tensor, compiled):
        """torch.eq(input, other) — both args tensors."""
        def eq_fn(x, y):
            return torch.eq(x, y)

        compare_with_cpu(eq_fn, input_tensor, other_tensor, compiled=compiled)

    def _run_eq_scalar_test(self, input_tensor, scalar, compiled):
        """torch.eq(input, scalar) — tensor vs number."""
        def eq_scalar_fn(x):
            return torch.eq(x, scalar)

        compare_with_cpu(eq_scalar_fn, input_tensor, compiled=compiled)

    def _run_eq_out_test(self, input_tensor, other_tensor, compiled):
        """Verify out= writes into a pre-allocated boolean tensor."""
        shape = input_tensor.shape

        def eq_out_fn(x, y):
            out = torch.empty(shape, dtype=torch.bool, device=x.device)
            result = torch.eq(x, y, out=out)
            assert result is out, f"{x.device}: out= did not return the same tensor"
            return result

        compare_with_cpu(eq_out_fn, input_tensor, other_tensor, compiled=compiled)


# ===========================================================================
# TestArange
# ===========================================================================

class TestArange(unittest.TestCase, metaclass=ParameterizedTestMeta):
    """
    Tests for torch.arange patterns observed inMistral-Small-3.2-24B-Instruct-2506.

    ``torch.arange`` is a factory op — it creates a 1-D tensor from scalar
    arguments.  Tests use ``needs_device=True`` so that ``device=`` is
    injected automatically.

    Output shapes sourced from theMistral model:
      [1]   — single-element
      [38]  — 38-element
    """

    pytestmark = pytest.mark.torch_arange

    torch.manual_seed(0)

    PARAMS = {
        # ------------------------------------------------------------------
        # pattern_000  arange(1) — single-element boundary
        # ------------------------------------------------------------------
        ("test_torch_arange_pattern_000", "_run_arange_test"): {
            "param_sets": {
                "shape_1_eager": (1, False),
                "shape_1_compiled": (1, True),
            },
        },
        # ------------------------------------------------------------------
        # pattern_001  arange(38) — multi-element range
        # ------------------------------------------------------------------
        ("test_torch_arange_pattern_001", "_run_arange_test"): {
            "param_sets": {
                "shape_38_eager": (38, False),
                "shape_38_compiled": (38, True),
            },
        },
        # ------------------------------------------------------------------
        # pattern_002  arange(0, 38, 2) — step parameter
        # ------------------------------------------------------------------
        ("test_torch_arange_pattern_002", "_run_arange_step_test"): {
            "param_sets": {
                "shape_38_step_2_eager": (0, 38, 2, False),
                "shape_38_step_2_compiled": (0, 38, 2, True),
            },
        },
        # ------------------------------------------------------------------
        # pattern_003  arange(38, out=...) — out parameter
        # ------------------------------------------------------------------
        ("test_torch_arange_pattern_003", "_run_arange_out_test"): {
            "param_sets": {
                "shape_38_out_eager": (38, False),
                "shape_38_out_compiled": (38, True),
            },
        },
    }

    def _run_arange_test(self, end, compiled):
        """torch.arange(end) — basic range with default start=0 and step=1."""
        def arange_fn(device=None):
            return torch.arange(end, dtype=torch.float16, device=device)

        compare_with_cpu(arange_fn, compiled=compiled, needs_device=True)

    def _run_arange_step_test(self, start, end, step, compiled):
        """torch.arange(start, end, step) — range with custom step."""
        def arange_fn(device=None):
            return torch.arange(start, end, step, dtype=torch.float16, device=device)

        compare_with_cpu(arange_fn, compiled=compiled, needs_device=True)

    def _run_arange_out_test(self, end, compiled):
        """Verify out= writes into a pre-allocated tensor."""
        cpu_out = torch.empty(end, dtype=torch.float16, device="cpu")
        cpu_result = torch.arange(end, dtype=torch.float16, out=cpu_out)
        assert cpu_result is cpu_out, "CPU: out= did not return the same tensor"

        alt_out = torch.empty(end, dtype=torch.float16, device=DEVICE)
        alt_result = torch.arange(end, dtype=torch.float16, out=alt_out)
        assert alt_result is alt_out, f"{DEVICE}: out= did not return the same tensor"

        torch.testing.assert_close(
            alt_result.to("cpu"), cpu_result,
            msg=(
                f"\nCPU vs {DEVICE} mismatch for torch.arange with out= parameter.\n"
                f"  end: {end}"
            ),
        )


# ===========================================================================
# torch.nn.functional.linear
# ===========================================================================

class TestLinear(unittest.TestCase, metaclass=ParameterizedTestMeta):
    """
    Tests for torch.nn.functional.linear patterns observed in
   Mistral-Small-3.2-24B-Instruct-2506.

    ``F.linear(input, weight, bias=None)`` applies y = x A^T + b.

    Shapes sourced from theMistral model's projection layers.
    Patterns 000-005: prefill (seq_len=38), without bias
    Patterns 006-011: decode  (seq_len=1),  without bias
    Patterns 012-017: prefill (seq_len=38), with bias
    Patterns 018-023: decode  (seq_len=1),  with bias
    """

    pytestmark = pytest.mark.torch_linear

    torch.manual_seed(0)

    PARAMS = {
        # ==================================================================
        # Without bias — prefill (seq_len=38) — patterns 000 through 005
        # ==================================================================

        ("test_torch_linear_pattern_000", "_run_linear_test"): {
            "param_sets": {
                "1x38x5120_4096x5120_eager": (
                    torch.randn(1, 38, 5120, dtype=torch.float16),
                    torch.randn(4096, 5120, dtype=torch.float16),
                    False,
                ),
                "1x38x5120_4096x5120_compiled": (
                    torch.randn(1, 38, 5120, dtype=torch.float16),
                    torch.randn(4096, 5120, dtype=torch.float16),
                    True,
                ),
            },
        },
        ("test_torch_linear_pattern_001", "_run_linear_test"): {
            "param_sets": {
                "1x38x5120_1024x5120_eager": (
                    torch.randn(1, 38, 5120, dtype=torch.float16),
                    torch.randn(1024, 5120, dtype=torch.float16),
                    False,
                ),
                "1x38x5120_1024x5120_compiled": (
                    torch.randn(1, 38, 5120, dtype=torch.float16),
                    torch.randn(1024, 5120, dtype=torch.float16),
                    True,
                ),
            },
        },
        ("test_torch_linear_pattern_002", "_run_linear_test"): {
            "param_sets": {
                "1x38x4096_5120x4096_eager": (
                    torch.randn(1, 38, 4096, dtype=torch.float16),
                    torch.randn(5120, 4096, dtype=torch.float16),
                    False,
                ),
                "1x38x4096_5120x4096_compiled": (
                    torch.randn(1, 38, 4096, dtype=torch.float16),
                    torch.randn(5120, 4096, dtype=torch.float16),
                    True,
                ),
            },
        },
        ("test_torch_linear_pattern_003", "_run_linear_test"): {
            "param_sets": {
                "1x38x5120_32768x5120_eager": (
                    torch.randn(1, 38, 5120, dtype=torch.float16),
                    torch.randn(32768, 5120, dtype=torch.float16),
                    False,
                ),
                "1x38x5120_32768x5120_compiled": (
                    torch.randn(1, 38, 5120, dtype=torch.float16),
                    torch.randn(32768, 5120, dtype=torch.float16),
                    True,
                ),
            },
        },
        ("test_torch_linear_pattern_004", "_run_linear_test"): {
            "param_sets": {
                "1x38x32768_5120x32768_eager": (
                    torch.randn(1, 38, 32768, dtype=torch.float16),
                    torch.randn(5120, 32768, dtype=torch.float16),
                    False,
                ),
                "1x38x32768_5120x32768_compiled": (
                    torch.randn(1, 38, 32768, dtype=torch.float16),
                    torch.randn(5120, 32768, dtype=torch.float16),
                    True,
                ),
            },
        },
        ("test_torch_linear_pattern_005", "_run_linear_test"): {
            "param_sets": {
                "1x38x5120_131072x5120_eager": (
                    torch.randn(1, 38, 5120, dtype=torch.float16),
                    torch.randn(131072, 5120, dtype=torch.float16),
                    False,
                ),
                "1x38x5120_131072x5120_compiled": (
                    torch.randn(1, 38, 5120, dtype=torch.float16),
                    torch.randn(131072, 5120, dtype=torch.float16),
                    True,
                ),
            },
        },

        # ==================================================================
        # Without bias — decode (seq_len=1) — patterns 006 through 011
        # ==================================================================

        ("test_torch_linear_pattern_006", "_run_linear_test"): {
            "param_sets": {
                "1x1x5120_4096x5120_eager": (
                    torch.randn(1, 1, 5120, dtype=torch.float16),
                    torch.randn(4096, 5120, dtype=torch.float16),
                    False,
                ),
                "1x1x5120_4096x5120_compiled": (
                    torch.randn(1, 1, 5120, dtype=torch.float16),
                    torch.randn(4096, 5120, dtype=torch.float16),
                    True,
                ),
            },
        },
        ("test_torch_linear_pattern_007", "_run_linear_test"): {
            "param_sets": {
                "1x1x5120_1024x5120_eager": (
                    torch.randn(1, 1, 5120, dtype=torch.float16),
                    torch.randn(1024, 5120, dtype=torch.float16),
                    False,
                ),
                "1x1x5120_1024x5120_compiled": (
                    torch.randn(1, 1, 5120, dtype=torch.float16),
                    torch.randn(1024, 5120, dtype=torch.float16),
                    True,
                ),
            },
        },
        ("test_torch_linear_pattern_008", "_run_linear_test"): {
            "param_sets": {
                "1x1x4096_5120x4096_eager": (
                    torch.randn(1, 1, 4096, dtype=torch.float16),
                    torch.randn(5120, 4096, dtype=torch.float16),
                    False,
                ),
                "1x1x4096_5120x4096_compiled": (
                    torch.randn(1, 1, 4096, dtype=torch.float16),
                    torch.randn(5120, 4096, dtype=torch.float16),
                    True,
                ),
            },
        },
        ("test_torch_linear_pattern_009", "_run_linear_test"): {
            "param_sets": {
                "1x1x5120_32768x5120_eager": (
                    torch.randn(1, 1, 5120, dtype=torch.float16),
                    torch.randn(32768, 5120, dtype=torch.float16),
                    False,
                ),
                "1x1x5120_32768x5120_compiled": (
                    torch.randn(1, 1, 5120, dtype=torch.float16),
                    torch.randn(32768, 5120, dtype=torch.float16),
                    True,
                ),
            },
        },
        ("test_torch_linear_pattern_010", "_run_linear_test"): {
            "param_sets": {
                "1x1x32768_5120x32768_eager": (
                    torch.randn(1, 1, 32768, dtype=torch.float16),
                    torch.randn(5120, 32768, dtype=torch.float16),
                    False,
                ),
                "1x1x32768_5120x32768_compiled": (
                    torch.randn(1, 1, 32768, dtype=torch.float16),
                    torch.randn(5120, 32768, dtype=torch.float16),
                    True,
                ),
            },
        },
        ("test_torch_linear_pattern_011", "_run_linear_test"): {
            "param_sets": {
                "1x1x5120_131072x5120_eager": (
                    torch.randn(1, 1, 5120, dtype=torch.float16),
                    torch.randn(131072, 5120, dtype=torch.float16),
                    False,
                ),
                "1x1x5120_131072x5120_compiled": (
                    torch.randn(1, 1, 5120, dtype=torch.float16),
                    torch.randn(131072, 5120, dtype=torch.float16),
                    True,
                ),
            },
        },

        # ==================================================================
        # With bias — prefill (seq_len=38) — patterns 012 through 017
        # ==================================================================

        ("test_torch_linear_pattern_012", "_run_linear_bias_test"): {
            "param_sets": {
                "1x38x5120_4096x5120_bias_eager": (
                    torch.randn(1, 38, 5120, dtype=torch.float16),
                    torch.randn(4096, 5120, dtype=torch.float16),
                    torch.randn(4096, dtype=torch.float16),
                    False,
                ),
                "1x38x5120_4096x5120_bias_compiled": (
                    torch.randn(1, 38, 5120, dtype=torch.float16),
                    torch.randn(4096, 5120, dtype=torch.float16),
                    torch.randn(4096, dtype=torch.float16),
                    True,
                ),
            },
        },
        ("test_torch_linear_pattern_013", "_run_linear_bias_test"): {
            "param_sets": {
                "1x38x5120_1024x5120_bias_eager": (
                    torch.randn(1, 38, 5120, dtype=torch.float16),
                    torch.randn(1024, 5120, dtype=torch.float16),
                    torch.randn(1024, dtype=torch.float16),
                    False,
                ),
                "1x38x5120_1024x5120_bias_compiled": (
                    torch.randn(1, 38, 5120, dtype=torch.float16),
                    torch.randn(1024, 5120, dtype=torch.float16),
                    torch.randn(1024, dtype=torch.float16),
                    True,
                ),
            },
        },
        ("test_torch_linear_pattern_014", "_run_linear_bias_test"): {
            "param_sets": {
                "1x38x4096_5120x4096_bias_eager": (
                    torch.randn(1, 38, 4096, dtype=torch.float16),
                    torch.randn(5120, 4096, dtype=torch.float16),
                    torch.randn(5120, dtype=torch.float16),
                    False,
                ),
                "1x38x4096_5120x4096_bias_compiled": (
                    torch.randn(1, 38, 4096, dtype=torch.float16),
                    torch.randn(5120, 4096, dtype=torch.float16),
                    torch.randn(5120, dtype=torch.float16),
                    True,
                ),
            },
        },
        ("test_torch_linear_pattern_015", "_run_linear_bias_test"): {
            "param_sets": {
                "1x38x5120_32768x5120_bias_eager": (
                    torch.randn(1, 38, 5120, dtype=torch.float16),
                    torch.randn(32768, 5120, dtype=torch.float16),
                    torch.randn(32768, dtype=torch.float16),
                    False,
                ),
                "1x38x5120_32768x5120_bias_compiled": (
                    torch.randn(1, 38, 5120, dtype=torch.float16),
                    torch.randn(32768, 5120, dtype=torch.float16),
                    torch.randn(32768, dtype=torch.float16),
                    True,
                ),
            },
        },
        ("test_torch_linear_pattern_016", "_run_linear_bias_test"): {
            "param_sets": {
                "1x38x32768_5120x32768_bias_eager": (
                    torch.randn(1, 38, 32768, dtype=torch.float16),
                    torch.randn(5120, 32768, dtype=torch.float16),
                    torch.randn(5120, dtype=torch.float16),
                    False,
                ),
                "1x38x32768_5120x32768_bias_compiled": (
                    torch.randn(1, 38, 32768, dtype=torch.float16),
                    torch.randn(5120, 32768, dtype=torch.float16),
                    torch.randn(5120, dtype=torch.float16),
                    True,
                ),
            },
        },
        ("test_torch_linear_pattern_017", "_run_linear_bias_test"): {
            "param_sets": {
                "1x38x5120_131072x5120_bias_eager": (
                    torch.randn(1, 38, 5120, dtype=torch.float16),
                    torch.randn(131072, 5120, dtype=torch.float16),
                    torch.randn(131072, dtype=torch.float16),
                    False,
                ),
                "1x38x5120_131072x5120_bias_compiled": (
                    torch.randn(1, 38, 5120, dtype=torch.float16),
                    torch.randn(131072, 5120, dtype=torch.float16),
                    torch.randn(131072, dtype=torch.float16),
                    True,
                ),
            },
        },

        # ==================================================================
        # With bias — decode (seq_len=1) — patterns 018 through 023
        # ==================================================================

        ("test_torch_linear_pattern_018", "_run_linear_bias_test"): {
            "param_sets": {
                "1x1x5120_4096x5120_bias_eager": (
                    torch.randn(1, 1, 5120, dtype=torch.float16),
                    torch.randn(4096, 5120, dtype=torch.float16),
                    torch.randn(4096, dtype=torch.float16),
                    False,
                ),
                "1x1x5120_4096x5120_bias_compiled": (
                    torch.randn(1, 1, 5120, dtype=torch.float16),
                    torch.randn(4096, 5120, dtype=torch.float16),
                    torch.randn(4096, dtype=torch.float16),
                    True,
                ),
            },
        },
        ("test_torch_linear_pattern_019", "_run_linear_bias_test"): {
            "param_sets": {
                "1x1x5120_1024x5120_bias_eager": (
                    torch.randn(1, 1, 5120, dtype=torch.float16),
                    torch.randn(1024, 5120, dtype=torch.float16),
                    torch.randn(1024, dtype=torch.float16),
                    False,
                ),
                "1x1x5120_1024x5120_bias_compiled": (
                    torch.randn(1, 1, 5120, dtype=torch.float16),
                    torch.randn(1024, 5120, dtype=torch.float16),
                    torch.randn(1024, dtype=torch.float16),
                    True,
                ),
            },
        },
        ("test_torch_linear_pattern_020", "_run_linear_bias_test"): {
            "param_sets": {
                "1x1x4096_5120x4096_bias_eager": (
                    torch.randn(1, 1, 4096, dtype=torch.float16),
                    torch.randn(5120, 4096, dtype=torch.float16),
                    torch.randn(5120, dtype=torch.float16),
                    False,
                ),
                "1x1x4096_5120x4096_bias_compiled": (
                    torch.randn(1, 1, 4096, dtype=torch.float16),
                    torch.randn(5120, 4096, dtype=torch.float16),
                    torch.randn(5120, dtype=torch.float16),
                    True,
                ),
            },
        },
        ("test_torch_linear_pattern_021", "_run_linear_bias_test"): {
            "param_sets": {
                "1x1x5120_32768x5120_bias_eager": (
                    torch.randn(1, 1, 5120, dtype=torch.float16),
                    torch.randn(32768, 5120, dtype=torch.float16),
                    torch.randn(32768, dtype=torch.float16),
                    False,
                ),
                "1x1x5120_32768x5120_bias_compiled": (
                    torch.randn(1, 1, 5120, dtype=torch.float16),
                    torch.randn(32768, 5120, dtype=torch.float16),
                    torch.randn(32768, dtype=torch.float16),
                    True,
                ),
            },
        },
        ("test_torch_linear_pattern_022", "_run_linear_bias_test"): {
            "param_sets": {
                "1x1x32768_5120x32768_bias_eager": (
                    torch.randn(1, 1, 32768, dtype=torch.float16),
                    torch.randn(5120, 32768, dtype=torch.float16),
                    torch.randn(5120, dtype=torch.float16),
                    False,
                ),
                "1x1x32768_5120x32768_bias_compiled": (
                    torch.randn(1, 1, 32768, dtype=torch.float16),
                    torch.randn(5120, 32768, dtype=torch.float16),
                    torch.randn(5120, dtype=torch.float16),
                    True,
                ),
            },
        },
        ("test_torch_linear_pattern_023", "_run_linear_bias_test"): {
            "param_sets": {
                "1x1x5120_131072x5120_bias_eager": (
                    torch.randn(1, 1, 5120, dtype=torch.float16),
                    torch.randn(131072, 5120, dtype=torch.float16),
                    torch.randn(131072, dtype=torch.float16),
                    False,
                ),
                "1x1x5120_131072x5120_bias_compiled": (
                    torch.randn(1, 1, 5120, dtype=torch.float16),
                    torch.randn(131072, 5120, dtype=torch.float16),
                    torch.randn(131072, dtype=torch.float16),
                    True,
                ),
            },
        },
    }

    def _run_linear_test(self, input_tensor, weight, compiled):
        """F.linear(input, weight) — without bias."""
        compare_with_cpu(
            lambda x, w: F.linear(x, w),
            input_tensor, weight,
            compiled=compiled,
        )

    def _run_linear_bias_test(self, input_tensor, weight, bias, compiled):
        """F.linear(input, weight, bias) — with bias."""
        compare_with_cpu(
            lambda x, w, b: F.linear(x, w, b),
            input_tensor, weight, bias,
            compiled=compiled,
        )

class TestView(unittest.TestCase, metaclass=ParameterizedTestMeta):
    """
    Tests for torch.Tensor.view() patterns observed in
    mistralai/Mistral-Small-3.2-24B-Instruct-2506.
 
    view() call-sites in the model
    --------------------------------
    Q/K/V projection reshape (MistralAttention):
      q: [batch, seq, NUM_Q_HEADS * HEAD_DIM]
         .view(batch, seq, NUM_Q_HEADS, HEAD_DIM)
      k: [batch, seq, NUM_KV_HEADS * HEAD_DIM]
         .view(batch, seq, NUM_KV_HEADS, HEAD_DIM)
 
    Attention output merge:
      out: [batch, NUM_Q_HEADS, seq, HEAD_DIM]
           .transpose(1,2).contiguous()
           .view(batch, seq, NUM_Q_HEADS * HEAD_DIM)
 
    MLP flatten / unflatten (MistralMLP):
      x: [batch, seq, HIDDEN_SIZE]
         .view(batch * seq, HIDDEN_SIZE)
      x: [batch * seq, INTERMEDIATE_SIZE]
         .view(batch, seq, INTERMEDIATE_SIZE)
 
    LM-head flatten:
      hidden: [batch, seq, HIDDEN_SIZE]
              .view(-1, HIDDEN_SIZE)
 
    Pattern layout
    --------------
      pattern_000–003 : Q/K split (prefill, batched, KV, decode)
      pattern_004–006 : Attention output merge
      pattern_007–009 : MLP flatten
      pattern_010–012 : MLP unflatten
      pattern_013–015 : Padded / medium / large-batch variants
    """
 
    pytestmark = pytest.mark.torch_view
 
    torch.manual_seed(0)
 
    PARAMS = {
        # ------------------------------------------------------------------
        # pattern_000
        # Q projection split — single batch prefill (seq=128)
        # (1, 128, 4096) → (1, 128, 32, 128)
        # ------------------------------------------------------------------
        ("test_torch_view_pattern_000", "_run_view_test"): {
            "param_sets": {
                "q_split_eager": (
                    torch.randn(1, 128, NUM_Q_HEADS * HEAD_DIM,
                                dtype=torch.float16),
                    (1, 128, NUM_Q_HEADS, HEAD_DIM),
                    False,
                ),
                "q_split_compiled": (
                    torch.randn(1, 128, NUM_Q_HEADS * HEAD_DIM,
                                dtype=torch.float16),
                    (1, 128, NUM_Q_HEADS, HEAD_DIM),
                    True,
                ),
            }
        },
 
        # ------------------------------------------------------------------
        # pattern_001
        # Q split — batched prefill (batch=4, seq=128)
        # (4, 128, 4096) → (4, 128, 32, 128)
        # ------------------------------------------------------------------
        ("test_torch_view_pattern_001", "_run_view_test"): {
            "param_sets": {
                "q_split_batch_eager": (
                    torch.randn(4, 128, NUM_Q_HEADS * HEAD_DIM,
                                dtype=torch.float16),
                    (4, 128, NUM_Q_HEADS, HEAD_DIM),
                    False,
                ),
                "q_split_batch_compiled": (
                    torch.randn(4, 128, NUM_Q_HEADS * HEAD_DIM,
                                dtype=torch.float16),
                    (4, 128, NUM_Q_HEADS, HEAD_DIM),
                    True,
                ),
            }
        },
 
        # ------------------------------------------------------------------
        # pattern_002
        # K projection split — KV heads (seq=128)
        # (1, 128, 1024) → (1, 128, 8, 128)
        # ------------------------------------------------------------------
        ("test_torch_view_pattern_002", "_run_view_test"): {
            "param_sets": {
                "k_split_eager": (
                    torch.randn(1, 128, NUM_KV_HEADS * HEAD_DIM,
                                dtype=torch.float16),
                    (1, 128, NUM_KV_HEADS, HEAD_DIM),
                    False,
                ),
                "k_split_compiled": (
                    torch.randn(1, 128, NUM_KV_HEADS * HEAD_DIM,
                                dtype=torch.float16),
                    (1, 128, NUM_KV_HEADS, HEAD_DIM),
                    True,
                ),
            }
        },
 
        # ------------------------------------------------------------------
        # pattern_003
        # Q split — decode step (seq=1)
        # (1, 1, 4096) → (1, 1, 32, 128)
        # ------------------------------------------------------------------
        ("test_torch_view_pattern_003", "_run_view_test"): {
            "param_sets": {
                "q_split_decode_eager": (
                    torch.randn(1, 1, NUM_Q_HEADS * HEAD_DIM,
                                dtype=torch.float16),
                    (1, 1, NUM_Q_HEADS, HEAD_DIM),
                    False,
                ),
                "q_split_decode_compiled": (
                    torch.randn(1, 1, NUM_Q_HEADS * HEAD_DIM,
                                dtype=torch.float16),
                    (1, 1, NUM_Q_HEADS, HEAD_DIM),
                    True,
                ),
            }
        },
 
        # ------------------------------------------------------------------
        # pattern_004
        # Attention output merge — single batch (seq=128)
        # (1, 128, 32, 128) → (1, 128, 4096)
        # ------------------------------------------------------------------
        ("test_torch_view_pattern_004", "_run_view_test"): {
            "param_sets": {
                "attn_merge_eager": (
                    torch.randn(1, 128, NUM_Q_HEADS, HEAD_DIM,
                                dtype=torch.float16)
                    .transpose(1, 2).contiguous(),
                    (1, 128, NUM_Q_HEADS * HEAD_DIM),
                    False,
                ),
                "attn_merge_compiled": (
                    torch.randn(1, 128, NUM_Q_HEADS, HEAD_DIM,
                                dtype=torch.float16)
                    .transpose(1, 2).contiguous(),
                    (1, 128, NUM_Q_HEADS * HEAD_DIM),
                    True,
                ),
            }
        },
 
        # ------------------------------------------------------------------
        # pattern_005
        # Attention output merge — batched (batch=4, seq=128)
        # (4, 128, 32, 128) → (4, 128, 4096)
        # ------------------------------------------------------------------
        ("test_torch_view_pattern_005", "_run_view_test"): {
            "param_sets": {
                "attn_merge_batch_eager": (
                    torch.randn(4, 128, NUM_Q_HEADS, HEAD_DIM,
                                dtype=torch.float16)
                    .transpose(1, 2).contiguous(),
                    (4, 128, NUM_Q_HEADS * HEAD_DIM),
                    False,
                ),
                "attn_merge_batch_compiled": (
                    torch.randn(4, 128, NUM_Q_HEADS, HEAD_DIM,
                                dtype=torch.float16)
                    .transpose(1, 2).contiguous(),
                    (4, 128, NUM_Q_HEADS * HEAD_DIM),
                    True,
                ),
            }
        },
 
        # ------------------------------------------------------------------
        # pattern_006
        # Attention output merge — decode (seq=1)
        # (1, 1, 32, 128) → (1, 1, 4096)
        # ------------------------------------------------------------------
        ("test_torch_view_pattern_006", "_run_view_test"): {
            "param_sets": {
                "attn_merge_decode_eager": (
                    torch.randn(1, 1, NUM_Q_HEADS, HEAD_DIM,
                                dtype=torch.float16)
                    .transpose(1, 2).contiguous(),
                    (1, 1, NUM_Q_HEADS * HEAD_DIM),
                    False,
                ),
                "attn_merge_decode_compiled": (
                    torch.randn(1, 1, NUM_Q_HEADS, HEAD_DIM,
                                dtype=torch.float16)
                    .transpose(1, 2).contiguous(),
                    (1, 1, NUM_Q_HEADS * HEAD_DIM),
                    True,
                ),
            }
        },
 
        # ------------------------------------------------------------------
        # pattern_007
        # MLP flatten — single batch (batch=1, seq=128)
        # (1, 128, 5120) → (128, 5120)
        # ------------------------------------------------------------------
        ("test_torch_view_pattern_007", "_run_view_test"): {
            "param_sets": {
                "mlp_flatten_eager": (
                    torch.randn(1, 128, HIDDEN_SIZE, dtype=torch.float16),
                    (128, HIDDEN_SIZE),
                    False,
                ),
                "mlp_flatten_compiled": (
                    torch.randn(1, 128, HIDDEN_SIZE, dtype=torch.float16),
                    (128, HIDDEN_SIZE),
                    True,
                ),
            }
        },
 
        # ------------------------------------------------------------------
        # pattern_008
        # MLP flatten — batched (batch=4, seq=128)
        # (4, 128, 5120) → (512, 5120)
        # ------------------------------------------------------------------
        ("test_torch_view_pattern_008", "_run_view_test"): {
            "param_sets": {
                "mlp_flatten_batch_eager": (
                    torch.randn(4, 128, HIDDEN_SIZE, dtype=torch.float16),
                    (4 * 128, HIDDEN_SIZE),
                    False,
                ),
                "mlp_flatten_batch_compiled": (
                    torch.randn(4, 128, HIDDEN_SIZE, dtype=torch.float16),
                    (4 * 128, HIDDEN_SIZE),
                    True,
                ),
            }
        },
 
        # ------------------------------------------------------------------
        # pattern_009
        # MLP intermediate flatten — gate output (batch=1, seq=128)
        # (1, 128, 32768) → (128, 32768)
        # ------------------------------------------------------------------
        ("test_torch_view_pattern_009", "_run_view_test"): {
            "param_sets": {
                "mlp_intermediate_flatten_eager": (
                    torch.randn(1, 128, INTERMEDIATE_SIZE, dtype=torch.float16),
                    (128, INTERMEDIATE_SIZE),
                    False,
                ),
                "mlp_intermediate_flatten_compiled": (
                    torch.randn(1, 128, INTERMEDIATE_SIZE, dtype=torch.float16),
                    (128, INTERMEDIATE_SIZE),
                    True,
                ),
            }
        },
 
        # ------------------------------------------------------------------
        # pattern_010
        # MLP unflatten — restore batch+seq (batch=1)
        # (128, 5120) → (1, 128, 5120)
        # ------------------------------------------------------------------
        ("test_torch_view_pattern_010", "_run_view_test"): {
            "param_sets": {
                "mlp_unflatten_eager": (
                    torch.randn(128, HIDDEN_SIZE, dtype=torch.float16),
                    (1, 128, HIDDEN_SIZE),
                    False,
                ),
                "mlp_unflatten_compiled": (
                    torch.randn(128, HIDDEN_SIZE, dtype=torch.float16),
                    (1, 128, HIDDEN_SIZE),
                    True,
                ),
            }
        },
 
        # ------------------------------------------------------------------
        # pattern_011
        # MLP unflatten — batched (batch=4)
        # (512, 5120) → (4, 128, 5120)
        # ------------------------------------------------------------------
        ("test_torch_view_pattern_011", "_run_view_test"): {
            "param_sets": {
                "mlp_unflatten_batch_eager": (
                    torch.randn(4 * 128, HIDDEN_SIZE, dtype=torch.float16),
                    (4, 128, HIDDEN_SIZE),
                    False,
                ),
                "mlp_unflatten_batch_compiled": (
                    torch.randn(4 * 128, HIDDEN_SIZE, dtype=torch.float16),
                    (4, 128, HIDDEN_SIZE),
                    True,
                ),
            }
        },
 
        # ------------------------------------------------------------------
        # pattern_012
        # LM-head flatten with -1 inference
        # (1, 128, 5120) → (128, 5120)
        # ------------------------------------------------------------------
        ("test_torch_view_pattern_012", "_run_view_test"): {
            "param_sets": {
                "lm_head_flatten_eager": (
                    torch.randn(1, 128, HIDDEN_SIZE, dtype=torch.float16),
                    (-1, HIDDEN_SIZE),
                    False,
                ),
                "lm_head_flatten_compiled": (
                    torch.randn(1, 128, HIDDEN_SIZE, dtype=torch.float16),
                    (-1, HIDDEN_SIZE),
                    True,
                ),
            }
        },
 
        # ------------------------------------------------------------------
        # pattern_013
        # Q split on padded input (actual=128, padded to 256)
        # (1, 256, 4096) → (1, 256, 32, 128)
        # ------------------------------------------------------------------
        ("test_torch_view_pattern_013", "_run_view_test"): {
            "param_sets": {
                "padded_q_split_eager": (
                    torch.cat([
                        torch.randn(1, 128, NUM_Q_HEADS * HEAD_DIM,
                                    dtype=torch.float16),
                        torch.zeros(1, 128, NUM_Q_HEADS * HEAD_DIM,
                                    dtype=torch.float16),
                    ], dim=1),
                    (1, 256, NUM_Q_HEADS, HEAD_DIM),
                    False,
                ),
                "padded_q_split_compiled": (
                    torch.cat([
                        torch.randn(1, 128, NUM_Q_HEADS * HEAD_DIM,
                                    dtype=torch.float16),
                        torch.zeros(1, 128, NUM_Q_HEADS * HEAD_DIM,
                                    dtype=torch.float16),
                    ], dim=1),
                    (1, 256, NUM_Q_HEADS, HEAD_DIM),
                    True,
                ),
            }
        },
 
        # ------------------------------------------------------------------
        # pattern_014
        # Medium-seq Q split (batch=2, seq=512)
        # (2, 512, 4096) → (2, 512, 32, 128)
        # ------------------------------------------------------------------
        ("test_torch_view_pattern_014", "_run_view_test"): {
            "param_sets": {
                "medium_q_split_eager": (
                    torch.randn(2, 512, NUM_Q_HEADS * HEAD_DIM,
                                dtype=torch.float16),
                    (2, 512, NUM_Q_HEADS, HEAD_DIM),
                    False,
                ),
                "medium_q_split_compiled": (
                    torch.randn(2, 512, NUM_Q_HEADS * HEAD_DIM,
                                dtype=torch.float16),
                    (2, 512, NUM_Q_HEADS, HEAD_DIM),
                    True,
                ),
            }
        },
 
        # ------------------------------------------------------------------
        # pattern_015
        # Large-batch MLP flatten (batch=8, seq=128)
        # (8, 128, 5120) → (1024, 5120)
        # ------------------------------------------------------------------
        ("test_torch_view_pattern_015", "_run_view_test"): {
            "param_sets": {
                "large_batch_mlp_flatten_eager": (
                    torch.randn(8, 128, HIDDEN_SIZE, dtype=torch.float16),
                    (8 * 128, HIDDEN_SIZE),
                    False,
                ),
                "large_batch_mlp_flatten_compiled": (
                    torch.randn(8, 128, HIDDEN_SIZE, dtype=torch.float16),
                    (8 * 128, HIDDEN_SIZE),
                    True,
                ),
            }
        },
    }
 
    def _run_view_test(self, tensor: torch.Tensor,
                        view_shape: tuple, compiled: bool):
        """view(*view_shape): verify shape, numel, zero-copy, then CPU compare."""
 
        def view_fn(t):
            return t.view(*view_shape)
 
        result = view_fn(tensor)
 
        # Resolve -1 dims before asserting shape
        total    = tensor.numel()
        resolved = []
        for i, s in enumerate(view_shape):
            if s == -1:
                other = 1
                for j, d in enumerate(view_shape):
                    if j != i and d != -1:
                        other *= d
                resolved.append(total // other)
            else:
                resolved.append(s)
        resolved = torch.Size(resolved)
 
        assert result.shape == resolved, (
            f"view() shape: got {tuple(result.shape)}, expected {tuple(resolved)}"
        )
        assert result.numel() == tensor.numel(), "view() changed numel"
        assert result.data_ptr() == tensor.data_ptr(), "view() must be zero-copy"
 
        compare_with_cpu(view_fn, tensor, compiled=compiled)

# ===========================================================================
# TestTo
# ===========================================================================
 
class TestTo(unittest.TestCase, metaclass=ParameterizedTestMeta):
    """
    Tests for Tensor.to() patterns observed in
    mistralai/Mistral-Small-3.2-24B-Instruct-2506.
 
    .to() call-sites
    ----------------
    Dtype transfer:
      - hidden_states.to(torch.float32)    RMSNorm upcast
      - hidden_states.to(input_dtype)      RMSNorm downcast
      - attention_mask.to(dtype)           Mask promotion
      - weights.to(torch.float16)          Weight loading
      - kv_cache.to(torch.bfloat16)        KV cache storage
      - ids.to(torch.int64)               Token-ID normalisation
 
    Device transfer (cpu baseline):
      - input_ids / mask / position_ids / activations / kv_cache to device
 
    Combined dtype + device:
      - tensor.to(device=device, dtype=dtype)
 
    Pattern layout
    --------------
      pattern_000–004 : dtype — activations & mask
      pattern_005–007 : dtype — weight matrices & KV cache
      pattern_008–010 : dtype — integer / bool tensors
      pattern_011–015 : device — various tensor types
      pattern_016–018 : combined dtype + device
    """
 
    pytestmark = pytest.mark.torch_to
 
    torch.manual_seed(0)
 
    PARAMS = {
        # ------------------------------------------------------------------
        # pattern_000
        # hidden_states.to(torch.float32) — RMSNorm upcast
        # (1, 128, 5120) fp16 → fp32
        # ------------------------------------------------------------------
        ("test_torch_to_pattern_000", "_run_to_test"): {
            "param_sets": {
                "fp16_to_fp32_eager": (
                    torch.randn(1, 128, HIDDEN_SIZE, dtype=torch.float16),
                    torch.float32, None, False,
                ),
                "fp16_to_fp32_compiled": (
                    torch.randn(1, 128, HIDDEN_SIZE, dtype=torch.float16),
                    torch.float32, None, True,
                ),
            }
        },
 
        # ------------------------------------------------------------------
        # pattern_001
        # hidden_states.to(input_dtype) — RMSNorm downcast (seq=512)
        # (1, 512, 5120) fp32 → fp16
        # ------------------------------------------------------------------
        ("test_torch_to_pattern_001", "_run_to_test"): {
            "param_sets": {
                "fp32_to_fp16_eager": (
                    torch.randn(1, 512, HIDDEN_SIZE, dtype=torch.float32),
                    torch.float16, None, False,
                ),
                "fp32_to_fp16_compiled": (
                    torch.randn(1, 512, HIDDEN_SIZE, dtype=torch.float32),
                    torch.float16, None, True,
                ),
            }
        },
 
        # ------------------------------------------------------------------
        # pattern_002
        # output.to(bfloat16) — mixed-precision activation cast (batch=4)
        # (4, 128, 5120) fp32 → bf16
        # ------------------------------------------------------------------
        ("test_torch_to_pattern_002", "_run_to_test"): {
            "param_sets": {
                "fp32_to_bf16_eager": (
                    torch.randn(4, 128, HIDDEN_SIZE, dtype=torch.float32),
                    torch.bfloat16, None, False,
                ),
                "fp32_to_bf16_compiled": (
                    torch.randn(4, 128, HIDDEN_SIZE, dtype=torch.float32),
                    torch.bfloat16, None, True,
                ),
            }
        },
 
        # ------------------------------------------------------------------
        # pattern_003
        # attention_mask.to(dtype) — bool → fp16 for additive masking
        # (1, 1, 128, 128)
        # ------------------------------------------------------------------
        ("test_torch_to_pattern_003", "_run_to_test"): {
            "param_sets": {
                "bool_to_fp16_eager": (
                    torch.randint(0, 2, (1, 1, 128, 128)).bool(),
                    torch.float16, None, False,
                ),
                "bool_to_fp16_compiled": (
                    torch.randint(0, 2, (1, 1, 128, 128)).bool(),
                    torch.float16, None, True,
                ),
            }
        },
 
        # ------------------------------------------------------------------
        # pattern_004
        # Padded activation upcast (batch=4, actual=128, padded to 256)
        # (4, 256, 5120) fp16 → fp32
        # ------------------------------------------------------------------
        ("test_torch_to_pattern_004", "_run_to_test"): {
            "param_sets": {
                "padded_fp16_to_fp32_eager": (
                    torch.cat([
                        torch.randn(4, 128, HIDDEN_SIZE, dtype=torch.float16),
                        torch.zeros(4, 128, HIDDEN_SIZE, dtype=torch.float16),
                    ], dim=1),
                    torch.float32, None, False,
                ),
                "padded_fp16_to_fp32_compiled": (
                    torch.cat([
                        torch.randn(4, 128, HIDDEN_SIZE, dtype=torch.float16),
                        torch.zeros(4, 128, HIDDEN_SIZE, dtype=torch.float16),
                    ], dim=1),
                    torch.float32, None, True,
                ),
            }
        },
 
        # ------------------------------------------------------------------
        # pattern_005
        # q_proj.weight.to(fp16) — QKV weight loading
        # (NUM_Q_HEADS*HEAD_DIM, HIDDEN_SIZE) = (4096, 5120) fp32 → fp16
        # ------------------------------------------------------------------
        ("test_torch_to_pattern_005", "_run_to_test"): {
            "param_sets": {
                "qkv_weight_fp32_to_fp16_eager": (
                    torch.randn(NUM_Q_HEADS * HEAD_DIM, HIDDEN_SIZE,
                                dtype=torch.float32),
                    torch.float16, None, False,
                ),
                "qkv_weight_fp32_to_fp16_compiled": (
                    torch.randn(NUM_Q_HEADS * HEAD_DIM, HIDDEN_SIZE,
                                dtype=torch.float32),
                    torch.float16, None, True,
                ),
            }
        },
 
        # ------------------------------------------------------------------
        # pattern_006
        # gate_proj.weight.to(bf16)
        # (INTERMEDIATE_SIZE, HIDDEN_SIZE) = (32768, 5120) fp32 → bf16
        # ------------------------------------------------------------------
        ("test_torch_to_pattern_006", "_run_to_test"): {
            "param_sets": {
                "gate_weight_fp32_to_bf16_eager": (
                    torch.randn(INTERMEDIATE_SIZE, HIDDEN_SIZE,
                                dtype=torch.float32),
                    torch.bfloat16, None, False,
                ),
                "gate_weight_fp32_to_bf16_compiled": (
                    torch.randn(INTERMEDIATE_SIZE, HIDDEN_SIZE,
                                dtype=torch.float32),
                    torch.bfloat16, None, True,
                ),
            }
        },
 
        # ------------------------------------------------------------------
        # pattern_007
        # kv_cache.to(fp16) — GQA KV storage
        # (1, NUM_KV_HEADS, 128, HEAD_DIM) = (1,8,128,128) fp32 → fp16
        # ------------------------------------------------------------------
        ("test_torch_to_pattern_007", "_run_to_test"): {
            "param_sets": {
                "kv_fp32_to_fp16_eager": (
                    torch.randn(1, NUM_KV_HEADS, 128, HEAD_DIM,
                                dtype=torch.float32),
                    torch.float16, None, False,
                ),
                "kv_fp32_to_fp16_compiled": (
                    torch.randn(1, NUM_KV_HEADS, 128, HEAD_DIM,
                                dtype=torch.float32),
                    torch.float16, None, True,
                ),
            }
        },
 
        # ------------------------------------------------------------------
        # pattern_008
        # input_ids.to(int64) — token-ID normalisation (batch=1, seq=128)
        # ------------------------------------------------------------------
        ("test_torch_to_pattern_008", "_run_to_test"): {
            "param_sets": {
                "int32_to_int64_eager": (
                    torch.randint(0, 1000, (1, 128), dtype=torch.int32),
                    torch.int64, None, False,
                ),
                "int32_to_int64_compiled": (
                    torch.randint(0, 1000, (1, 128), dtype=torch.int32),
                    torch.int64, None, True,
                ),
            }
        },
 
        # ------------------------------------------------------------------
        # pattern_009
        # position_ids.to(int64) — RoPE position normalisation
        # ------------------------------------------------------------------
        ("test_torch_to_pattern_009", "_run_to_test"): {
            "param_sets": {
                "pos_int32_to_int64_eager": (
                    torch.arange(128, dtype=torch.int32).unsqueeze(0),
                    torch.int64, None, False,
                ),
                "pos_int32_to_int64_compiled": (
                    torch.arange(128, dtype=torch.int32).unsqueeze(0),
                    torch.int64, None, True,
                ),
            }
        },
 
        # ------------------------------------------------------------------
        # pattern_010
        # causal_mask.to(fp32) — additive mask cast (1,1,128,128)
        # ------------------------------------------------------------------
        ("test_torch_to_pattern_010", "_run_to_test"): {
            "param_sets": {
                "bool_to_fp32_eager": (
                    torch.randint(0, 2, (1, 1, 128, 128)).bool(),
                    torch.float32, None, False,
                ),
                "bool_to_fp32_compiled": (
                    torch.randint(0, 2, (1, 1, 128, 128)).bool(),
                    torch.float32, None, True,
                ),
            }
        },
 
        # ------------------------------------------------------------------
        # pattern_011
        # hidden_states.to(device) — activation transfer (1, 128, 5120)
        # ------------------------------------------------------------------
        ("test_torch_to_pattern_011", "_run_to_test"): {
            "param_sets": {
                "activation_cpu_eager": (
                    torch.randn(1, 128, HIDDEN_SIZE, dtype=torch.float16),
                    None, "cpu", False,
                ),
                "activation_cpu_compiled": (
                    torch.randn(1, 128, HIDDEN_SIZE, dtype=torch.float16),
                    None, "cpu", True,
                ),
            }
        },
 
        # ------------------------------------------------------------------
        # pattern_012
        # batched hidden_states.to(device) (batch=4, seq=128)
        # ------------------------------------------------------------------
        ("test_torch_to_pattern_012", "_run_to_test"): {
            "param_sets": {
                "batch_cpu_eager": (
                    torch.randn(4, 128, HIDDEN_SIZE, dtype=torch.float16),
                    None, "cpu", False,
                ),
                "batch_cpu_compiled": (
                    torch.randn(4, 128, HIDDEN_SIZE, dtype=torch.float16),
                    None, "cpu", True,
                ),
            }
        },
 
        # ------------------------------------------------------------------
        # pattern_013
        # kv_cache.to(device) — (1, 8, 128, 128)
        # ------------------------------------------------------------------
        ("test_torch_to_pattern_013", "_run_to_test"): {
            "param_sets": {
                "kv_cpu_eager": (
                    torch.randn(1, NUM_KV_HEADS, 128, HEAD_DIM,
                                dtype=torch.float16),
                    None, "cpu", False,
                ),
                "kv_cpu_compiled": (
                    torch.randn(1, NUM_KV_HEADS, 128, HEAD_DIM,
                                dtype=torch.float16),
                    None, "cpu", True,
                ),
            }
        },
 
        # ------------------------------------------------------------------
        # pattern_014
        # input_ids.to(device) — (1, 128) int64
        # ------------------------------------------------------------------
        ("test_torch_to_pattern_014", "_run_to_test"): {
            "param_sets": {
                "ids_cpu_eager": (
                    torch.randint(0, 1000, (1, 128), dtype=torch.int64),
                    None, "cpu", False,
                ),
                "ids_cpu_compiled": (
                    torch.randint(0, 1000, (1, 128), dtype=torch.int64),
                    None, "cpu", True,
                ),
            }
        },
 
        # ------------------------------------------------------------------
        # pattern_015
        # position_ids.to(device) — (1, 128) int64
        # ------------------------------------------------------------------
        ("test_torch_to_pattern_015", "_run_to_test"): {
            "param_sets": {
                "pos_cpu_eager": (
                    torch.arange(128, dtype=torch.int64).unsqueeze(0),
                    None, "cpu", False,
                ),
                "pos_cpu_compiled": (
                    torch.arange(128, dtype=torch.int64).unsqueeze(0),
                    None, "cpu", True,
                ),
            }
        },
 
        # ------------------------------------------------------------------
        # pattern_016
        # Combined — fp32 activation → fp16 + cpu (1, 128, 5120)
        # ------------------------------------------------------------------
        ("test_torch_to_pattern_016", "_run_to_test"): {
            "param_sets": {
                "fp32_to_fp16_cpu_eager": (
                    torch.randn(1, 128, HIDDEN_SIZE, dtype=torch.float32),
                    torch.float16, "cpu", False,
                ),
                "fp32_to_fp16_cpu_compiled": (
                    torch.randn(1, 128, HIDDEN_SIZE, dtype=torch.float32),
                    torch.float16, "cpu", True,
                ),
            }
        },
 
        # ------------------------------------------------------------------
        # pattern_017
        # Combined — fp32 KV cache → bf16 + cpu
        # ------------------------------------------------------------------
        ("test_torch_to_pattern_017", "_run_to_test"): {
            "param_sets": {
                "fp32_kv_to_bf16_cpu_eager": (
                    torch.randn(1, NUM_KV_HEADS, 128, HEAD_DIM,
                                dtype=torch.float32),
                    torch.bfloat16, "cpu", False,
                ),
                "fp32_kv_to_bf16_cpu_compiled": (
                    torch.randn(1, NUM_KV_HEADS, 128, HEAD_DIM,
                                dtype=torch.float32),
                    torch.bfloat16, "cpu", True,
                ),
            }
        },
 
        # ------------------------------------------------------------------
        # pattern_018
        # Combined — batched fp16 → fp32 + cpu (4, 512, 5120)
        # ------------------------------------------------------------------
        ("test_torch_to_pattern_018", "_run_to_test"): {
            "param_sets": {
                "batch_fp16_to_fp32_cpu_eager": (
                    torch.randn(4, 512, HIDDEN_SIZE, dtype=torch.float16),
                    torch.float32, "cpu", False,
                ),
                "batch_fp16_to_fp32_cpu_compiled": (
                    torch.randn(4, 512, HIDDEN_SIZE, dtype=torch.float32),
                    torch.float32, "cpu", True,
                ),
            }
        },
    }
 
    def _run_to_test(self, tensor: torch.Tensor, dtype, device, compiled: bool):
        """tensor.to(dtype, device): verify dtype, device, shape, then CPU compare."""
        expected_dtype  = dtype  if dtype  is not None else tensor.dtype
        expected_device = torch.device(device) if device is not None \
                          else tensor.device
 
        def to_fn(t):
            kwargs = {}
            if dtype  is not None: kwargs["dtype"]  = dtype
            if device is not None: kwargs["device"] = device
            return t.to(**kwargs)
 
        result = to_fn(tensor)
        assert result.dtype == expected_dtype, (
            f".to() dtype: got {result.dtype}, expected {expected_dtype}"
        )
        assert result.device.type == expected_device.type, (
            f".to() device: got {result.device.type}, expected {expected_device.type}"
        )
        assert result.shape == tensor.shape, (
            f".to() shape changed: {tensor.shape} → {result.shape}"
        )
 
        compare_with_cpu(to_fn, tensor, compiled=compiled)

# ===========================================================================
# TestDiff
# ===========================================================================
 
class TestDiff(unittest.TestCase, metaclass=ParameterizedTestMeta):
    """
    Tests for torch.diff() patterns observed in
    mistralai/Mistral-Small-3.2-24B-Instruct-2506.
 
    torch.diff() call-sites in the model
    ---------------------------------------
    torch.diff(input, n=1, dim=-1) computes discrete differences.
    It appears in Mistral-Small-3.2 at:
 
    Vision token splitting (Mistral3ForConditionalGeneration):
      image_token_mask = (input_ids == IMAGE_TOKEN_ID)
      positions = torch.where(image_token_mask)[1]       # 1-D
      gaps = torch.diff(positions)                       # detect boundaries
      Used to locate where image patches start/end in the flat token stream.
 
    Position-ID boundary detection:
      torch.diff(position_ids, dim=-1)  — identify gaps in position sequence
      to handle non-contiguous cache segments during KV-cache reuse.
 
    Attention span computation:
      torch.diff(cumulative_lengths)    — recover individual segment lengths
      from prefix sums in variable-length batch packing.
 
    Pattern layout
    --------------
      pattern_000–003 : 1-D token position differences (image token splitting)
      pattern_004–006 : position-ID boundary detection (2-D, dim=-1)
      pattern_007–009 : segment-length recovery from cumsum (1-D and batched)
      pattern_010–012 : padded / large-batch / multi-step (n>1) variants
    """
 
    pytestmark = pytest.mark.torch_diff
 
    torch.manual_seed(0)
 
    PARAMS = {
        # ------------------------------------------------------------------
        # pattern_000
        # 1-D image token position diff (short prompt, ~8 image patches)
        # positions: [12, 13, 14, 15, 50, 51, 52, 53]
        # diff: [1, 1, 1, 35, 1, 1, 1]
        # ------------------------------------------------------------------
        ("test_torch_diff_pattern_000", "_run_diff_test"): {
            "param_sets": {
                "image_positions_eager": (
                    torch.tensor([12, 13, 14, 15, 50, 51, 52, 53],
                                 dtype=torch.int64),
                    1, -1, False,
                ),
                "image_positions_compiled": (
                    torch.tensor([12, 13, 14, 15, 50, 51, 52, 53],
                                 dtype=torch.int64),
                    1, -1, True,
                ),
            }
        },
 
        # ------------------------------------------------------------------
        # pattern_001
        # 1-D diff — longer image patch sequence (64 patches)
        # ------------------------------------------------------------------
        ("test_torch_diff_pattern_001", "_run_diff_test"): {
            "param_sets": {
                "long_positions_eager": (
                    torch.arange(0, 128, 2, dtype=torch.int64),
                    1, -1, False,
                ),
                "long_positions_compiled": (
                    torch.arange(0, 128, 2, dtype=torch.int64),
                    1, -1, True,
                ),
            }
        },
 
        # ------------------------------------------------------------------
        # pattern_002
        # 1-D diff — float activation sequence (e.g. cumulative score)
        # ------------------------------------------------------------------
        ("test_torch_diff_pattern_002", "_run_diff_test"): {
            "param_sets": {
                "float_seq_eager": (
                    torch.randn(128, dtype=torch.float32).cumsum(0),
                    1, -1, False,
                ),
                "float_seq_compiled": (
                    torch.randn(128, dtype=torch.float32).cumsum(0),
                    1, -1, True,
                ),
            }
        },
 
        # ------------------------------------------------------------------
        # pattern_003
        # 1-D diff — variable image sizes (positions with uneven gaps)
        # ------------------------------------------------------------------
        ("test_torch_diff_pattern_003", "_run_diff_test"): {
            "param_sets": {
                "uneven_gaps_eager": (
                    torch.tensor([0, 1, 2, 3, 20, 21, 22, 100, 101],
                                 dtype=torch.int64),
                    1, -1, False,
                ),
                "uneven_gaps_compiled": (
                    torch.tensor([0, 1, 2, 3, 20, 21, 22, 100, 101],
                                 dtype=torch.int64),
                    1, -1, True,
                ),
            }
        },
 
        # ------------------------------------------------------------------
        # pattern_004
        # Position-ID boundary detection — single batch (dim=-1)
        # position_ids: (1, 128) — diff along seq dim
        # ------------------------------------------------------------------
        ("test_torch_diff_pattern_004", "_run_diff_test"): {
            "param_sets": {
                "pos_ids_single_eager": (
                    torch.arange(128, dtype=torch.int64).unsqueeze(0),
                    1, -1, False,
                ),
                "pos_ids_single_compiled": (
                    torch.arange(128, dtype=torch.int64).unsqueeze(0),
                    1, -1, True,
                ),
            }
        },
 
        # ------------------------------------------------------------------
        # pattern_005
        # Position-ID detection — batched (batch=4, seq=128)
        # ------------------------------------------------------------------
        ("test_torch_diff_pattern_005", "_run_diff_test"): {
            "param_sets": {
                "pos_ids_batch_eager": (
                    torch.arange(128, dtype=torch.int64)
                    .unsqueeze(0).expand(4, -1).contiguous(),
                    1, -1, False,
                ),
                "pos_ids_batch_compiled": (
                    torch.arange(128, dtype=torch.int64)
                    .unsqueeze(0).expand(4, -1).contiguous(),
                    1, -1, True,
                ),
            }
        },
 
        # ------------------------------------------------------------------
        # pattern_006
        # Position-ID detection — with a gap (simulates KV-cache reuse)
        # e.g. positions 0-63 then 128-191
        # ------------------------------------------------------------------
        ("test_torch_diff_pattern_006", "_run_diff_test"): {
            "param_sets": {
                "pos_gap_eager": (
                    torch.cat([
                        torch.arange(64, dtype=torch.int64),
                        torch.arange(128, 192, dtype=torch.int64),
                    ]).unsqueeze(0),
                    1, -1, False,
                ),
                "pos_gap_compiled": (
                    torch.cat([
                        torch.arange(64, dtype=torch.int64),
                        torch.arange(128, 192, dtype=torch.int64),
                    ]).unsqueeze(0),
                    1, -1, True,
                ),
            }
        },
 
        # ------------------------------------------------------------------
        # pattern_007
        # Segment lengths from cumsum — recover individual lengths
        # cumsum([10, 20, 30, 40]) = [10, 30, 60, 100]
        # diff → [20, 30, 40]   (all but first)
        # ------------------------------------------------------------------
        ("test_torch_diff_pattern_007", "_run_diff_test"): {
            "param_sets": {
                "cumsum_lengths_eager": (
                    torch.tensor([10, 30, 60, 100, 150],
                                 dtype=torch.int64),
                    1, -1, False,
                ),
                "cumsum_lengths_compiled": (
                    torch.tensor([10, 30, 60, 100, 150],
                                 dtype=torch.int64),
                    1, -1, True,
                ),
            }
        },
 
        # ------------------------------------------------------------------
        # pattern_008
        # Segment lengths from cumsum — 2-D batch packing (batch=4)
        # ------------------------------------------------------------------
        ("test_torch_diff_pattern_008", "_run_diff_test"): {
            "param_sets": {
                "batch_cumsum_eager": (
                    torch.randint(1, 50, (4, 8), dtype=torch.int64).cumsum(-1),
                    1, -1, False,
                ),
                "batch_cumsum_compiled": (
                    torch.randint(1, 50, (4, 8), dtype=torch.int64).cumsum(-1),
                    1, -1, True,
                ),
            }
        },
 
        # ------------------------------------------------------------------
        # pattern_009
        # Float activations diff along seq dim (2-D, dim=-1)
        # (1, 128) — diff over sequence → (1, 127)
        # ------------------------------------------------------------------
        ("test_torch_diff_pattern_009", "_run_diff_test"): {
            "param_sets": {
                "float_batch_seq_eager": (
                    torch.randn(1, 128, dtype=torch.float32),
                    1, -1, False,
                ),
                "float_batch_seq_compiled": (
                    torch.randn(1, 128, dtype=torch.float32),
                    1, -1, True,
                ),
            }
        },
 
        # ------------------------------------------------------------------
        # pattern_010
        # n=2 second-order diff — acceleration of attention scores
        # 1-D float (128,) → (126,)
        # ------------------------------------------------------------------
        ("test_torch_diff_pattern_010", "_run_diff_test"): {
            "param_sets": {
                "second_order_eager": (
                    torch.randn(128, dtype=torch.float32),
                    2, -1, False,
                ),
                "second_order_compiled": (
                    torch.randn(128, dtype=torch.float32),
                    2, -1, True,
                ),
            }
        },
 
        # ------------------------------------------------------------------
        # pattern_011
        # Padded position IDs — zeros at end (actual=128, padded to 256)
        # ------------------------------------------------------------------
        ("test_torch_diff_pattern_011", "_run_diff_test"): {
            "param_sets": {
                "padded_pos_ids_eager": (
                    torch.cat([
                        torch.arange(128, dtype=torch.int64),
                        torch.zeros(128, dtype=torch.int64),
                    ]).unsqueeze(0),
                    1, -1, False,
                ),
                "padded_pos_ids_compiled": (
                    torch.cat([
                        torch.arange(128, dtype=torch.int64),
                        torch.zeros(128, dtype=torch.int64),
                    ]).unsqueeze(0),
                    1, -1, True,
                ),
            }
        },
 
        # ------------------------------------------------------------------
        # pattern_012
        # Large batch position-ID diff (batch=8, seq=128)
        # ------------------------------------------------------------------
        ("test_torch_diff_pattern_012", "_run_diff_test"): {
            "param_sets": {
                "large_batch_pos_eager": (
                    torch.arange(128, dtype=torch.int64)
                    .unsqueeze(0).expand(8, -1).contiguous(),
                    1, -1, False,
                ),
                "large_batch_pos_compiled": (
                    torch.arange(128, dtype=torch.int64)
                    .unsqueeze(0).expand(8, -1).contiguous(),
                    1, -1, True,
                ),
            }
        },
    }
 
    def _run_diff_test(self, tensor: torch.Tensor, n: int, dim: int,
                        compiled: bool):
        """torch.diff(tensor, n=n, dim=dim): verify output shape then CPU compare."""
 
        def diff_fn(t):
            return torch.diff(t, n=n, dim=dim)
 
        result = diff_fn(tensor)
 
        # Expected shape: shrinks by n along `dim`
        dim_idx = dim if dim >= 0 else tensor.dim() + dim
        expected = list(tensor.shape)
        expected[dim_idx] -= n
        assert result.shape == torch.Size(expected), (
            f"diff() shape: got {tuple(result.shape)}, expected {tuple(expected)}"
        )
 
        compare_with_cpu(diff_fn, tensor, compiled=compiled)

# ===========================================================================
# TestAll
# ===========================================================================
 
class TestAll(unittest.TestCase, metaclass=ParameterizedTestMeta):
    """
    Tests for torch.all() patterns observed in
    mistralai/Mistral-Small-3.2-24B-Instruct-2506.
 
    torch.all() call-sites in the model
    --------------------------------------
    torch.all() (global reduction or dim-wise) appears at:
 
    Stopping criteria (GenerationMixin.generate):
      torch.all(unfinished_sequences == 0)
      → True when every sequence in the batch has hit EOS.
 
    Attention mask validation:
      torch.all(attention_mask == 1, dim=-1)
      → Per-row check: is the entire row unpadded?
 
    Image token presence check:
      torch.all(image_token_mask, dim=-1)
      → Ensure every expected image-token slot is filled.
 
    Cache validity check:
      torch.all(cache_lengths > 0)
      → Confirm all KV-cache entries are populated.
 
    Pattern layout
    --------------
      pattern_000–003 : global reduction (EOS / validity checks)
      pattern_004–006 : dim=-1 (row-wise mask checks)
      pattern_007–009 : dim=0 / dim=1 column-wise checks
      pattern_010–012 : padded / batched / bool input variants
    """
 
    pytestmark = pytest.mark.torch_all
 
    torch.manual_seed(0)
 
    PARAMS = {
        # ------------------------------------------------------------------
        # pattern_000
        # Global all — EOS check: all sequences finished
        # unfinished_sequences == 0 → all True → generation stops
        # (batch=4,) int32
        # ------------------------------------------------------------------
        ("test_torch_all_pattern_000", "_run_all_test"): {
            "param_sets": {
                "eos_all_done_eager": (
                    torch.zeros(4, dtype=torch.int32),
                    None, False,
                ),
                "eos_all_done_compiled": (
                    torch.zeros(4, dtype=torch.int32),
                    None, True,
                ),
            }
        },
 
        # ------------------------------------------------------------------
        # pattern_001
        # Global all — EOS check: some sequences still running
        # ------------------------------------------------------------------
        ("test_torch_all_pattern_001", "_run_all_test"): {
            "param_sets": {
                "eos_partial_eager": (
                    torch.tensor([0, 1, 0, 0], dtype=torch.int32),
                    None, False,
                ),
                "eos_partial_compiled": (
                    torch.tensor([0, 1, 0, 0], dtype=torch.int32),
                    None, True,
                ),
            }
        },
 
        # ------------------------------------------------------------------
        # pattern_002
        # Global all — cache validity: all lengths > 0
        # kv_cache lengths for 8 KV heads
        # ------------------------------------------------------------------
        ("test_torch_all_pattern_002", "_run_all_test"): {
            "param_sets": {
                "cache_valid_eager": (
                    torch.randint(1, 512, (NUM_KV_HEADS,), dtype=torch.int64),
                    None, False,
                ),
                "cache_valid_compiled": (
                    torch.randint(1, 512, (NUM_KV_HEADS,), dtype=torch.int64),
                    None, True,
                ),
            }
        },
 
        # ------------------------------------------------------------------
        # pattern_003
        # Global all — bool tensor (image token mask fully populated)
        # ------------------------------------------------------------------
        ("test_torch_all_pattern_003", "_run_all_test"): {
            "param_sets": {
                "image_mask_full_eager": (
                    torch.ones(64, dtype=torch.bool),
                    None, False,
                ),
                "image_mask_full_compiled": (
                    torch.ones(64, dtype=torch.bool),
                    None, True,
                ),
            }
        },
 
        # ------------------------------------------------------------------
        # pattern_004
        # dim=-1 — attention mask row check: is each row fully unpadded?
        # (1, 128) → (1,)   all-1 mask
        # ------------------------------------------------------------------
        ("test_torch_all_pattern_004", "_run_all_test"): {
            "param_sets": {
                "mask_rowcheck_single_eager": (
                    torch.ones(1, 128, dtype=torch.int32),
                    -1, False,
                ),
                "mask_rowcheck_single_compiled": (
                    torch.ones(1, 128, dtype=torch.int32),
                    -1, True,
                ),
            }
        },
 
        # ------------------------------------------------------------------
        # pattern_005
        # dim=-1 — attention mask row check: batched with some padding
        # (4, 128)  rows: full | padded | full | padded
        # ------------------------------------------------------------------
        ("test_torch_all_pattern_005", "_run_all_test"): {
            "param_sets": {
                "mask_rowcheck_batch_eager": (
                    torch.cat([
                        torch.ones(2, 128, dtype=torch.int32),
                        torch.cat([
                            torch.ones(2, 64, dtype=torch.int32),
                            torch.zeros(2, 64, dtype=torch.int32),
                        ], dim=1),
                    ], dim=0),
                    -1, False,
                ),
                "mask_rowcheck_batch_compiled": (
                    torch.cat([
                        torch.ones(2, 128, dtype=torch.int32),
                        torch.cat([
                            torch.ones(2, 64, dtype=torch.int32),
                            torch.zeros(2, 64, dtype=torch.int32),
                        ], dim=1),
                    ], dim=0),
                    -1, True,
                ),
            }
        },
 
        # ------------------------------------------------------------------
        # pattern_006
        # dim=-1 — image token presence per row (batch=4, 64 slots)
        # ------------------------------------------------------------------
        ("test_torch_all_pattern_006", "_run_all_test"): {
            "param_sets": {
                "image_token_present_eager": (
                    torch.ones(4, 64, dtype=torch.bool),
                    -1, False,
                ),
                "image_token_present_compiled": (
                    torch.ones(4, 64, dtype=torch.bool),
                    -1, True,
                ),
            }
        },
 
        # ------------------------------------------------------------------
        # pattern_007
        # dim=0 — column-wise validity across batch
        # (4, 128) — each column must be non-zero in all batch rows
        # ------------------------------------------------------------------
        ("test_torch_all_pattern_007", "_run_all_test"): {
            "param_sets": {
                "col_valid_eager": (
                    torch.ones(4, 128, dtype=torch.int32),
                    0, False,
                ),
                "col_valid_compiled": (
                    torch.ones(4, 128, dtype=torch.int32),
                    0, True,
                ),
            }
        },
 
        # ------------------------------------------------------------------
        # pattern_008
        # dim=1 — per-head KV validity (all seq positions filled per head)
        # (NUM_KV_HEADS, 128) bool
        # ------------------------------------------------------------------
        ("test_torch_all_pattern_008", "_run_all_test"): {
            "param_sets": {
                "head_kv_valid_eager": (
                    torch.ones(NUM_KV_HEADS, 128, dtype=torch.bool),
                    1, False,
                ),
                "head_kv_valid_compiled": (
                    torch.ones(NUM_KV_HEADS, 128, dtype=torch.bool),
                    1, True,
                ),
            }
        },
 
        # ------------------------------------------------------------------
        # pattern_009
        # dim=-1 — float tensor thresholded to bool, row-wise check
        # (4, 128): value > 0 ↔ token is real
        # ------------------------------------------------------------------
        ("test_torch_all_pattern_009", "_run_all_test"): {
            "param_sets": {
                "float_threshold_eager": (
                    (torch.randn(4, 128) > 0),
                    -1, False,
                ),
                "float_threshold_compiled": (
                    (torch.randn(4, 128) > 0),
                    -1, True,
                ),
            }
        },
 
        # ------------------------------------------------------------------
        # pattern_010
        # Global all — padded mask: some zeros present → all() is False
        # ------------------------------------------------------------------
        ("test_torch_all_pattern_010", "_run_all_test"): {
            "param_sets": {
                "padded_mask_global_eager": (
                    torch.cat([
                        torch.ones(128, dtype=torch.int32),
                        torch.zeros(128, dtype=torch.int32),
                    ]),
                    None, False,
                ),
                "padded_mask_global_compiled": (
                    torch.cat([
                        torch.ones(128, dtype=torch.int32),
                        torch.zeros(128, dtype=torch.int32),
                    ]),
                    None, True,
                ),
            }
        },
 
        # ------------------------------------------------------------------
        # pattern_011
        # Large batch — all sequences finished (batch=8)
        # ------------------------------------------------------------------
        ("test_torch_all_pattern_011", "_run_all_test"): {
            "param_sets": {
                "large_batch_eos_eager": (
                    torch.zeros(8, dtype=torch.int32),
                    None, False,
                ),
                "large_batch_eos_compiled": (
                    torch.zeros(8, dtype=torch.int32),
                    None, True,
                ),
            }
        },
 
        # ------------------------------------------------------------------
        # pattern_012
        # dim=-1 — medium padded batch (batch=2, seq=512, actual=256)
        # ------------------------------------------------------------------
        ("test_torch_all_pattern_012", "_run_all_test"): {
            "param_sets": {
                "medium_padded_rowcheck_eager": (
                    torch.cat([
                        torch.ones(2, 256, dtype=torch.int32),
                        torch.zeros(2, 256, dtype=torch.int32),
                    ], dim=1),
                    -1, False,
                ),
                "medium_padded_rowcheck_compiled": (
                    torch.cat([
                        torch.ones(2, 256, dtype=torch.int32),
                        torch.zeros(2, 256, dtype=torch.int32),
                    ], dim=1),
                    -1, True,
                ),
            }
        },
    }
 
    def _run_all_test(self, tensor: torch.Tensor, dim, compiled: bool):
        """torch.all(tensor) or torch.all(tensor, dim=dim): verify shape then CPU compare."""
 
        def all_fn(t):
            if dim is None:
                return torch.all(t)
            return torch.all(t, dim=dim)
 
        result = all_fn(tensor)
 
        if dim is None:
            assert result.shape == torch.Size([]), (
                f"torch.all() global: expected scalar, got {result.shape}"
            )
        else:
            dim_idx  = dim if dim >= 0 else tensor.dim() + dim
            expected = list(tensor.shape)
            expected.pop(dim_idx)
            assert result.shape == torch.Size(expected), (
                f"torch.all(dim={dim}): got {tuple(result.shape)}, "
                f"expected {tuple(expected)}"
            )
        assert result.dtype == torch.bool, (
            f"torch.all() must return bool, got {result.dtype}"
        )
 
        compare_with_cpu(all_fn, tensor, compiled=compiled)

# ===========================================================================
# TestFloat
# ===========================================================================
 
class TestFloat(unittest.TestCase, metaclass=ParameterizedTestMeta):
    """
    Tests for dtype-cast shorthand methods observed in
    mistralai/Mistral-Small-3.2-24B-Instruct-2506:
      .float()    → torch.float32
      .half()     → torch.float16
      .bfloat16() → torch.bfloat16
 
    Call-sites in the model
    -----------------------
    .float():
      hidden_states.float()   — RMSNorm upcast before norm computation
      logits.float()          — logit upcast before cross-entropy loss
 
    .half():
      weight.half()           — fp16 weight loading
      attention_mask.half()   — mask dtype for additive attention
 
    .bfloat16():
      output.bfloat16()       — downcast after float32 accumulation
      kv_cache.bfloat16()     — memory-efficient KV storage
 
    Pattern layout
    --------------
      pattern_000–003 : .float()    on activation & weight tensors
      pattern_004–006 : .half()     on activation & weight tensors
      pattern_007–009 : .bfloat16() on activation & KV tensors
      pattern_010–012 : 1-D norm weights
      pattern_013–015 : logits / padded / large-batch
    """
 
    pytestmark = pytest.mark.torch_float
 
    torch.manual_seed(0)
 
    PARAMS = {
        # ------------------------------------------------------------------
        # pattern_000
        # hidden_states.float() — RMSNorm upcast (batch=1, seq=128)
        # (1, 128, 5120) fp16 → fp32
        # ------------------------------------------------------------------
        ("test_torch_float_pattern_000", "_run_float_test"): {
            "param_sets": {
                "fp16_to_float_eager": (
                    torch.randn(1, 128, HIDDEN_SIZE, dtype=torch.float16),
                    "float", False,
                ),
                "fp16_to_float_compiled": (
                    torch.randn(1, 128, HIDDEN_SIZE, dtype=torch.float16),
                    "float", True,
                ),
            }
        },
 
        # ------------------------------------------------------------------
        # pattern_001
        # hidden_states.float() — medium prompt (seq=512)
        # ------------------------------------------------------------------
        ("test_torch_float_pattern_001", "_run_float_test"): {
            "param_sets": {
                "fp16_to_float_eager": (
                    torch.randn(1, 512, HIDDEN_SIZE, dtype=torch.float16),
                    "float", False,
                ),
                "fp16_to_float_compiled": (
                    torch.randn(1, 512, HIDDEN_SIZE, dtype=torch.float16),
                    "float", True,
                ),
            }
        },
 
        # ------------------------------------------------------------------
        # pattern_002
        # hidden_states.float() — batched (batch=4, seq=128)
        # ------------------------------------------------------------------
        ("test_torch_float_pattern_002", "_run_float_test"): {
            "param_sets": {
                "fp16_to_float_eager": (
                    torch.randn(4, 128, HIDDEN_SIZE, dtype=torch.float16),
                    "float", False,
                ),
                "fp16_to_float_compiled": (
                    torch.randn(4, 128, HIDDEN_SIZE, dtype=torch.float16),
                    "float", True,
                ),
            }
        },
 
        # ------------------------------------------------------------------
        # pattern_003
        # output.float() — bfloat16 → float32 upcast (batch=2, seq=256)
        # ------------------------------------------------------------------
        ("test_torch_float_pattern_003", "_run_float_test"): {
            "param_sets": {
                "bf16_to_float_eager": (
                    torch.randn(2, 256, HIDDEN_SIZE, dtype=torch.bfloat16),
                    "float", False,
                ),
                "bf16_to_float_compiled": (
                    torch.randn(2, 256, HIDDEN_SIZE, dtype=torch.bfloat16),
                    "float", True,
                ),
            }
        },
 
        # ------------------------------------------------------------------
        # pattern_004
        # attention_weights.half() — single batch
        # (1, NUM_Q_HEADS, 128, 128) fp32 → fp16
        # ------------------------------------------------------------------
        ("test_torch_float_pattern_004", "_run_float_test"): {
            "param_sets": {
                "fp32_to_half_eager": (
                    torch.randn(1, NUM_Q_HEADS, 128, HEAD_DIM,
                                dtype=torch.float32),
                    "half", False,
                ),
                "fp32_to_half_compiled": (
                    torch.randn(1, NUM_Q_HEADS, 128, HEAD_DIM,
                                dtype=torch.float32),
                    "half", True,
                ),
            }
        },
 
        # ------------------------------------------------------------------
        # pattern_005
        # Q/K/V weight.half() — weight loading
        # (NUM_Q_HEADS*HEAD_DIM, HIDDEN_SIZE) = (4096, 5120)
        # ------------------------------------------------------------------
        ("test_torch_float_pattern_005", "_run_float_test"): {
            "param_sets": {
                "qkv_fp32_to_half_eager": (
                    torch.randn(NUM_Q_HEADS * HEAD_DIM, HIDDEN_SIZE,
                                dtype=torch.float32),
                    "half", False,
                ),
                "qkv_fp32_to_half_compiled": (
                    torch.randn(NUM_Q_HEADS * HEAD_DIM, HIDDEN_SIZE,
                                dtype=torch.float32),
                    "half", True,
                ),
            }
        },
 
        # ------------------------------------------------------------------
        # pattern_006
        # kv_cache.bfloat16() — GQA KV storage
        # (1, NUM_KV_HEADS, 128, 128) fp32 → bf16
        # ------------------------------------------------------------------
        ("test_torch_float_pattern_006", "_run_float_test"): {
            "param_sets": {
                "fp32_to_bf16_eager": (
                    torch.randn(1, NUM_KV_HEADS, 128, HEAD_DIM,
                                dtype=torch.float32),
                    "bfloat16", False,
                ),
                "fp32_to_bf16_compiled": (
                    torch.randn(1, NUM_KV_HEADS, 128, HEAD_DIM,
                                dtype=torch.float32),
                    "bfloat16", True,
                ),
            }
        },
 
        # ------------------------------------------------------------------
        # pattern_007
        # output.bfloat16() — mixed-precision downcast (batch=4, seq=128)
        # ------------------------------------------------------------------
        ("test_torch_float_pattern_007", "_run_float_test"): {
            "param_sets": {
                "fp32_to_bf16_eager": (
                    torch.randn(4, 128, HIDDEN_SIZE, dtype=torch.float32),
                    "bfloat16", False,
                ),
                "fp32_to_bf16_compiled": (
                    torch.randn(4, 128, HIDDEN_SIZE, dtype=torch.float32),
                    "bfloat16", True,
                ),
            }
        },
 
        # ------------------------------------------------------------------
        # pattern_008
        # gate_proj.weight.bfloat16()
        # (INTERMEDIATE_SIZE, HIDDEN_SIZE) = (32768, 5120) fp32 → bf16
        # ------------------------------------------------------------------
        ("test_torch_float_pattern_008", "_run_float_test"): {
            "param_sets": {
                "gate_fp32_to_bf16_eager": (
                    torch.randn(INTERMEDIATE_SIZE, HIDDEN_SIZE,
                                dtype=torch.float32),
                    "bfloat16", False,
                ),
                "gate_fp32_to_bf16_compiled": (
                    torch.randn(INTERMEDIATE_SIZE, HIDDEN_SIZE,
                                dtype=torch.float32),
                    "bfloat16", True,
                ),
            }
        },
 
        # ------------------------------------------------------------------
        # pattern_009
        # o_proj.weight.bfloat16()
        # (HIDDEN_SIZE, NUM_Q_HEADS*HEAD_DIM) = (5120, 4096) fp32 → bf16
        # ------------------------------------------------------------------
        ("test_torch_float_pattern_009", "_run_float_test"): {
            "param_sets": {
                "o_proj_fp32_to_bf16_eager": (
                    torch.randn(HIDDEN_SIZE, NUM_Q_HEADS * HEAD_DIM,
                                dtype=torch.float32),
                    "bfloat16", False,
                ),
                "o_proj_fp32_to_bf16_compiled": (
                    torch.randn(HIDDEN_SIZE, NUM_Q_HEADS * HEAD_DIM,
                                dtype=torch.float32),
                    "bfloat16", True,
                ),
            }
        },
 
        # ------------------------------------------------------------------
        # pattern_010
        # RMSNorm weight.float() — 1-D norm weight upcast (HIDDEN_SIZE,)
        # ------------------------------------------------------------------
        ("test_torch_float_pattern_010", "_run_float_test"): {
            "param_sets": {
                "norm_weight_fp16_to_float_eager": (
                    torch.randn(HIDDEN_SIZE, dtype=torch.float16),
                    "float", False,
                ),
                "norm_weight_fp16_to_float_compiled": (
                    torch.randn(HIDDEN_SIZE, dtype=torch.float16),
                    "float", True,
                ),
            }
        },
 
        # ------------------------------------------------------------------
        # pattern_011
        # Layer norm weight.half() — 1-D cast to fp16
        # ------------------------------------------------------------------
        ("test_torch_float_pattern_011", "_run_float_test"): {
            "param_sets": {
                "norm_weight_fp32_to_half_eager": (
                    torch.randn(HIDDEN_SIZE, dtype=torch.float32),
                    "half", False,
                ),
                "norm_weight_fp32_to_half_compiled": (
                    torch.randn(HIDDEN_SIZE, dtype=torch.float32),
                    "half", True,
                ),
            }
        },
 
        # ------------------------------------------------------------------
        # pattern_012
        # RMSNorm weight.bfloat16() — 1-D cast for bf16 path
        # ------------------------------------------------------------------
        ("test_torch_float_pattern_012", "_run_float_test"): {
            "param_sets": {
                "norm_weight_fp32_to_bf16_eager": (
                    torch.randn(HIDDEN_SIZE, dtype=torch.float32),
                    "bfloat16", False,
                ),
                "norm_weight_fp32_to_bf16_compiled": (
                    torch.randn(HIDDEN_SIZE, dtype=torch.float32),
                    "bfloat16", True,
                ),
            }
        },
 
        # ------------------------------------------------------------------
        # pattern_013
        # logits.float() — before cross-entropy (batch=1, seq=128, vocab=32)
        # ------------------------------------------------------------------
        ("test_torch_float_pattern_013", "_run_float_test"): {
            "param_sets": {
                "logits_fp16_to_float_eager": (
                    torch.randn(1, 128, 32, dtype=torch.float16),
                    "float", False,
                ),
                "logits_fp16_to_float_compiled": (
                    torch.randn(1, 128, 32, dtype=torch.float16),
                    "float", True,
                ),
            }
        },
 
        # ------------------------------------------------------------------
        # pattern_014
        # logits.float() — batched (batch=4)
        # ------------------------------------------------------------------
        ("test_torch_float_pattern_014", "_run_float_test"): {
            "param_sets": {
                "logits_batch_fp16_to_float_eager": (
                    torch.randn(4, 128, 32, dtype=torch.float16),
                    "float", False,
                ),
                "logits_batch_fp16_to_float_compiled": (
                    torch.randn(4, 128, 32, dtype=torch.float16),
                    "float", True,
                ),
            }
        },
 
        # ------------------------------------------------------------------
        # pattern_015
        # Padded activation .float() (batch=1, actual=128, padded to 256)
        # ------------------------------------------------------------------
        ("test_torch_float_pattern_015", "_run_float_test"): {
            "param_sets": {
                "padded_fp16_to_float_eager": (
                    torch.cat([
                        torch.randn(1, 128, HIDDEN_SIZE, dtype=torch.float16),
                        torch.zeros(1, 128, HIDDEN_SIZE, dtype=torch.float16),
                    ], dim=1),
                    "float", False,
                ),
                "padded_fp16_to_float_compiled": (
                    torch.cat([
                        torch.randn(1, 128, HIDDEN_SIZE, dtype=torch.float16),
                        torch.zeros(1, 128, HIDDEN_SIZE, dtype=torch.float16),
                    ], dim=1),
                    "float", True,
                ),
            }
        },
    }
 
    def _run_float_test(self, tensor: torch.Tensor, cast: str, compiled: bool):
        """tensor.{float,half,bfloat16}(): verify dtype & shape, then CPU compare."""
        dtype_map = {
            "float":    torch.float32,
            "half":     torch.float16,
            "bfloat16": torch.bfloat16,
        }
        expected_dtype = dtype_map[cast]
 
        def cast_fn(t):
            return getattr(t, cast)()
 
        result = cast_fn(tensor)
        assert result.shape == tensor.shape, (
            f".{cast}() shape changed: {tensor.shape} → {result.shape}"
        )
        assert result.dtype == expected_dtype, (
            f".{cast}() dtype: got {result.dtype}, expected {expected_dtype}"
        )
 
        compare_with_cpu(cast_fn, tensor, compiled=compiled)


# ─────────────────────────────────────────────────────────────────────────────
# TestTranspose
# ─────────────────────────────────────────────────────────────────────────────

class TestTranspose(unittest.TestCase, metaclass=ParameterizedTestMeta):
    """
    Tests for torch.transpose for the Mistral model.

    Input shapes and dtypes:
        float32  : [1, 64, 1],  [1, 64, 38]
        bfloat16 : [1, 38, 32, 128], [1, 38,  8, 128]
                   [1, 32, 38, 128], [1,  1, 32, 128]
                   [1,  1,  8, 128], [1, 32,  1, 128]

    Sub-group marks (auto-derived by ParameterizedTestMeta):
        _run_transpose_shape_test       → @pytest.mark.torch_transpose_shape
        _run_transpose_values_test      → @pytest.mark.torch_transpose_values
        _run_transpose_neg_dims_test    → @pytest.mark.torch_transpose_neg_dims
        _run_transpose_contiguity_test  → @pytest.mark.torch_transpose_contiguity
        _run_transpose_contig_copy_test → @pytest.mark.torch_transpose_contig_copy
        _run_transpose_dtype_test       → @pytest.mark.torch_transpose_dtype
    """

    pytestmark = pytest.mark.torch_transpose

    torch.manual_seed(0)

    PARAMS = {

        # ══════════════════════════════════════════════════════════════════
        # SHAPE CORRECTNESS
        # ══════════════════════════════════════════════════════════════════

        # 3-D [1, 64, 1]  float32 ─────────────────────────────────────────
        # (0,1): [1,64,1] → [64,1,1]
        ("test_torch_transpose_shape_pattern_000", "_run_transpose_shape_test"): {
            "param_sets": {
                "s_1x64x1_d01_eager":    (0, 1, torch.randn(1, 64, 1, dtype=F32), False),
                "s_1x64x1_d01_compiled": (0, 1, torch.randn(1, 64, 1, dtype=F32), True),
            },
        },
        # (1,2): [1,64,1] → [1,1,64]
        ("test_torch_transpose_shape_pattern_001", "_run_transpose_shape_test"): {
            "param_sets": {
                "s_1x64x1_d12_eager":    (1, 2, torch.randn(1, 64, 1, dtype=F32), False),
                "s_1x64x1_d12_compiled": (1, 2, torch.randn(1, 64, 1, dtype=F32), True),
            },
        },

        # 3-D [1, 64, 38]  float32 ────────────────────────────────────────
        # (0,1): [1,64,38] → [64,1,38]
        ("test_torch_transpose_shape_pattern_002", "_run_transpose_shape_test"): {
            "param_sets": {
                "s_1x64x38_d01_eager":    (0, 1, torch.randn(1, 64, 38, dtype=F32), False),
                "s_1x64x38_d01_compiled": (0, 1, torch.randn(1, 64, 38, dtype=F32), True),
            },
        },
        # (1,2): [1,64,38] → [1,38,64]
        ("test_torch_transpose_shape_pattern_003", "_run_transpose_shape_test"): {
            "param_sets": {
                "s_1x64x38_d12_eager":    (1, 2, torch.randn(1, 64, 38, dtype=F32), False),
                "s_1x64x38_d12_compiled": (1, 2, torch.randn(1, 64, 38, dtype=F32), True),
            },
        },

        # 4-D [1, 38, 32, 128]  bfloat16 ──────────────────────────────────
        # (1,2): [1,38,32,128] → [1,32,38,128]
        ("test_torch_transpose_shape_pattern_004", "_run_transpose_shape_test"): {
            "param_sets": {
                "s_1x38x32x128_d12_eager":    (1, 2, torch.randn(1, 38, 32, 128, dtype=BF16), False),
                "s_1x38x32x128_d12_compiled": (1, 2, torch.randn(1, 38, 32, 128, dtype=BF16), True),
            },
        },
        # (2,3): [1,38,32,128] → [1,38,128,32]
        ("test_torch_transpose_shape_pattern_005", "_run_transpose_shape_test"): {
            "param_sets": {
                "s_1x38x32x128_d23_eager":    (2, 3, torch.randn(1, 38, 32, 128, dtype=BF16), False),
                "s_1x38x32x128_d23_compiled": (2, 3, torch.randn(1, 38, 32, 128, dtype=BF16), True),
            },
        },

        # 4-D [1, 38, 8, 128]  bfloat16 ───────────────────────────────────
        # (1,2): [1,38,8,128] → [1,8,38,128]
        ("test_torch_transpose_shape_pattern_006", "_run_transpose_shape_test"): {
            "param_sets": {
                "s_1x38x8x128_d12_eager":    (1, 2, torch.randn(1, 38, 8, 128, dtype=BF16), False),
                "s_1x38x8x128_d12_compiled": (1, 2, torch.randn(1, 38, 8, 128, dtype=BF16), True),
            },
        },
        # (2,3): [1,38,8,128] → [1,38,128,8]
        ("test_torch_transpose_shape_pattern_007", "_run_transpose_shape_test"): {
            "param_sets": {
                "s_1x38x8x128_d23_eager":    (2, 3, torch.randn(1, 38, 8, 128, dtype=BF16), False),
                "s_1x38x8x128_d23_compiled": (2, 3, torch.randn(1, 38, 8, 128, dtype=BF16), True),
            },
        },

        # 4-D [1, 32, 38, 128]  bfloat16 ──────────────────────────────────
        # (1,2): [1,32,38,128] → [1,38,32,128]
        ("test_torch_transpose_shape_pattern_008", "_run_transpose_shape_test"): {
            "param_sets": {
                "s_1x32x38x128_d12_eager":    (1, 2, torch.randn(1, 32, 38, 128, dtype=BF16), False),
                "s_1x32x38x128_d12_compiled": (1, 2, torch.randn(1, 32, 38, 128, dtype=BF16), True),
            },
        },
        # (2,3): [1,32,38,128] → [1,32,128,38]
        ("test_torch_transpose_shape_pattern_009", "_run_transpose_shape_test"): {
            "param_sets": {
                "s_1x32x38x128_d23_eager":    (2, 3, torch.randn(1, 32, 38, 128, dtype=BF16), False),
                "s_1x32x38x128_d23_compiled": (2, 3, torch.randn(1, 32, 38, 128, dtype=BF16), True),
            },
        },

        # 4-D [1, 1, 32, 128]  bfloat16 ───────────────────────────────────
        # (1,3): [1,1,32,128] → [1,128,32,1]
        ("test_torch_transpose_shape_pattern_010", "_run_transpose_shape_test"): {
            "param_sets": {
                "s_1x1x32x128_d13_eager":    (1, 3, torch.randn(1, 1, 32, 128, dtype=BF16), False),
                "s_1x1x32x128_d13_compiled": (1, 3, torch.randn(1, 1, 32, 128, dtype=BF16), True),
            },
        },
        # (2,3): [1,1,32,128] → [1,1,128,32]
        ("test_torch_transpose_shape_pattern_011", "_run_transpose_shape_test"): {
            "param_sets": {
                "s_1x1x32x128_d23_eager":    (2, 3, torch.randn(1, 1, 32, 128, dtype=BF16), False),
                "s_1x1x32x128_d23_compiled": (2, 3, torch.randn(1, 1, 32, 128, dtype=BF16), True),
            },
        },

        # 4-D [1, 1, 8, 128]  bfloat16 ────────────────────────────────────
        # (1,3): [1,1,8,128] → [1,128,8,1]
        ("test_torch_transpose_shape_pattern_012", "_run_transpose_shape_test"): {
            "param_sets": {
                "s_1x1x8x128_d13_eager":    (1, 3, torch.randn(1, 1, 8, 128, dtype=BF16), False),
                "s_1x1x8x128_d13_compiled": (1, 3, torch.randn(1, 1, 8, 128, dtype=BF16), True),
            },
        },
        # (2,3): [1,1,8,128] → [1,1,128,8]
        ("test_torch_transpose_shape_pattern_013", "_run_transpose_shape_test"): {
            "param_sets": {
                "s_1x1x8x128_d23_eager":    (2, 3, torch.randn(1, 1, 8, 128, dtype=BF16), False),
                "s_1x1x8x128_d23_compiled": (2, 3, torch.randn(1, 1, 8, 128, dtype=BF16), True),
            },
        },

        # 4-D [1, 32, 1, 128]  bfloat16 ───────────────────────────────────
        # (1,3): [1,32,1,128] → [1,128,1,32]
        ("test_torch_transpose_shape_pattern_014", "_run_transpose_shape_test"): {
            "param_sets": {
                "s_1x32x1x128_d13_eager":    (1, 3, torch.randn(1, 32, 1, 128, dtype=BF16), False),
                "s_1x32x1x128_d13_compiled": (1, 3, torch.randn(1, 32, 1, 128, dtype=BF16), True),
            },
        },
        # (2,3): [1,32,1,128] → [1,32,128,1]
        ("test_torch_transpose_shape_pattern_015", "_run_transpose_shape_test"): {
            "param_sets": {
                "s_1x32x1x128_d23_eager":    (2, 3, torch.randn(1, 32, 1, 128, dtype=BF16), False),
                "s_1x32x1x128_d23_compiled": (2, 3, torch.randn(1, 32, 1, 128, dtype=BF16), True),
            },
        },


        # ══════════════════════════════════════════════════════════════════
        # VALUE CORRECTNESS
        # ══════════════════════════════════════════════════════════════════

        # [1,64,38] float32  (1,2): t[b,r,c] == result[b,c,r]
        ("test_torch_transpose_values_pattern_000", "_run_transpose_values_test"): {
            "param_sets": {
                "v_1x64x38_d12_eager":    (1, 2, torch.randn(1, 64, 38, dtype=F32), False),
                "v_1x64x38_d12_compiled": (1, 2, torch.randn(1, 64, 38, dtype=F32), True),
            },
        },
        # [1,38,32,128] bfloat16  (2,3): t[b,h,s,d] == result[b,h,d,s]
        ("test_torch_transpose_values_pattern_001", "_run_transpose_values_test"): {
            "param_sets": {
                "v_1x38x32x128_d23_eager":    (2, 3, torch.randn(1, 38, 32, 128, dtype=BF16), False),
                "v_1x38x32x128_d23_compiled": (2, 3, torch.randn(1, 38, 32, 128, dtype=BF16), True),
            },
        },
        # [1,38,8,128] bfloat16  (1,3): t[b,h,s,d] == result[b,d,s,h]
        ("test_torch_transpose_values_pattern_002", "_run_transpose_values_test"): {
            "param_sets": {
                "v_1x38x8x128_d13_eager":    (1, 3, torch.randn(1, 38, 8, 128, dtype=BF16), False),
                "v_1x38x8x128_d13_compiled": (1, 3, torch.randn(1, 38, 8, 128, dtype=BF16), True),
            },
        },
        # [1,32,38,128] bfloat16  (1,2)
        ("test_torch_transpose_values_pattern_003", "_run_transpose_values_test"): {
            "param_sets": {
                "v_1x32x38x128_d12_eager":    (1, 2, torch.randn(1, 32, 38, 128, dtype=BF16), False),
                "v_1x32x38x128_d12_compiled": (1, 2, torch.randn(1, 32, 38, 128, dtype=BF16), True),
            },
        },
        # [1,1,32,128] bfloat16  (2,3)
        ("test_torch_transpose_values_pattern_004", "_run_transpose_values_test"): {
            "param_sets": {
                "v_1x1x32x128_d23_eager":    (2, 3, torch.randn(1, 1, 32, 128, dtype=BF16), False),
                "v_1x1x32x128_d23_compiled": (2, 3, torch.randn(1, 1, 32, 128, dtype=BF16), True),
            },
        },
        # [1,1,8,128] bfloat16  (2,3)
        ("test_torch_transpose_values_pattern_005", "_run_transpose_values_test"): {
            "param_sets": {
                "v_1x1x8x128_d23_eager":    (2, 3, torch.randn(1, 1, 8, 128, dtype=BF16), False),
                "v_1x1x8x128_d23_compiled": (2, 3, torch.randn(1, 1, 8, 128, dtype=BF16), True),
            },
        },
        # [1,32,1,128] bfloat16  (1,3)
        ("test_torch_transpose_values_pattern_006", "_run_transpose_values_test"): {
            "param_sets": {
                "v_1x32x1x128_d13_eager":    (1, 3, torch.randn(1, 32, 1, 128, dtype=BF16), False),
                "v_1x32x1x128_d13_compiled": (1, 3, torch.randn(1, 32, 1, 128, dtype=BF16), True),
            },
        },
        # [1,32,38,128] bfloat16  (2,3)
        ("test_torch_transpose_values_pattern_007", "_run_transpose_values_test"): {
            "param_sets": {
                "v_1x32x38x128_d23_eager":    (2, 3, torch.randn(1, 32, 38, 128, dtype=BF16), False),
                "v_1x32x38x128_d23_compiled": (2, 3, torch.randn(1, 32, 38, 128, dtype=BF16), True),
            },
        },


        # ══════════════════════════════════════════════════════════════════
        # NEGATIVE DIMENSION INDEXING
        # ══════════════════════════════════════════════════════════════════

        # [1,64,38] float32  (-2,-1) == (1,2)
        ("test_torch_transpose_neg_dims_pattern_000", "_run_transpose_neg_dims_test"): {
            "param_sets": {
                "neg_1x64x38_m2m1_vs_12_eager":    (-2, -1, 1, 2, torch.randn(1, 64, 38, dtype=F32), False),
                "neg_1x64x38_m2m1_vs_12_compiled": (-2, -1, 1, 2, torch.randn(1, 64, 38, dtype=F32), True),
            },
        },
        # [1,64,1] float32  (-3,-2) == (0,1)
        ("test_torch_transpose_neg_dims_pattern_001", "_run_transpose_neg_dims_test"): {
            "param_sets": {
                "neg_1x64x1_m3m2_vs_01_eager":    (-3, -2, 0, 1, torch.randn(1, 64, 1, dtype=F32), False),
                "neg_1x64x1_m3m2_vs_01_compiled": (-3, -2, 0, 1, torch.randn(1, 64, 1, dtype=F32), True),
            },
        },
        # [1,38,32,128] bfloat16  (-2,-1) == (2,3)
        ("test_torch_transpose_neg_dims_pattern_002", "_run_transpose_neg_dims_test"): {
            "param_sets": {
                "neg_1x38x32x128_m2m1_vs_23_eager":    (-2, -1, 2, 3, torch.randn(1, 38, 32, 128, dtype=BF16), False),
                "neg_1x38x32x128_m2m1_vs_23_compiled": (-2, -1, 2, 3, torch.randn(1, 38, 32, 128, dtype=BF16), True),
            },
        },
        # [1,38,32,128] bfloat16  (-3,-1) == (1,3)
        ("test_torch_transpose_neg_dims_pattern_003", "_run_transpose_neg_dims_test"): {
            "param_sets": {
                "neg_1x38x32x128_m3m1_vs_13_eager":    (-3, -1, 1, 3, torch.randn(1, 38, 32, 128, dtype=BF16), False),
                "neg_1x38x32x128_m3m1_vs_13_compiled": (-3, -1, 1, 3, torch.randn(1, 38, 32, 128, dtype=BF16), True),
            },
        },
        # [1,38,8,128] bfloat16  (-2,-1) == (2,3)
        ("test_torch_transpose_neg_dims_pattern_004", "_run_transpose_neg_dims_test"): {
            "param_sets": {
                "neg_1x38x8x128_m2m1_vs_23_eager":    (-2, -1, 2, 3, torch.randn(1, 38, 8, 128, dtype=BF16), False),
                "neg_1x38x8x128_m2m1_vs_23_compiled": (-2, -1, 2, 3, torch.randn(1, 38, 8, 128, dtype=BF16), True),
            },
        },
        # [1,32,38,128] bfloat16  (-3,-2) == (1,2)
        ("test_torch_transpose_neg_dims_pattern_005", "_run_transpose_neg_dims_test"): {
            "param_sets": {
                "neg_1x32x38x128_m3m2_vs_12_eager":    (-3, -2, 1, 2, torch.randn(1, 32, 38, 128, dtype=BF16), False),
                "neg_1x32x38x128_m3m2_vs_12_compiled": (-3, -2, 1, 2, torch.randn(1, 32, 38, 128, dtype=BF16), True),
            },
        },
        # [1,1,32,128] bfloat16  (-2,-1) == (2,3)
        ("test_torch_transpose_neg_dims_pattern_006", "_run_transpose_neg_dims_test"): {
            "param_sets": {
                "neg_1x1x32x128_m2m1_vs_23_eager":    (-2, -1, 2, 3, torch.randn(1, 1, 32, 128, dtype=BF16), False),
                "neg_1x1x32x128_m2m1_vs_23_compiled": (-2, -1, 2, 3, torch.randn(1, 1, 32, 128, dtype=BF16), True),
            },
        },
        # [1,1,8,128] bfloat16  (-2,-1) == (2,3)
        ("test_torch_transpose_neg_dims_pattern_007", "_run_transpose_neg_dims_test"): {
            "param_sets": {
                "neg_1x1x8x128_m2m1_vs_23_eager":    (-2, -1, 2, 3, torch.randn(1, 1, 8, 128, dtype=BF16), False),
                "neg_1x1x8x128_m2m1_vs_23_compiled": (-2, -1, 2, 3, torch.randn(1, 1, 8, 128, dtype=BF16), True),
            },
        },
        # [1,32,1,128] bfloat16  (-3,-1) == (1,3)
        ("test_torch_transpose_neg_dims_pattern_008", "_run_transpose_neg_dims_test"): {
            "param_sets": {
                "neg_1x32x1x128_m3m1_vs_13_eager":    (-3, -1, 1, 3, torch.randn(1, 32, 1, 128, dtype=BF16), False),
                "neg_1x32x1x128_m3m1_vs_13_compiled": (-3, -1, 1, 3, torch.randn(1, 32, 1, 128, dtype=BF16), True),
            },
        },
        # [1,32,1,128] bfloat16  (-4,-1) == (0,3)
        ("test_torch_transpose_neg_dims_pattern_009", "_run_transpose_neg_dims_test"): {
            "param_sets": {
                "neg_1x32x1x128_m4m1_vs_03_eager":    (-4, -1, 0, 3, torch.randn(1, 32, 1, 128, dtype=BF16), False),
                "neg_1x32x1x128_m4m1_vs_03_compiled": (-4, -1, 0, 3, torch.randn(1, 32, 1, 128, dtype=BF16), True),
            },
        },

        # ══════════════════════════════════════════════════════════════════
        # DTYPE PRESERVATION — dtype must not change after transpose
        # ══════════════════════════════════════════════════════════════════

        # [1,64,1] float32  (0,1)
        ("test_torch_transpose_dtype_pattern_000", "_run_transpose_dtype_test"): {
            "param_sets": {
                "dtype_1x64x1_f32_d01_eager":    (0, 1, torch.randn(1, 64, 1, dtype=F32), False),
                "dtype_1x64x1_f32_d01_compiled": (0, 1, torch.randn(1, 64, 1, dtype=F32), True),
            },
        },
        # [1,64,38] float32  (1,2)
        ("test_torch_transpose_dtype_pattern_001", "_run_transpose_dtype_test"): {
            "param_sets": {
                "dtype_1x64x38_f32_d12_eager":    (1, 2, torch.randn(1, 64, 38, dtype=F32), False),
                "dtype_1x64x38_f32_d12_compiled": (1, 2, torch.randn(1, 64, 38, dtype=F32), True),
            },
        },
        # [1,38,32,128] bfloat16  (2,3)
        ("test_torch_transpose_dtype_pattern_002", "_run_transpose_dtype_test"): {
            "param_sets": {
                "dtype_1x38x32x128_bf16_d23_eager":    (2, 3, torch.randn(1, 38, 32, 128, dtype=BF16), False),
                "dtype_1x38x32x128_bf16_d23_compiled": (2, 3, torch.randn(1, 38, 32, 128, dtype=BF16), True),
            },
        },
        # [1,38,8,128] bfloat16  (2,3)
        ("test_torch_transpose_dtype_pattern_003", "_run_transpose_dtype_test"): {
            "param_sets": {
                "dtype_1x38x8x128_bf16_d23_eager":    (2, 3, torch.randn(1, 38, 8, 128, dtype=BF16), False),
                "dtype_1x38x8x128_bf16_d23_compiled": (2, 3, torch.randn(1, 38, 8, 128, dtype=BF16), True),
            },
        },
        # [1,32,38,128] bfloat16  (1,2)
        ("test_torch_transpose_dtype_pattern_004", "_run_transpose_dtype_test"): {
            "param_sets": {
                "dtype_1x32x38x128_bf16_d12_eager":    (1, 2, torch.randn(1, 32, 38, 128, dtype=BF16), False),
                "dtype_1x32x38x128_bf16_d12_compiled": (1, 2, torch.randn(1, 32, 38, 128, dtype=BF16), True),
            },
        },
        # [1,1,32,128] bfloat16  (2,3)
        ("test_torch_transpose_dtype_pattern_005", "_run_transpose_dtype_test"): {
            "param_sets": {
                "dtype_1x1x32x128_bf16_d23_eager":    (2, 3, torch.randn(1, 1, 32, 128, dtype=BF16), False),
                "dtype_1x1x32x128_bf16_d23_compiled": (2, 3, torch.randn(1, 1, 32, 128, dtype=BF16), True),
            },
        },
        # [1,1,8,128] bfloat16  (2,3)
        ("test_torch_transpose_dtype_pattern_006", "_run_transpose_dtype_test"): {
            "param_sets": {
                "dtype_1x1x8x128_bf16_d23_eager":    (2, 3, torch.randn(1, 1, 8, 128, dtype=BF16), False),
                "dtype_1x1x8x128_bf16_d23_compiled": (2, 3, torch.randn(1, 1, 8, 128, dtype=BF16), True),
            },
        },
        # [1,32,1,128] bfloat16  (1,3)
        ("test_torch_transpose_dtype_pattern_007", "_run_transpose_dtype_test"): {
            "param_sets": {
                "dtype_1x32x1x128_bf16_d13_eager":    (1, 3, torch.randn(1, 32, 1, 128, dtype=BF16), False),
                "dtype_1x32x1x128_bf16_d13_compiled": (1, 3, torch.randn(1, 32, 1, 128, dtype=BF16), True),
            },
        },
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # ── Base test methods ──────────────────────────────────────────────────

    def _run_transpose_shape_test(self, dim0, dim1, x, compiled):
        expected = list(x.shape)
        d0, d1 = dim0 % x.ndim, dim1 % x.ndim
        expected[d0], expected[d1] = expected[d1], expected[d0]

        def fn(t):
            out = torch.transpose(t, dim0, dim1).contiguous()
            assert list(out.shape) == expected, (
                f"Shape mismatch: expected {expected}, got {list(out.shape)}"
            )
            return out

        compare_with_cpu(fn, x, compiled=compiled)

    def _run_transpose_values_test(self, dim0, dim1, x, compiled):
        compare_with_cpu(
            lambda t: torch.transpose(t, dim0, dim1).contiguous(),
            x,
            compiled=compiled,
        )

    def _run_transpose_neg_dims_test(self, neg0, neg1, pos0, pos1, x, compiled):
        def fn(t):
            neg_result = torch.transpose(t, neg0, neg1).contiguous()
            pos_result = torch.transpose(t, pos0, pos1).contiguous()
            torch.testing.assert_close(
                neg_result, pos_result,
                msg=f"transpose({neg0},{neg1}) differs from transpose({pos0},{pos1})",
            )
            return neg_result

        compare_with_cpu(fn, x, compiled=compiled)

    def _run_transpose_dtype_test(self, dim0, dim1, x, compiled):
        def fn(t):
            result = torch.transpose(t, dim0, dim1).contiguous()
            assert result.dtype == t.dtype, (
                f"dtype changed after transpose({dim0},{dim1}): "
                f"expected {t.dtype}, got {result.dtype}"
            )
            return result

        compare_with_cpu(fn, x, compiled=compiled)


# ─────────────────────────────────────────────────────────────────────────────
# TestCos
# ─────────────────────────────────────────────────────────────────────────────

class TestCos(unittest.TestCase, metaclass=ParameterizedTestMeta):
    """
    Tests for torch.cos for the Mistral model.

    Op specification:
        op    : torch.cos
        dtype : torch.float32

    Input shapes:
        [1, 38, 128]   (prefill — 38 tokens, head_dim=128)
        [1,  1, 128]   (decode  — single token, head_dim=128)

    Note: cos is a unary pointwise op — no neg_dims or contiguity sections apply.

    Sub-group marks (auto-derived by ParameterizedTestMeta):
        _run_cos_shape_test  → @pytest.mark.torch_cos_shape
        _run_cos_values_test → @pytest.mark.torch_cos_values
        _run_cos_dtype_test  → @pytest.mark.torch_cos_dtype
    """

    pytestmark = pytest.mark.torch_cos

    torch.manual_seed(0)

    # Positive-valued tensors shared across eager/compiled pairs within each pattern.
    _x38     = torch.abs(torch.randn(1, 38, 128, dtype=F32)) + 1e-4
    _x38_one = torch.ones(1, 38, 128, dtype=F32)
    _x38_pi  = torch.full((1, 38, 128), math.pi / 2, dtype=F32)
    _x1      = torch.abs(torch.randn(1, 1,  128, dtype=F32)) + 1e-4
    _x1_one  = torch.ones(1, 1,  128, dtype=F32)
    _x1_pi   = torch.full((1, 1,  128), math.pi / 2, dtype=F32)

    PARAMS = {

        # ══════════════════════════════════════════════════════════════════
        # SHAPE CORRECTNESS — cos is pointwise: output shape == input shape
        # ══════════════════════════════════════════════════════════════════

        # [1, 38, 128]
        ("test_torch_cos_shape_pattern_000", "_run_cos_shape_test"): {
            "param_sets": {
                "s_1x38x128_eager":    (_x38, False),
                "s_1x38x128_compiled": (_x38, True),
            },
        },
        # [1, 1, 128]
        ("test_torch_cos_shape_pattern_001", "_run_cos_shape_test"): {
            "param_sets": {
                "s_1x1x128_eager":    (_x1, False),
                "s_1x1x128_compiled": (_x1, True),
            },
        },


        # ══════════════════════════════════════════════════════════════════
        # VALUE CORRECTNESS — cos(x) CPU vs Spyre element-wise
        # ══════════════════════════════════════════════════════════════════

        # [1, 38, 128]  random positive input
        ("test_torch_cos_values_pattern_000", "_run_cos_values_test"): {
            "param_sets": {
                "v_1x38x128_rand_eager":    (_x38, False),
                "v_1x38x128_rand_compiled": (_x38, True),
            },
        },
        # [1, 38, 128]  all-ones input: cos(1.0) ≈ 0.5403 — fixed reference
        ("test_torch_cos_values_pattern_001", "_run_cos_values_test"): {
            "param_sets": {
                "v_1x38x128_ones_eager":    (_x38_one, False),
                "v_1x38x128_ones_compiled": (_x38_one, True),
            },
        },
        # [1, 38, 128]  pi/2 input: cos(π/2) ≈ 0.0 — near-zero boundary
        ("test_torch_cos_values_pattern_002", "_run_cos_values_test"): {
            "param_sets": {
                "v_1x38x128_pi2_eager":    (_x38_pi, False),
                "v_1x38x128_pi2_compiled": (_x38_pi, True),
            },
        },
        # [1, 1, 128]  random positive input
        ("test_torch_cos_values_pattern_003", "_run_cos_values_test"): {
            "param_sets": {
                "v_1x1x128_rand_eager":    (_x1, False),
                "v_1x1x128_rand_compiled": (_x1, True),
            },
        },
        # [1, 1, 128]  all-ones input
        ("test_torch_cos_values_pattern_004", "_run_cos_values_test"): {
            "param_sets": {
                "v_1x1x128_ones_eager":    (_x1_one, False),
                "v_1x1x128_ones_compiled": (_x1_one, True),
            },
        },
        # [1, 1, 128]  pi/2 input: near-zero boundary check
        ("test_torch_cos_values_pattern_005", "_run_cos_values_test"): {
            "param_sets": {
                "v_1x1x128_pi2_eager":    (_x1_pi, False),
                "v_1x1x128_pi2_compiled": (_x1_pi, True),
            },
        },


        # ══════════════════════════════════════════════════════════════════
        # DTYPE PRESERVATION — float32 in must give float32 out
        # ══════════════════════════════════════════════════════════════════

        # [1, 38, 128]
        ("test_torch_cos_dtype_pattern_000", "_run_cos_dtype_test"): {
            "param_sets": {
                "dtype_1x38x128_f32_eager":    (_x38, False),
                "dtype_1x38x128_f32_compiled": (_x38, True),
            },
        },
        # [1, 1, 128]
        ("test_torch_cos_dtype_pattern_001", "_run_cos_dtype_test"): {
            "param_sets": {
                "dtype_1x1x128_f32_eager":    (_x1, False),
                "dtype_1x1x128_f32_compiled": (_x1, True),
            },
        },
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # ── Base test methods ──────────────────────────────────────────────────

    def _run_cos_shape_test(self, x, compiled):
        def fn(t):
            out = torch.cos(t)
            assert list(out.shape) == list(t.shape), (
                f"Shape mismatch: expected {list(t.shape)}, got {list(out.shape)}"
            )
            return out

        compare_with_cpu(fn, x, compiled=compiled)

    def _run_cos_values_test(self, x, compiled):
        compare_with_cpu(torch.cos, x, compiled=compiled)

    def _run_cos_dtype_test(self, x, compiled):
        def fn(t):
            result = torch.cos(t)
            assert result.dtype == t.dtype, (
                f"dtype changed after cos: expected {t.dtype}, got {result.dtype}"
            )
            return result

        compare_with_cpu(fn, x, compiled=compiled)


# ─────────────────────────────────────────────────────────────────────────────
# TestContiguous
# ─────────────────────────────────────────────────────────────────────────────

class TestContiguous(unittest.TestCase, metaclass=ParameterizedTestMeta):
    """
    Tests for torch.Tensor.contiguous for the Mistral model.

    Input shapes (dtype=bfloat16):
        4-D : [1, 38, 32, 128],  [1, 1, 32, 128]
        3-D : [1, 38, 4096],     [1,  1, 4096]

    Sub-group marks (auto-derived by ParameterizedTestMeta):
        _run_contiguous_shape_test     → @pytest.mark.torch_contiguous_shape
        _run_contiguous_values_test    → @pytest.mark.torch_contiguous_values
        _run_contiguous_noncontig_test → @pytest.mark.torch_contiguous_noncontig
        _run_contiguous_dtype_test     → @pytest.mark.torch_contiguous_dtype
    """

    pytestmark = pytest.mark.torch_contiguous

    torch.manual_seed(0)

    PARAMS = {

        # ══════════════════════════════════════════════════════════════════
        # SHAPE CORRECTNESS — .contiguous() must not change shape
        # ══════════════════════════════════════════════════════════════════

        # [1, 38, 32, 128]
        ("test_torch_contiguous_shape_pattern_000", "_run_contiguous_shape_test"): {
            "param_sets": {
                "s_1x38x32x128_bf16_eager":    (torch.randn(1, 38, 32, 128, dtype=BF16), False),
                "s_1x38x32x128_bf16_compiled": (torch.randn(1, 38, 32, 128, dtype=BF16), True),
            },
        },
        # [1, 38, 4096]
        ("test_torch_contiguous_shape_pattern_001", "_run_contiguous_shape_test"): {
            "param_sets": {
                "s_1x38x4096_bf16_eager":    (torch.randn(1, 38, 4096, dtype=BF16), False),
                "s_1x38x4096_bf16_compiled": (torch.randn(1, 38, 4096, dtype=BF16), True),
            },
        },
        # [1, 1, 32, 128]
        ("test_torch_contiguous_shape_pattern_002", "_run_contiguous_shape_test"): {
            "param_sets": {
                "s_1x1x32x128_bf16_eager":    (torch.randn(1, 1, 32, 128, dtype=BF16), False),
                "s_1x1x32x128_bf16_compiled": (torch.randn(1, 1, 32, 128, dtype=BF16), True),
            },
        },
        # [1, 1, 4096]
        ("test_torch_contiguous_shape_pattern_003", "_run_contiguous_shape_test"): {
            "param_sets": {
                "s_1x1x4096_bf16_eager":    (torch.randn(1, 1, 4096, dtype=BF16), False),
                "s_1x1x4096_bf16_compiled": (torch.randn(1, 1, 4096, dtype=BF16), True),
            },
        },


        # ══════════════════════════════════════════════════════════════════
        # VALUE CORRECTNESS — values preserved after .contiguous()
        # ══════════════════════════════════════════════════════════════════

        # [1, 38, 32, 128]  already-contiguous input
        ("test_torch_contiguous_values_pattern_000", "_run_contiguous_values_test"): {
            "param_sets": {
                "v_1x38x32x128_bf16_eager":    (torch.randn(1, 38, 32, 128, dtype=BF16), False),
                "v_1x38x32x128_bf16_compiled": (torch.randn(1, 38, 32, 128, dtype=BF16), True),
            },
        },
        # [1, 38, 4096]  already-contiguous input
        ("test_torch_contiguous_values_pattern_001", "_run_contiguous_values_test"): {
            "param_sets": {
                "v_1x38x4096_bf16_eager":    (torch.randn(1, 38, 4096, dtype=BF16), False),
                "v_1x38x4096_bf16_compiled": (torch.randn(1, 38, 4096, dtype=BF16), True),
            },
        },
        # [1, 1, 32, 128]  already-contiguous input
        ("test_torch_contiguous_values_pattern_002", "_run_contiguous_values_test"): {
            "param_sets": {
                "v_1x1x32x128_bf16_eager":    (torch.randn(1, 1, 32, 128, dtype=BF16), False),
                "v_1x1x32x128_bf16_compiled": (torch.randn(1, 1, 32, 128, dtype=BF16), True),
            },
        },
        # [1, 1, 4096]  already-contiguous input
        ("test_torch_contiguous_values_pattern_003", "_run_contiguous_values_test"): {
            "param_sets": {
                "v_1x1x4096_bf16_eager":    (torch.randn(1, 1, 4096, dtype=BF16), False),
                "v_1x1x4096_bf16_compiled": (torch.randn(1, 1, 4096, dtype=BF16), True),
            },
        },


        # ══════════════════════════════════════════════════════════════════
        # NON-CONTIGUOUS INPUT — transpose first to get a non-contig view,
        # then .contiguous() must produce is_contiguous()==True + same values.
        # Only dim pairs where BOTH swapped sizes > 1 guarantee non-contiguity.
        # ══════════════════════════════════════════════════════════════════

        # [1, 38, 32, 128]  (1,2) sizes 38↔32 — both > 1 ✓
        ("test_torch_contiguous_noncontig_pattern_000", "_run_contiguous_noncontig_test"): {
            "param_sets": {
                "nc_1x38x32x128_d12_eager":    (1, 2, torch.randn(1, 38, 32, 128, dtype=BF16), False),
                "nc_1x38x32x128_d12_compiled": (1, 2, torch.randn(1, 38, 32, 128, dtype=BF16), True),
            },
        },
        # [1, 38, 32, 128]  (1,3) sizes 38↔128 — both > 1 ✓
        ("test_torch_contiguous_noncontig_pattern_001", "_run_contiguous_noncontig_test"): {
            "param_sets": {
                "nc_1x38x32x128_d13_eager":    (1, 3, torch.randn(1, 38, 32, 128, dtype=BF16), False),
                "nc_1x38x32x128_d13_compiled": (1, 3, torch.randn(1, 38, 32, 128, dtype=BF16), True),
            },
        },
        # [1, 38, 32, 128]  (2,3) sizes 32↔128 — both > 1 ✓
        ("test_torch_contiguous_noncontig_pattern_002", "_run_contiguous_noncontig_test"): {
            "param_sets": {
                "nc_1x38x32x128_d23_eager":    (2, 3, torch.randn(1, 38, 32, 128, dtype=BF16), False),
                "nc_1x38x32x128_d23_compiled": (2, 3, torch.randn(1, 38, 32, 128, dtype=BF16), True),
            },
        },
        # [1, 38, 4096]  (1,2) sizes 38↔4096 — both > 1 ✓
        ("test_torch_contiguous_noncontig_pattern_003", "_run_contiguous_noncontig_test"): {
            "param_sets": {
                "nc_1x38x4096_d12_eager":    (1, 2, torch.randn(1, 38, 4096, dtype=BF16), False),
                "nc_1x38x4096_d12_compiled": (1, 2, torch.randn(1, 38, 4096, dtype=BF16), True),
            },
        },
        # [1, 1, 32, 128]  (2,3) sizes 32↔128 — both > 1 ✓
        ("test_torch_contiguous_noncontig_pattern_004", "_run_contiguous_noncontig_test"): {
            "param_sets": {
                "nc_1x1x32x128_d23_eager":    (2, 3, torch.randn(1, 1, 32, 128, dtype=BF16), False),
                "nc_1x1x32x128_d23_compiled": (2, 3, torch.randn(1, 1, 32, 128, dtype=BF16), True),
            },
        },
        # [1, 1, 4096]  — no pair with both dims > 1: copy test only (fallback)
        ("test_torch_contiguous_noncontig_pattern_005", "_run_contiguous_values_test"): {
            "param_sets": {
                "nc_copy_1x1x4096_bf16_eager":    (torch.randn(1, 1, 4096, dtype=BF16), False),
                "nc_copy_1x1x4096_bf16_compiled": (torch.randn(1, 1, 4096, dtype=BF16), True),
            },
        },


        # ══════════════════════════════════════════════════════════════════
        # DTYPE PRESERVATION — bfloat16 in must give bfloat16 out
        # ══════════════════════════════════════════════════════════════════

        # [1, 38, 32, 128]
        ("test_torch_contiguous_dtype_pattern_000", "_run_contiguous_dtype_test"): {
            "param_sets": {
                "dtype_1x38x32x128_bf16_eager":    (torch.randn(1, 38, 32, 128, dtype=BF16), False),
                "dtype_1x38x32x128_bf16_compiled": (torch.randn(1, 38, 32, 128, dtype=BF16), True),
            },
        },
        # [1, 38, 4096]
        ("test_torch_contiguous_dtype_pattern_001", "_run_contiguous_dtype_test"): {
            "param_sets": {
                "dtype_1x38x4096_bf16_eager":    (torch.randn(1, 38, 4096, dtype=BF16), False),
                "dtype_1x38x4096_bf16_compiled": (torch.randn(1, 38, 4096, dtype=BF16), True),
            },
        },
        # [1, 1, 32, 128]
        ("test_torch_contiguous_dtype_pattern_002", "_run_contiguous_dtype_test"): {
            "param_sets": {
                "dtype_1x1x32x128_bf16_eager":    (torch.randn(1, 1, 32, 128, dtype=BF16), False),
                "dtype_1x1x32x128_bf16_compiled": (torch.randn(1, 1, 32, 128, dtype=BF16), True),
            },
        },
        # [1, 1, 4096]
        ("test_torch_contiguous_dtype_pattern_003", "_run_contiguous_dtype_test"): {
            "param_sets": {
                "dtype_1x1x4096_bf16_eager":    (torch.randn(1, 1, 4096, dtype=BF16), False),
                "dtype_1x1x4096_bf16_compiled": (torch.randn(1, 1, 4096, dtype=BF16), True),
            },
        },
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # ── Base test methods ──────────────────────────────────────────────────

    def _run_contiguous_shape_test(self, x, compiled):
        def fn(t):
            out = t.contiguous()
            assert list(out.shape) == list(t.shape), (
                f"Shape mismatch: expected {list(t.shape)}, got {list(out.shape)}"
            )
            return out

        compare_with_cpu(fn, x, compiled=compiled)

    def _run_contiguous_values_test(self, x, compiled):
        compare_with_cpu(lambda t: t.contiguous(), x, compiled=compiled)

    def _run_contiguous_noncontig_test(self, dim0, dim1, x, compiled):
        raw = torch.transpose(x, dim0, dim1)
        assert not raw.is_contiguous(), (
            f"transpose({dim0},{dim1}) on shape {list(x.shape)} should be non-contiguous"
        )

        def fn(t):
            view = torch.transpose(t, dim0, dim1)
            out  = view.contiguous()
            assert out.is_contiguous(), (
                f"contiguous() result should be contiguous for shape {list(t.shape)}"
            )
            return out

        compare_with_cpu(fn, x, compiled=compiled)

    def _run_contiguous_dtype_test(self, x, compiled):
        def fn(t):
            result = t.contiguous()
            assert result.dtype == t.dtype, (
                f"dtype changed after contiguous(): expected {t.dtype}, got {result.dtype}"
            )
            return result

        compare_with_cpu(fn, x, compiled=compiled)


# ─────────────────────────────────────────────────────────────────────────────
# TestEmbedding
# ─────────────────────────────────────────────────────────────────────────────

class TestEmbedding(unittest.TestCase, metaclass=ParameterizedTestMeta):
    """
    Tests for torch.nn.functional.embedding for the Mistral model.

    Op specification:
        op           : torch.nn.functional.embedding
        weight shape : [131072, 5120]   (vocab_size × embed_dim)
        index shapes : [1, 38]  (prefill),  [1, 1]  (decode)
        index dtype  : torch.int64
        weight dtype : torch.bfloat16
        output shape : [*index.shape, embed_dim]

    Sub-group marks (auto-derived by ParameterizedTestMeta):
        _run_embedding_shape_test  → @pytest.mark.torch_embedding_shape
        _run_embedding_values_test → @pytest.mark.torch_embedding_values
        _run_embedding_dtype_test  → @pytest.mark.torch_embedding_dtype
    """

    pytestmark = pytest.mark.torch_embedding

    PARAMS = {

        # ══════════════════════════════════════════════════════════════════
        # SHAPE CORRECTNESS — output shape: [*index.shape, embed_dim]
        # ══════════════════════════════════════════════════════════════════

        # [1, 38] + [131072, 5120] → [1, 38, 5120]
        ("test_torch_embedding_shape_pattern_000", "_run_embedding_shape_test"): {
            "param_sets": {
                "s_1x38_eager":    (torch.randint(0, VOCAB_SIZE, (1, 38), dtype=I64), _W, False),
                "s_1x38_compiled": (torch.randint(0, VOCAB_SIZE, (1, 38), dtype=I64), _W, True),
            },
        },
        # [1, 1] + [131072, 5120] → [1, 1, 5120]
        ("test_torch_embedding_shape_pattern_001", "_run_embedding_shape_test"): {
            "param_sets": {
                "s_1x1_eager":    (torch.randint(0, VOCAB_SIZE, (1, 1), dtype=I64), _W, False),
                "s_1x1_compiled": (torch.randint(0, VOCAB_SIZE, (1, 1), dtype=I64), _W, True),
            },
        },


        # ══════════════════════════════════════════════════════════════════
        # VALUE CORRECTNESS — embedding(indices, weight) CPU vs Spyre
        # ══════════════════════════════════════════════════════════════════

        # [1, 38]  random indices — general prefill lookup
        ("test_torch_embedding_values_pattern_000", "_run_embedding_values_test"): {
            "param_sets": {
                "v_1x38_rand_eager":    (torch.randint(0, VOCAB_SIZE, (1, 38), dtype=I64), _W, False),
                "v_1x38_rand_compiled": (torch.randint(0, VOCAB_SIZE, (1, 38), dtype=I64), _W, True),
            },
        },
        # [1, 38]  all-zero indices — exercises first row of weight table
        ("test_torch_embedding_values_pattern_001", "_run_embedding_values_test"): {
            "param_sets": {
                "v_1x38_zero_eager":    (torch.zeros(1, 38, dtype=I64), _W, False),
                "v_1x38_zero_compiled": (torch.zeros(1, 38, dtype=I64), _W, True),
            },
        },
        # [1, 38]  last vocab index — upper boundary check (index == 131071)
        ("test_torch_embedding_values_pattern_002", "_run_embedding_values_test"): {
            "param_sets": {
                "v_1x38_last_eager":    (torch.full((1, 38), VOCAB_SIZE - 1, dtype=I64), _W, False),
                "v_1x38_last_compiled": (torch.full((1, 38), VOCAB_SIZE - 1, dtype=I64), _W, True),
            },
        },
        # [1, 1]  random index — single-token decode step
        ("test_torch_embedding_values_pattern_003", "_run_embedding_values_test"): {
            "param_sets": {
                "v_1x1_rand_eager":    (torch.randint(0, VOCAB_SIZE, (1, 1), dtype=I64), _W, False),
                "v_1x1_rand_compiled": (torch.randint(0, VOCAB_SIZE, (1, 1), dtype=I64), _W, True),
            },
        },
        # [1, 1]  all-zero index — exercises first row of weight table (decode)
        ("test_torch_embedding_values_pattern_004", "_run_embedding_values_test"): {
            "param_sets": {
                "v_1x1_zero_eager":    (torch.zeros(1, 1, dtype=I64), _W, False),
                "v_1x1_zero_compiled": (torch.zeros(1, 1, dtype=I64), _W, True),
            },
        },
        # [1, 1]  last vocab index — upper boundary check (decode)
        ("test_torch_embedding_values_pattern_005", "_run_embedding_values_test"): {
            "param_sets": {
                "v_1x1_last_eager":    (torch.full((1, 1), VOCAB_SIZE - 1, dtype=I64), _W, False),
                "v_1x1_last_compiled": (torch.full((1, 1), VOCAB_SIZE - 1, dtype=I64), _W, True),
            },
        },


        # ══════════════════════════════════════════════════════════════════
        # DTYPE PRESERVATION — output dtype must match weight dtype (bfloat16)
        # ══════════════════════════════════════════════════════════════════

        # [1, 38]
        ("test_torch_embedding_dtype_pattern_000", "_run_embedding_dtype_test"): {
            "param_sets": {
                "dtype_1x38_bf16_eager":    (torch.randint(0, VOCAB_SIZE, (1, 38), dtype=I64), _W, False),
                "dtype_1x38_bf16_compiled": (torch.randint(0, VOCAB_SIZE, (1, 38), dtype=I64), _W, True),
            },
        },
        # [1, 1]
        ("test_torch_embedding_dtype_pattern_001", "_run_embedding_dtype_test"): {
            "param_sets": {
                "dtype_1x1_bf16_eager":    (torch.randint(0, VOCAB_SIZE, (1, 1), dtype=I64), _W, False),
                "dtype_1x1_bf16_compiled": (torch.randint(0, VOCAB_SIZE, (1, 1), dtype=I64), _W, True),
            },
        },
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # ── Base test methods ──────────────────────────────────────────────────

    def _run_embedding_shape_test(self, indices, weight, compiled):
        expected = list(indices.shape) + [weight.shape[1]]

        def fn(idx, w):
            out = F.embedding(idx, w)
            assert list(out.shape) == expected, (
                f"Shape mismatch: expected {expected}, got {list(out.shape)}"
            )
            return out

        compare_with_cpu(fn, indices, weight, compiled=compiled)

    def _run_embedding_values_test(self, indices, weight, compiled):
        compare_with_cpu(
            lambda idx, w: F.embedding(idx, w),
            indices, weight,
            compiled=compiled,
        )

    def _run_embedding_dtype_test(self, indices, weight, compiled):
        def fn(idx, w):
            result = F.embedding(idx, w)
            assert result.dtype == w.dtype, (
                f"dtype changed after embedding: expected {w.dtype}, got {result.dtype}"
            )
            return result

        compare_with_cpu(fn, indices, weight, compiled=compiled)


# ─────────────────────────────────────────────────────────────────────────────
# TestGetitem
# ─────────────────────────────────────────────────────────────────────────────

class TestGetitem(unittest.TestCase, metaclass=ParameterizedTestMeta):
    """
    Tests for torch.Tensor.__getitem__ for the Mistral model.

    Index shapes  (int64)   : [64], [1, 38], [1, 1]
    Data  shapes  (float32) : [64]
    Data  shapes  (bfloat16): [1, 32, 38, 128], [1, 8, 38, 128], [1, 38, 5120]
                              [1, 32,  1, 128], [1, 8,  1, 128], [1,  1, 5120]

    Param tuple layout
    ------------------
    _run_getitem_shape_test  : (x, idx, expected_shape, compiled)
    _run_getitem_values_test : (x, idx, compiled)
    _run_getitem_dtype_test  : (x, idx, compiled)

    Sub-group marks (auto-derived by ParameterizedTestMeta):
        _run_getitem_shape_test  → @pytest.mark.torch_getitem_shape
        _run_getitem_values_test → @pytest.mark.torch_getitem_values
        _run_getitem_dtype_test  → @pytest.mark.torch_getitem_dtype
    """

    pytestmark = pytest.mark.torch_getitem

    torch.manual_seed(0)

    PARAMS = {

        # ══════════════════════════════════════════════════════════════════
        # SHAPE CORRECTNESS
        # ══════════════════════════════════════════════════════════════════

        # [64] float32  →  [:32]  →  [32]
        ("test_torch_getitem_shape_pattern_000", "_run_getitem_shape_test"): {
            "param_sets": {
                "s_64_sl32_f32_eager":    (torch.randn(64, dtype=F32), S(None, 32), [32], False),
                "s_64_sl32_f32_compiled": (torch.randn(64, dtype=F32), S(None, 32), [32], True),
            },
        },
        # [1, 38] int64  →  [0]  →  [38]
        ("test_torch_getitem_shape_pattern_001", "_run_getitem_shape_test"): {
            "param_sets": {
                "s_1x38_idx0_i64_eager":    (torch.randint(0, 1000, (1, 38), dtype=I64), 0, [38], False),
                "s_1x38_idx0_i64_compiled": (torch.randint(0, 1000, (1, 38), dtype=I64), 0, [38], True),
            },
        },
        # [1, 1] int64  →  [0]  →  [1]
        ("test_torch_getitem_shape_pattern_002", "_run_getitem_shape_test"): {
            "param_sets": {
                "s_1x1_idx0_i64_eager":    (torch.randint(0, 1000, (1, 1), dtype=I64), 0, [1], False),
                "s_1x1_idx0_i64_compiled": (torch.randint(0, 1000, (1, 1), dtype=I64), 0, [1], True),
            },
        },
        # [1, 32, 38, 128] bfloat16  →  [:, :, :1, :]  →  [1, 32, 1, 128]  (decode slice)
        ("test_torch_getitem_shape_pattern_003", "_run_getitem_shape_test"): {
            "param_sets": {
                "s_1x32x38x128_d2sl1_bf16_eager":    (torch.randn(1, 32, 38, 128, dtype=BF16), (_, _, S(None, 1), _), [1, 32, 1, 128], False),
                "s_1x32x38x128_d2sl1_bf16_compiled": (torch.randn(1, 32, 38, 128, dtype=BF16), (_, _, S(None, 1), _), [1, 32, 1, 128], True),
            },
        },
        # [1, 8, 38, 128] bfloat16  →  [:, :, :1, :]  →  [1, 8, 1, 128]
        ("test_torch_getitem_shape_pattern_004", "_run_getitem_shape_test"): {
            "param_sets": {
                "s_1x8x38x128_d2sl1_bf16_eager":    (torch.randn(1, 8, 38, 128, dtype=BF16), (_, _, S(None, 1), _), [1, 8, 1, 128], False),
                "s_1x8x38x128_d2sl1_bf16_compiled": (torch.randn(1, 8, 38, 128, dtype=BF16), (_, _, S(None, 1), _), [1, 8, 1, 128], True),
            },
        },
        # [1, 32, 1, 128] bfloat16  →  [0]  →  [32, 1, 128]
        ("test_torch_getitem_shape_pattern_005", "_run_getitem_shape_test"): {
            "param_sets": {
                "s_1x32x1x128_idx0_bf16_eager":    (torch.randn(1, 32, 1, 128, dtype=BF16), 0, [32, 1, 128], False),
                "s_1x32x1x128_idx0_bf16_compiled": (torch.randn(1, 32, 1, 128, dtype=BF16), 0, [32, 1, 128], True),
            },
        },
        # [1, 8, 1, 128] bfloat16  →  [0]  →  [8, 1, 128]
        ("test_torch_getitem_shape_pattern_006", "_run_getitem_shape_test"): {
            "param_sets": {
                "s_1x8x1x128_idx0_bf16_eager":    (torch.randn(1, 8, 1, 128, dtype=BF16), 0, [8, 1, 128], False),
                "s_1x8x1x128_idx0_bf16_compiled": (torch.randn(1, 8, 1, 128, dtype=BF16), 0, [8, 1, 128], True),
            },
        },
        # [1, 1, 5120] bfloat16  →  [0]  →  [1, 5120]
        ("test_torch_getitem_shape_pattern_007", "_run_getitem_shape_test"): {
            "param_sets": {
                "s_1x1x5120_idx0_bf16_eager":    (torch.randn(1, 1, 5120, dtype=BF16), 0, [1, 5120], False),
                "s_1x1x5120_idx0_bf16_compiled": (torch.randn(1, 1, 5120, dtype=BF16), 0, [1, 5120], True),
            },
        },
        # [1, 38, 5120] bfloat16  →  [:, 0, :]  →  [1, 5120]  (first token hidden state)
        ("test_torch_getitem_shape_pattern_008", "_run_getitem_shape_test"): {
            "param_sets": {
                "s_1x38x5120_d1idx0_bf16_eager":    (torch.randn(1, 38, 5120, dtype=BF16), (_, 0, _), [1, 5120], False),
                "s_1x38x5120_d1idx0_bf16_compiled": (torch.randn(1, 38, 5120, dtype=BF16), (_, 0, _), [1, 5120], True),
            },
        },


        # ══════════════════════════════════════════════════════════════════
        # VALUE CORRECTNESS
        # ══════════════════════════════════════════════════════════════════

        # [64] float32  →  [:32]  first half
        ("test_torch_getitem_values_pattern_000", "_run_getitem_values_test"): {
            "param_sets": {
                "v_64_sl32_f32_eager":    (torch.randn(64, dtype=F32), S(None, 32), False),
                "v_64_sl32_f32_compiled": (torch.randn(64, dtype=F32), S(None, 32), True),
            },
        },
        # [64] float32  →  [32:]  second half
        ("test_torch_getitem_values_pattern_001", "_run_getitem_values_test"): {
            "param_sets": {
                "v_64_sl32end_f32_eager":    (torch.randn(64, dtype=F32), S(32, None), False),
                "v_64_sl32end_f32_compiled": (torch.randn(64, dtype=F32), S(32, None), True),
            },
        },
        # [1, 38] int64  →  [0]
        ("test_torch_getitem_values_pattern_002", "_run_getitem_values_test"): {
            "param_sets": {
                "v_1x38_idx0_i64_eager":    (torch.randint(0, 1000, (1, 38), dtype=I64), 0, False),
                "v_1x38_idx0_i64_compiled": (torch.randint(0, 1000, (1, 38), dtype=I64), 0, True),
            },
        },
        # [1, 1] int64  →  [0]
        ("test_torch_getitem_values_pattern_003", "_run_getitem_values_test"): {
            "param_sets": {
                "v_1x1_idx0_i64_eager":    (torch.randint(0, 1000, (1, 1), dtype=I64), 0, False),
                "v_1x1_idx0_i64_compiled": (torch.randint(0, 1000, (1, 1), dtype=I64), 0, True),
            },
        },
        # [1, 32, 38, 128] bfloat16  →  [:, :, :1, :]  (decode slice)
        ("test_torch_getitem_values_pattern_004", "_run_getitem_values_test"): {
            "param_sets": {
                "v_1x32x38x128_d2sl1_bf16_eager":    (torch.randn(1, 32, 38, 128, dtype=BF16), (_, _, S(None, 1), _), False),
                "v_1x32x38x128_d2sl1_bf16_compiled": (torch.randn(1, 32, 38, 128, dtype=BF16), (_, _, S(None, 1), _), True),
            },
        },
        # [1, 32, 38, 128] bfloat16  →  [:, :, :38, :]  full seq slice (all tokens)
        ("test_torch_getitem_values_pattern_005", "_run_getitem_values_test"): {
            "param_sets": {
                "v_1x32x38x128_d2sl38_bf16_eager":    (torch.randn(1, 32, 38, 128, dtype=BF16), (_, _, S(None, 38), _), False),
                "v_1x32x38x128_d2sl38_bf16_compiled": (torch.randn(1, 32, 38, 128, dtype=BF16), (_, _, S(None, 38), _), True),
            },
        },
        # [1, 8, 38, 128] bfloat16  →  [:, :, :1, :]  (decode slice)
        ("test_torch_getitem_values_pattern_006", "_run_getitem_values_test"): {
            "param_sets": {
                "v_1x8x38x128_d2sl1_bf16_eager":    (torch.randn(1, 8, 38, 128, dtype=BF16), (_, _, S(None, 1), _), False),
                "v_1x8x38x128_d2sl1_bf16_compiled": (torch.randn(1, 8, 38, 128, dtype=BF16), (_, _, S(None, 1), _), True),
            },
        },
        # [1, 8, 38, 128] bfloat16  →  [:, :, :38, :]  full seq slice
        ("test_torch_getitem_values_pattern_007", "_run_getitem_values_test"): {
            "param_sets": {
                "v_1x8x38x128_d2sl38_bf16_eager":    (torch.randn(1, 8, 38, 128, dtype=BF16), (_, _, S(None, 38), _), False),
                "v_1x8x38x128_d2sl38_bf16_compiled": (torch.randn(1, 8, 38, 128, dtype=BF16), (_, _, S(None, 38), _), True),
            },
        },
        # [1, 32, 1, 128] bfloat16  →  [:, :, 0, :]  →  [1, 32, 128]
        ("test_torch_getitem_values_pattern_008", "_run_getitem_values_test"): {
            "param_sets": {
                "v_1x32x1x128_d2idx0_bf16_eager":    (torch.randn(1, 32, 1, 128, dtype=BF16), (_, _, 0, _), False),
                "v_1x32x1x128_d2idx0_bf16_compiled": (torch.randn(1, 32, 1, 128, dtype=BF16), (_, _, 0, _), True),
            },
        },
        # [1, 8, 1, 128] bfloat16  →  [:, :, 0, :]  →  [1, 8, 128]
        ("test_torch_getitem_values_pattern_009", "_run_getitem_values_test"): {
            "param_sets": {
                "v_1x8x1x128_d2idx0_bf16_eager":    (torch.randn(1, 8, 1, 128, dtype=BF16), (_, _, 0, _), False),
                "v_1x8x1x128_d2idx0_bf16_compiled": (torch.randn(1, 8, 1, 128, dtype=BF16), (_, _, 0, _), True),
            },
        },
        # [1, 1, 5120] bfloat16  →  [0]  →  [1, 5120]
        ("test_torch_getitem_values_pattern_010", "_run_getitem_values_test"): {
            "param_sets": {
                "v_1x1x5120_idx0_bf16_eager":    (torch.randn(1, 1, 5120, dtype=BF16), 0, False),
                "v_1x1x5120_idx0_bf16_compiled": (torch.randn(1, 1, 5120, dtype=BF16), 0, True),
            },
        },
        # [1, 38, 5120] bfloat16  →  [:, 0, :]  first token hidden state
        ("test_torch_getitem_values_pattern_011", "_run_getitem_values_test"): {
            "param_sets": {
                "v_1x38x5120_d1idx0_bf16_eager":    (torch.randn(1, 38, 5120, dtype=BF16), (_, 0, _), False),
                "v_1x38x5120_d1idx0_bf16_compiled": (torch.randn(1, 38, 5120, dtype=BF16), (_, 0, _), True),
            },
        },
        # [1, 38, 5120] bfloat16  →  [:, -1, :]  last token hidden state (boundary)
        ("test_torch_getitem_values_pattern_012", "_run_getitem_values_test"): {
            "param_sets": {
                "v_1x38x5120_d1idxm1_bf16_eager":    (torch.randn(1, 38, 5120, dtype=BF16), (_, -1, _), False),
                "v_1x38x5120_d1idxm1_bf16_compiled": (torch.randn(1, 38, 5120, dtype=BF16), (_, -1, _), True),
            },
        },


        # ══════════════════════════════════════════════════════════════════
        # DTYPE PRESERVATION — dtype must not change after indexing
        # ══════════════════════════════════════════════════════════════════

        # [64] float32  →  [:32]
        ("test_torch_getitem_dtype_pattern_000", "_run_getitem_dtype_test"): {
            "param_sets": {
                "dtype_64_f32_eager":    (torch.randn(64, dtype=F32), S(None, 32), False),
                "dtype_64_f32_compiled": (torch.randn(64, dtype=F32), S(None, 32), True),
            },
        },
        # [1, 38] int64  →  [0]
        ("test_torch_getitem_dtype_pattern_001", "_run_getitem_dtype_test"): {
            "param_sets": {
                "dtype_1x38_i64_eager":    (torch.randint(0, 1000, (1, 38), dtype=I64), 0, False),
                "dtype_1x38_i64_compiled": (torch.randint(0, 1000, (1, 38), dtype=I64), 0, True),
            },
        },
        # [1, 1] int64  →  [0]
        ("test_torch_getitem_dtype_pattern_002", "_run_getitem_dtype_test"): {
            "param_sets": {
                "dtype_1x1_i64_eager":    (torch.randint(0, 1000, (1, 1), dtype=I64), 0, False),
                "dtype_1x1_i64_compiled": (torch.randint(0, 1000, (1, 1), dtype=I64), 0, True),
            },
        },
        # [1, 32, 38, 128] bfloat16  →  [:, :, :1, :]
        ("test_torch_getitem_dtype_pattern_003", "_run_getitem_dtype_test"): {
            "param_sets": {
                "dtype_1x32x38x128_bf16_eager":    (torch.randn(1, 32, 38, 128, dtype=BF16), (_, _, S(None, 1), _), False),
                "dtype_1x32x38x128_bf16_compiled": (torch.randn(1, 32, 38, 128, dtype=BF16), (_, _, S(None, 1), _), True),
            },
        },
        # [1, 8, 38, 128] bfloat16  →  [:, :, :1, :]
        ("test_torch_getitem_dtype_pattern_004", "_run_getitem_dtype_test"): {
            "param_sets": {
                "dtype_1x8x38x128_bf16_eager":    (torch.randn(1, 8, 38, 128, dtype=BF16), (_, _, S(None, 1), _), False),
                "dtype_1x8x38x128_bf16_compiled": (torch.randn(1, 8, 38, 128, dtype=BF16), (_, _, S(None, 1), _), True),
            },
        },
        # [1, 32, 1, 128] bfloat16  →  [0]
        ("test_torch_getitem_dtype_pattern_005", "_run_getitem_dtype_test"): {
            "param_sets": {
                "dtype_1x32x1x128_bf16_eager":    (torch.randn(1, 32, 1, 128, dtype=BF16), 0, False),
                "dtype_1x32x1x128_bf16_compiled": (torch.randn(1, 32, 1, 128, dtype=BF16), 0, True),
            },
        },
        # [1, 8, 1, 128] bfloat16  →  [0]
        ("test_torch_getitem_dtype_pattern_006", "_run_getitem_dtype_test"): {
            "param_sets": {
                "dtype_1x8x1x128_bf16_eager":    (torch.randn(1, 8, 1, 128, dtype=BF16), 0, False),
                "dtype_1x8x1x128_bf16_compiled": (torch.randn(1, 8, 1, 128, dtype=BF16), 0, True),
            },
        },
        # [1, 1, 5120] bfloat16  →  [0]
        ("test_torch_getitem_dtype_pattern_007", "_run_getitem_dtype_test"): {
            "param_sets": {
                "dtype_1x1x5120_bf16_eager":    (torch.randn(1, 1, 5120, dtype=BF16), 0, False),
                "dtype_1x1x5120_bf16_compiled": (torch.randn(1, 1, 5120, dtype=BF16), 0, True),
            },
        },
        # [1, 38, 5120] bfloat16  →  [:, 0, :]
        ("test_torch_getitem_dtype_pattern_008", "_run_getitem_dtype_test"): {
            "param_sets": {
                "dtype_1x38x5120_bf16_eager":    (torch.randn(1, 38, 5120, dtype=BF16), (_, 0, _), False),
                "dtype_1x38x5120_bf16_compiled": (torch.randn(1, 38, 5120, dtype=BF16), (_, 0, _), True),
            },
        },
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # ── Base test methods ──────────────────────────────────────────────────

    def _run_getitem_shape_test(self, x, idx, expected_shape, compiled):
        def fn(t):
            out = t[idx]
            assert list(out.shape) == expected_shape, (
                f"Shape mismatch: expected {expected_shape}, got {list(out.shape)}"
            )
            return out

        compare_with_cpu(fn, x, compiled=compiled)

    def _run_getitem_values_test(self, x, idx, compiled):
        compare_with_cpu(lambda t: t[idx], x, compiled=compiled)

    def _run_getitem_dtype_test(self, x, idx, compiled):
        def fn(t):
            result = t[idx]
            assert result.dtype == t.dtype, (
                f"dtype changed after getitem: expected {t.dtype}, got {result.dtype}"
            )
            return result

        compare_with_cpu(fn, x, compiled=compiled)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main()
