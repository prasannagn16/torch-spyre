import os
import sys
import math
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
    make_strided_tensor,
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
    compare_with_cpu,
)

S = slice
_ = slice(None)

# ═════════════════════════════════════════════════════════════════════════════
#  SDPA Strided Tensor Factories for Mistral-Small-3.2-24B-Instruct-2506
# ═════════════════════════════════════════════════════════════════════════════

# KV-cache configuration
_KV_CACHE_LEN = 2048  # Max cache length (from config max_position_embeddings)
_PREFILL_SEQ = 38     # Typical prefill sequence length from trace
_DECODE_SEQ = 1       # Decode sequence length

def _make_q_prefill_strides(batch: int = 1, seq_len: int = _PREFILL_SEQ) -> tuple:
    """Calculate Q strides for prefill: [B, 32, S, 128]"""
    batch_stride = NUM_Q_HEADS * seq_len * HEAD_DIM
    heads_stride = HEAD_DIM
    seq_stride = NUM_Q_HEADS * HEAD_DIM
    return (batch_stride, heads_stride, seq_stride, 1)

def _make_q_decode_strides(batch: int = 1) -> tuple:
    """Calculate Q strides for decode: [B, 32, 1, 128]"""
    batch_stride = NUM_Q_HEADS * _DECODE_SEQ * HEAD_DIM
    heads_stride = HEAD_DIM
    seq_stride = NUM_Q_HEADS * HEAD_DIM
    return (batch_stride, heads_stride, seq_stride, 1)

def _make_kv_strides(batch: int = 1, cache_len: int = _KV_CACHE_LEN) -> tuple:
    """Calculate K/V strides for KV-cache: [B, 8, L, 128]"""
    batch_stride = NUM_KV_HEADS * cache_len * HEAD_DIM
    heads_stride = HEAD_DIM
    seq_stride = NUM_KV_HEADS * HEAD_DIM
    return (batch_stride, heads_stride, seq_stride, 1)

def _make_q_prefill(
    batch: int = 1, 
    seq: int = _PREFILL_SEQ, 
    dtype: torch.dtype = DEFAULT_DTYPE,
    fill: str = "randn"
) -> torch.Tensor:
    """Return Q with prefill layout [B, 32, S, 128] with proper strides."""
    strides = _make_q_prefill_strides(batch, seq)
    return make_strided_tensor(
        (batch, NUM_Q_HEADS, seq, HEAD_DIM),
        strides,
        dtype=dtype,
        fill=fill,
    )

def _make_q_decode(
    batch: int = 1,
    dtype: torch.dtype = DEFAULT_DTYPE,
    fill: str = "randn"
) -> torch.Tensor:
    """Return Q with decode layout [B, 32, 1, 128] with proper strides."""
    strides = _make_q_decode_strides(batch)
    return make_strided_tensor(
        (batch, NUM_Q_HEADS, 1, HEAD_DIM),
        strides,
        dtype=dtype,
        fill=fill,
    )

def _make_kv(
    batch: int = 1,
    cache_len: int = _KV_CACHE_LEN,
    dtype: torch.dtype = DEFAULT_DTYPE,
    fill: str = "randn"
) -> tuple:
    """Return (k, v) with KV-cache layout [B, 8, L, 128]."""
    strides = _make_kv_strides(batch, cache_len)
    k = make_strided_tensor(
        (batch, NUM_KV_HEADS, cache_len, HEAD_DIM),
        strides,
        dtype=dtype,
        fill=fill,
    )
    v = make_strided_tensor(
        (batch, NUM_KV_HEADS, cache_len, HEAD_DIM),
        strides,
        dtype=dtype,
        fill=fill,
    )
    return k, v

# Prefill param dicts with strided tensors
_STRIDED_PREFILL_PARAMS = {
    "bs1_seq38_fp16": (_make_q_prefill(1, 38, F16), *_make_kv(1, _KV_CACHE_LEN, F16)),
    "bs1_seq38_fp32": (_make_q_prefill(1, 38, torch.float32), *_make_kv(1, _KV_CACHE_LEN, torch.float32)),
    "bs1_seq38_bf16": (_make_q_prefill(1, 38, torch.bfloat16), *_make_kv(1, _KV_CACHE_LEN, torch.bfloat16)),
}

# Additional prefill variants with different sequence lengths
_STRIDED_PREFILL_VARIANTS = {
    "bs1_seq1_fp16":   (_make_q_prefill(1, 1, F16), *_make_kv(1, _KV_CACHE_LEN, F16)),
    "bs1_seq8_fp16":   (_make_q_prefill(1, 8, F16), *_make_kv(1, _KV_CACHE_LEN, F16)),
    "bs1_seq16_fp16":  (_make_q_prefill(1, 16, F16), *_make_kv(1, _KV_CACHE_LEN, F16)),
    "bs1_seq32_fp16":  (_make_q_prefill(1, 32, F16), *_make_kv(1, _KV_CACHE_LEN, F16)),
    "bs1_seq64_fp16":  (_make_q_prefill(1, 64, F16), *_make_kv(1, _KV_CACHE_LEN, F16)),
    "bs1_seq128_fp16": (_make_q_prefill(1, 128, F16), *_make_kv(1, _KV_CACHE_LEN, F16)),
}

# Decode param dicts with strided tensors
_STRIDED_DECODE_PARAMS = {
    "bs1_kv2048_fp16": (_make_q_decode(1, F16), *_make_kv(1, _KV_CACHE_LEN, F16)),
    "bs1_kv2048_zeros": (_make_q_decode(1, F16, fill="zeros"), *_make_kv(1, _KV_CACHE_LEN, F16, fill="zeros")),
    "bs1_kv2048_ones": (_make_q_decode(1, F16, fill="ones"), *_make_kv(1, _KV_CACHE_LEN, F16, fill="ones")),
}

# Growing KV-cache variants for decode (different cache lengths)
_STRIDED_GROWING_KV_PARAMS = {
    f"kv{kv}": (_make_q_decode(1, F16), *_make_kv(1, kv, F16))
    for kv in [1, 2, 4, 8, 16, 32, 38, 64, 128, 256, 512, 1024, 2048]
}

# Batch variants for consistency testing
_STRIDED_BATCH_PARAMS = {
    "bs2_seq38_fp16": (_make_q_prefill(2, 38, F16), *_make_kv(2, _KV_CACHE_LEN, F16)),
    "bs4_seq38_fp16": (_make_q_prefill(4, 38, F16), *_make_kv(4, _KV_CACHE_LEN, F16)),
    "bs8_seq38_fp16": (_make_q_prefill(8, 38, F16), *_make_kv(8, _KV_CACHE_LEN, F16)),
}

# Sliding window variants (though SLIDING_WINDOW is None for this model)
_STRIDED_SLIDING_WINDOW_PARAMS = {
    "seq38_kv2048": (_make_q_prefill(1, 38, F16), *_make_kv(1, _KV_CACHE_LEN, F16)),
    "seq64_kv2048": (_make_q_prefill(1, 64, F16), *_make_kv(1, _KV_CACHE_LEN, F16)),
    "seq128_kv2048": (_make_q_prefill(1, 128, F16), *_make_kv(1, _KV_CACHE_LEN, F16)),
}

def _slice_kv_to_seq(kv: torch.Tensor, seq_len: int) -> torch.Tensor:
    """
    Slice KV cache to specific sequence length while preserving stride patterns.
    Returns a contiguous tensor to avoid stride issues on Spyre.
    """
    return kv[:, :, :seq_len, :].contiguous()


# ═════════════════════════════════════════════════════════════════════════════
#  TestSDPA  —  torch.nn.functional.scaled_dot_product_attention
# ═════════════════════════════════════════════════════════════════════════════

class TestSDPA(unittest.TestCase, metaclass=ParameterizedTestMeta):
    """
    Eager and compiled CPU vs Spyre SDPA comparison tests for
    Mistral-Small-3.2-24B-Instruct-2506.
    """
 
    pytestmark = pytest.mark.torch_sdpa
 
    torch.manual_seed(0xBEEF_2506)
    
    # Adjusted tolerances based on torch-spyre SDPA implementation
    # Spyre uses different accumulation order which can cause small differences
    SDPA_EAGER_ATOL = 2e-2
    SDPA_EAGER_RTOL = 2e-2
    SDPA_COMPILED_ATOL = 5e-2
    SDPA_COMPILED_RTOL = 5e-2
 
    PARAMS = {
        # ── Decode tests (target shape) ───────────────────────────────────────
        ("test_sdpa_decode", "test_sdpa_decode"): {
            "param_sets": {
                "bs1_kv2048_fp16_eager":    (*_STRIDED_DECODE_PARAMS["bs1_kv2048_fp16"], False),
                "bs1_kv2048_fp16_compiled": (*_STRIDED_DECODE_PARAMS["bs1_kv2048_fp16"], True),
            },
        },
 
        # ── Prefill causal tests ──────────────────────────────────────────────
        ("test_sdpa_prefill_causal", "test_sdpa_prefill_causal"): {
            "param_sets": {
                "bs1_seq38_fp16_eager":    (*_STRIDED_PREFILL_PARAMS["bs1_seq38_fp16"], False),
                "bs1_seq38_fp16_compiled": (*_STRIDED_PREFILL_PARAMS["bs1_seq38_fp16"], True),
                "bs1_seq8_fp16_eager":     (*_STRIDED_PREFILL_VARIANTS["bs1_seq8_fp16"], False),
                "bs1_seq16_fp16_eager":    (*_STRIDED_PREFILL_VARIANTS["bs1_seq16_fp16"], False),
                "bs1_seq32_fp16_eager":    (*_STRIDED_PREFILL_VARIANTS["bs1_seq32_fp16"], False),
                "bs1_seq64_fp16_eager":    (*_STRIDED_PREFILL_VARIANTS["bs1_seq64_fp16"], False),
                "bs1_seq128_fp16_eager":   (*_STRIDED_PREFILL_VARIANTS["bs1_seq128_fp16"], False),
            },
        },
 
        # ── Growing KV cache (autoregressive decode) ──────────────────────────
        ("test_sdpa_growing_kvcache", "test_sdpa_growing_kvcache"): {
            "param_sets": {
                **{f"{k}_eager":    (*v, False) for k, v in _STRIDED_GROWING_KV_PARAMS.items() 
                   if int(k.replace("kv", "")) >= 8},  # Skip very small KV caches
                **{f"{k}_compiled": (*v, True)  for k, v in _STRIDED_GROWING_KV_PARAMS.items()
                   if int(k.replace("kv", "")) >= 8},
            },
        },
 
        # ── Batch consistency tests ───────────────────────────────────────────
        ("test_sdpa_batch_consistency", "test_sdpa_batch_consistency"): {
            "param_sets": {
                "bs2_seq38_fp16_eager":    (*_STRIDED_BATCH_PARAMS["bs2_seq38_fp16"], False),
                "bs2_seq38_fp16_compiled": (*_STRIDED_BATCH_PARAMS["bs2_seq38_fp16"], True),
                "bs4_seq38_fp16_eager":    (*_STRIDED_BATCH_PARAMS["bs4_seq38_fp16"], False),
            },
        },
 
        # ── Causal flag vs mask tests ─────────────────────────────────────────
        ("test_sdpa_causal_flag_vs_mask", "test_sdpa_causal_flag_vs_mask"): {
            "param_sets": {
                "bs1_seq38_fp16_eager":    (*_STRIDED_PREFILL_PARAMS["bs1_seq38_fp16"], False),
                "bs1_seq38_fp16_compiled": (*_STRIDED_PREFILL_PARAMS["bs1_seq38_fp16"], True),
            },
        },
 
        # ── Attention weights sum to one tests ────────────────────────────────
        ("test_sdpa_weights_sum_to_one", "test_sdpa_weights_sum_to_one"): {
            "param_sets": {
                "bs1_seq38_fp16_eager":    (*_STRIDED_PREFILL_PARAMS["bs1_seq38_fp16"], False),
            },
        },
 
        # ── GQA shape tests ───────────────────────────────────────────────────
        ("test_sdpa_gqa_shape", "test_sdpa_gqa_shape"): {
            "param_sets": {
                "bs2_seq38_fp16_eager":    (*_STRIDED_BATCH_PARAMS["bs2_seq38_fp16"], False),
                "bs2_seq38_fp16_compiled": (*_STRIDED_BATCH_PARAMS["bs2_seq38_fp16"], True),
            },
        },
 
        # ── Gradient flow tests (eager only) ──────────────────────────────────
        ("test_sdpa_gradient_flow", "test_sdpa_gradient_flow"): {
            "param_sets": {
                "bs1_seq38_fp16_eager": (*_STRIDED_PREFILL_PARAMS["bs1_seq38_fp16"], False),
            },
        },
 
        # ── Determinism tests ─────────────────────────────────────────────────
        ("test_sdpa_determinism", "test_sdpa_determinism"): {
            "param_sets": {
                "bs1_seq38_fp16_eager":    (*_STRIDED_PREFILL_PARAMS["bs1_seq38_fp16"], False),
                "bs1_decode_fp16_eager":   (*_STRIDED_DECODE_PARAMS["bs1_kv2048_fp16"], False),
            },
        },
    }
 
    # ── Helper to create causal mask without in-place ops ─────────────────────
    def _make_causal_mask(self, seq_len: int, dtype: torch.dtype, device):
        """Create causal mask using torch.where (no in-place modification)."""
        # Create a boolean mask for upper triangular (excluding diagonal)
        upper_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)
        # Use where to create mask (no in-place ops)
        mask = torch.where(upper_mask.unsqueeze(0).unsqueeze(0), 
                          torch.tensor(float("-inf"), dtype=dtype, device=device),
                          torch.tensor(0.0, dtype=dtype, device=device))
        return mask
 
    # ── Helper to create padding mask without in-place ops ────────────────────
    def _make_padding_mask(self, seq_len: int, kv_len: int, dtype: torch.dtype, device):
        """Create padding mask without in-place operations."""
        half = kv_len // 2
        # Create boolean mask for second half
        pad_mask_bool = torch.zeros(1, 1, seq_len, kv_len, dtype=torch.bool, device=device)
        # Use slicing (returns a view, then assign with where)
        pad_mask_bool[:, :, :, half:] = True
        # Apply -inf using where
        mask = torch.where(pad_mask_bool,
                          torch.tensor(float("-inf"), dtype=dtype, device=device),
                          torch.tensor(0.0, dtype=dtype, device=device))
        return mask
 
    # ── Base test methods ─────────────────────────────────────────────────────
 
    def test_sdpa_prefill_causal(self, q, k, v, compiled):
        """Prefill: is_causal=True — CPU vs Spyre."""
        seq_q = q.shape[2]
        # Slice KV to match Q sequence length and make contiguous
        k_sliced = k[:, :, :seq_q, :].contiguous()
        v_sliced = v[:, :, :seq_q, :].contiguous()
        
        def fn(q, k, v):
            return sdpa_fn(q, k, v, is_causal=True)
        
        # Use appropriate tolerances based on compilation mode
        atol = self.SDPA_COMPILED_ATOL if compiled else self.SDPA_EAGER_ATOL
        rtol = self.SDPA_COMPILED_RTOL if compiled else self.SDPA_EAGER_RTOL
        
        compare_with_cpu(fn, q, k_sliced, v_sliced, compiled=compiled, 
                        atol=atol, rtol=rtol)
 
    def test_sdpa_decode(self, q, k, v, compiled):
        """Decode (seq_q=1): no mask — CPU vs Spyre."""
        def fn(q, k, v):
            return sdpa_fn(q, k, v, is_causal=False)
        
        atol = self.SDPA_COMPILED_ATOL if compiled else self.SDPA_EAGER_ATOL
        rtol = self.SDPA_COMPILED_RTOL if compiled else self.SDPA_EAGER_RTOL
        
        compare_with_cpu(fn, q, k, v, compiled=compiled, atol=atol, rtol=rtol)
 
    def test_sdpa_growing_kvcache(self, q, k, v, compiled):
        """Growing KV cache test for decode with adjusted tolerances."""
        def fn(q, k, v):
            return sdpa_fn(q, k, v, is_causal=False)
        
        # More lenient tolerances for larger KV caches
        atol = 3e-2 if compiled else 2e-2
        rtol = 3e-2 if compiled else 2e-2
        
        compare_with_cpu(fn, q, k, v, compiled=compiled, atol=atol, rtol=rtol)
 
    def test_sdpa_causal_flag_vs_mask(self, q, k, v, compiled):
        """is_causal=True flag and explicit causal mask must agree."""
        seq_q = q.shape[2]
        # Slice KV to match Q sequence length
        k_sliced = k[:, :, :seq_q, :].contiguous()
        v_sliced = v[:, :, :seq_q, :].contiguous()
        
        def fn(q, k, v):
            out_flag = sdpa_fn(q, k, v, is_causal=True)
            mask = self._make_causal_mask(seq_q, q.dtype, q.device)
            out_mask = sdpa_fn(q, k, v, attn_mask=mask, is_causal=False)
            torch.testing.assert_close(
                out_flag, out_mask,
                atol=self.SDPA_EAGER_ATOL, rtol=self.SDPA_EAGER_RTOL,
                msg="is_causal flag vs explicit mask differ",
            )
            return out_flag
        
        atol = self.SDPA_COMPILED_ATOL if compiled else self.SDPA_EAGER_ATOL
        rtol = self.SDPA_COMPILED_RTOL if compiled else self.SDPA_EAGER_RTOL
        
        compare_with_cpu(fn, q, k_sliced, v_sliced, compiled=compiled, 
                        atol=atol, rtol=rtol)
 
    def test_sdpa_weights_sum_to_one(self, q, k, v, compiled):
        """Each attention-weight row must sum to 1 (softmax in fp32)."""
        if compiled:
            self.skipTest("Weights sum to one test only runs in eager mode")
        
        seq_q = q.shape[2]
        k_sliced = k[:, :, :seq_q, :].contiguous()
        v_sliced = v[:, :, :seq_q, :].contiguous()
 
        def fn(q, k, v):
            k_exp, v_exp = expand_kv(k, v)
            scores = torch.matmul(q, k_exp.transpose(-2, -1)) * SCALE
            mask = self._make_causal_mask(seq_q, q.dtype, q.device)
            scores = scores + mask
            weights = torch.softmax(scores.float(), dim=-1)
            row_sums = weights.sum(dim=-1)
            # All rows should sum to 1.0 within tolerance
            torch.testing.assert_close(
                row_sums, torch.ones_like(row_sums),
                atol=1e-3, rtol=1e-3,
                msg="Attention weights do not sum to 1",
            )
            return row_sums
 
        compare_with_cpu(fn, q, k_sliced, v_sliced, compiled=False,
                        atol=self.SDPA_EAGER_ATOL, rtol=self.SDPA_EAGER_RTOL)
 
    def test_sdpa_gqa_shape(self, q, k, v, compiled):
        """After GQA head expansion K and V shapes must have correct heads."""
        seq_q = q.shape[2]
        k_sliced = k[:, :, :seq_q, :].contiguous()
        v_sliced = v[:, :, :seq_q, :].contiguous()
        
        def fn(q, k, v):
            k_exp, v_exp = expand_kv(k, v)
            # Check heads dimension
            assert k_exp.shape[1] == NUM_Q_HEADS, (
                f"K heads mismatch: {k_exp.shape[1]} != {NUM_Q_HEADS}"
            )
            assert v_exp.shape[1] == NUM_Q_HEADS, (
                f"V heads mismatch: {v_exp.shape[1]} != {NUM_Q_HEADS}"
            )
            # Check batch and head dim match
            assert k_exp.shape[0] == q.shape[0], "Batch mismatch"
            assert k_exp.shape[3] == q.shape[3], "Head dim mismatch"
            return sdpa_fn(q, k, v, is_causal=True)
        
        atol = self.SDPA_COMPILED_ATOL if compiled else self.SDPA_EAGER_ATOL
        rtol = self.SDPA_COMPILED_RTOL if compiled else self.SDPA_EAGER_RTOL
        
        compare_with_cpu(fn, q, k_sliced, v_sliced, compiled=compiled,
                        atol=atol, rtol=rtol)
 
    def test_sdpa_batch_consistency(self, q, k, v, compiled):
        """All batch items must produce identical outputs."""
        tol = TOLERANCES[q.dtype]
        B = q.shape[0]
        seq_q = q.shape[2]
        
        # Slice KV to match Q sequence length
        k_sliced = k[:, :, :seq_q, :].contiguous()
        v_sliced = v[:, :, :seq_q, :].contiguous()
 
        def fn(q, k, v):
            out = sdpa_fn(q, k, v, is_causal=True)
            for i in range(1, B):
                torch.testing.assert_close(
                    out[0], out[i],
                    atol=self.SDPA_EAGER_ATOL, rtol=self.SDPA_EAGER_RTOL,
                    msg=f"Batch item {i} differs from item 0",
                )
            return out
        
        atol = self.SDPA_COMPILED_ATOL if compiled else self.SDPA_EAGER_ATOL
        rtol = self.SDPA_COMPILED_RTOL if compiled else self.SDPA_EAGER_RTOL
 
        compare_with_cpu(fn, q, k_sliced, v_sliced, compiled=compiled,
                        atol=atol, rtol=rtol)
 
    def test_sdpa_gradient_flow(self, q, k, v, compiled):
        """Q gradient back-propagates through SDPA without NaNs (eager only)."""
        if compiled:
            self.skipTest("Gradient test only runs in eager mode")
        
        seq_q = q.shape[2]
        k_sliced = k[:, :, :seq_q, :].contiguous()
        v_sliced = v[:, :, :seq_q, :].contiguous()
        
        # Create a fresh tensor with requires_grad=True
        q_grad = q.clone().detach().requires_grad_(True)
        k_grad = k_sliced.clone().detach()
        v_grad = v_sliced.clone().detach()
        
        # Run forward pass
        out = sdpa_fn(q_grad, k_grad, v_grad, is_causal=True)
        out.sum().backward()
        
        # Check gradients
        assert q_grad.grad is not None, "Gradient for Q is None"
        assert not torch.isnan(q_grad.grad).any(), "NaN in Q gradient"
        assert q_grad.grad.abs().max() > 0, "Q gradient is all zeros"
        
        # Compare with CPU reference
        q_cpu = q.clone().detach().requires_grad_(True)
        k_cpu = k_sliced.clone().detach()
        v_cpu = v_sliced.clone().detach()
        
        out_cpu = sdpa_fn(q_cpu, k_cpu, v_cpu, is_causal=True)
        out_cpu.sum().backward()
        
        # Check CPU gradient for reference
        assert q_cpu.grad is not None, "CPU gradient for Q is None"
        
        # Compare gradients between CPU and Spyre (with tolerance)
        torch.testing.assert_close(
            q_grad.grad.cpu(), q_cpu.grad,
            atol=self.SDPA_EAGER_ATOL, rtol=self.SDPA_EAGER_RTOL,
            msg="Q gradient mismatch between CPU and Spyre",
        )
 
    def test_sdpa_determinism(self, q, k, v, compiled):
        """Two consecutive identical SDPA calls must return the same output."""
        if compiled:
            self.skipTest("Determinism test only runs in eager mode")
        
        seq_q = q.shape[2]
        if seq_q > 1:
            k = k[:, :, :seq_q, :].contiguous()
            v = v[:, :, :seq_q, :].contiguous()
 
        def fn(q, k, v):
            out1 = sdpa_fn(q, k, v, is_causal=(seq_q > 1))
            out2 = sdpa_fn(q, k, v, is_causal=(seq_q > 1))
            # Use strict tolerance for determinism
            torch.testing.assert_close(
                out1, out2,
                atol=1e-5, rtol=1e-5,
                msg="Two identical SDPA calls returned different results",
            )
            return out1
 
        compare_with_cpu(fn, q, k, v, compiled=False,
                        atol=self.SDPA_EAGER_ATOL, rtol=self.SDPA_EAGER_RTOL)  
# ═════════════════════════════════════════════════════════════════════════════
#  TestFunctionalSilu  —  torch.nn.functional.silu
# ═════════════════════════════════════════════════════════════════════════════

class TestFunctionalSilu(unittest.TestCase, metaclass=ParameterizedTestMeta):
    pytestmark = pytest.mark.torch_nn_functional_silu
    torch.manual_seed(0xCAFE_2506)
    
    PARAMS = {
        ("test_silu_A000", "_run_silu_ffn_gate_test"): {
            "param_sets": {
                "decode_1x1x32768_eager":    (make_strided_tensor((1, 1, 32768), (32768, 32768, 1), F16), False),
                "decode_1x1x32768_compiled": (make_strided_tensor((1, 1, 32768), (32768, 32768, 1), F16), True),
                "decode_batch2_2x1x32768_eager": (make_strided_tensor((2, 1, 32768), (32768, 32768, 1), F16), False),
                "decode_batch4_4x1x32768_eager": (make_strided_tensor((4, 1, 32768), (32768, 32768, 1), F16), False),
            }
        },
        
        ("test_silu_B000", "_run_silu_ffn_gate_test"): {
            "param_sets": {
                "prefill_1x38x32768_eager":    (make_strided_tensor((1, 38, 32768), (1245184, 32768, 1), F16), False),
                "prefill_1x38x32768_compiled": (make_strided_tensor((1, 38, 32768), (1245184, 32768, 1), F16), True),
                "prefill_1x128x32768_eager":   (make_strided_tensor((1, 128, 32768), (4194304, 32768, 1), F16), False),
                "decode_batch2_strided_eager": (make_strided_tensor((2, 38, 32768), (2490368, 32768, 1), F16), False),
            }
        },
        
        ("test_silu_C000", "_run_silu_ffn_gate_test"): {
            "param_sets": {
                "hidden_decode_1x1x5120_eager":    (make_strided_tensor((1, 1, 5120), (5120, 5120, 1), F16), False),
                "hidden_decode_1x1x5120_compiled": (make_strided_tensor((1, 1, 5120), (5120, 5120, 1), F16), True),
                "hidden_prefill_1x38x5120_eager":  (make_strided_tensor((1, 38, 5120), (194560, 5120, 1), F16), False),
            }
        },
        
        ("test_silu_D000", "_run_silu_swiglu_test"): {
            "param_sets": {
                "swiglu_decode_1x1x32768_eager": (
                    make_strided_tensor((1, 1, 32768), (32768, 32768, 1), F16),
                    make_strided_tensor((1, 1, 32768), (32768, 32768, 1), F16),
                    False,
                ),
                "swiglu_decode_1x1x32768_compiled": (
                    make_strided_tensor((1, 1, 32768), (32768, 32768, 1), F16),
                    make_strided_tensor((1, 1, 32768), (32768, 32768, 1), F16),
                    True,
                ),
                "swiglu_prefill_1x38x32768_eager": (
                    make_strided_tensor((1, 38, 32768), (1245184, 32768, 1), F16),
                    make_strided_tensor((1, 38, 32768), (1245184, 32768, 1), F16),
                    False,
                ),
            }
        },
        
        ("test_silu_E000", "_run_silu_ffn_gate_test"): {
            "param_sets": {
                "bf16_decode_1x1x32768_eager":    (make_strided_tensor((1, 1, 32768), (32768, 32768, 1), BF16), False),
                "fp16_decode_1x1x32768_eager":    (make_strided_tensor((1, 1, 32768), (32768, 32768, 1), F16), False),
                "fp32_decode_1x1x32768_eager":    (make_strided_tensor((1, 1, 32768), (32768, 32768, 1), F32), False),
                "bf16_decode_1x1x32768_compiled": (make_strided_tensor((1, 1, 32768), (32768, 32768, 1), BF16), True),
            }
        },
        
        ("test_silu_F000", "_run_silu_noncontig_test"): {
            "param_sets": {
                "noncontig_gate_decode_eager":    (make_strided_tensor((1, 32768, 1), (32768, 1, 32768), F16), 0, 1, False),
                "noncontig_gate_decode_compiled": (make_strided_tensor((1, 32768, 1), (32768, 1, 32768), F16), 0, 1, True),
                "noncontig_hidden_prefill_eager": (make_strided_tensor((1, 5120, 38), (194560, 38, 1), F16), 1, 2, False),
            }
        },
    }
    
    def _run_silu_ffn_gate_test(self, tensor, compiled):
        compare_with_cpu(lambda t: torch.nn.functional.silu(t), tensor, compiled=compiled)
    
    def _run_silu_swiglu_test(self, gate, up, compiled):
        compare_with_cpu(lambda g, u: torch.nn.functional.silu(g) * u, gate, up, compiled=compiled)
    
    def _run_silu_noncontig_test(self, tensor, d0, d1, compiled):
        compare_with_cpu(lambda t: torch.nn.functional.silu(t.transpose(d0, d1).contiguous()), tensor, compiled=compiled)
    
    def test_silu_zero_fixed_point(self):
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            assert torch.nn.functional.silu(torch.zeros(1, dtype=dtype)).item() == 0.0


# ═════════════════════════════════════════════════════════════════════════════
#  TestPow  —  torch.pow
# ═════════════════════════════════════════════════════════════════════════════

class TestPow(unittest.TestCase, metaclass=ParameterizedTestMeta):
    pytestmark = pytest.mark.torch_pow
    torch.manual_seed(0)
    
    PARAMS = {
        ("test_torch_pow_prefill", "_run_pow_test"): {
            "param_sets": {
                "prefill_exp2_eager": (make_strided_tensor((1, 38, 5120), (194560, 5120, 1), F32), 2, False),
                "prefill_exp2_compiled": (make_strided_tensor((1, 38, 5120), (194560, 5120, 1), F32), 2, True),
                "prefill_exp2_batch_eager": (make_strided_tensor((2, 38, 5120), (389120, 5120, 1), F32), 2, False),
            }
        },
        
        ("test_torch_pow_decode", "_run_pow_test"): {
            "param_sets": {
                "decode_exp2_eager": (make_strided_tensor((1, 1, 5120), (5120, 5120, 1), F32), 2, False),
                "decode_exp2_compiled": (make_strided_tensor((1, 1, 5120), (5120, 5120, 1), F32), 2, True),
            }
        },
        
        ("test_torch_pow_fractional", "_run_pow_test"): {
            "param_sets": {
                "prefill_exp0p5_eager": (make_strided_tensor((1, 38, 5120), (194560, 5120, 1), F32, fill="ones"), 0.5, False),
                "prefill_exp0p5_compiled": (make_strided_tensor((1, 38, 5120), (194560, 5120, 1), F32, fill="ones"), 0.5, True),
                "decode_exp0p5_eager": (make_strided_tensor((1, 1, 5120), (5120, 5120, 1), F32, fill="ones"), 0.5, False),
            }
        },
    }
    
    def _run_pow_test(self, x, exp, compiled):
        compare_with_cpu(lambda t: torch.pow(t, exp), x, compiled=compiled)


# ═════════════════════════════════════════════════════════════════════════════
#  TestRsqrt  —  torch.rsqrt
# ═════════════════════════════════════════════════════════════════════════════

class TestRsqrt(unittest.TestCase, metaclass=ParameterizedTestMeta):
    pytestmark = pytest.mark.torch_rsqrt
    torch.manual_seed(0)
    
    _EPS = 1e-5
    
    PARAMS = {
        ("test_torch_rsqrt_prefill", "_run_rsqrt_test"): {
            "param_sets": {
                "prefill_eager": (make_strided_tensor((1, 38, 1), (38, 1, 1), F32, fill="ones"), False),
                "prefill_compiled": (make_strided_tensor((1, 38, 1), (38, 1, 1), F32, fill="ones"), True),
                "prefill_batch2_eager": (make_strided_tensor((2, 38, 1), (76, 1, 1), F32, fill="ones"), False),
            }
        },
        
        ("test_torch_rsqrt_decode", "_run_rsqrt_test"): {
            "param_sets": {
                "decode_eager": (make_strided_tensor((1, 1, 1), (1, 1, 1), F32, fill="ones"), False),
                "decode_compiled": (make_strided_tensor((1, 1, 1), (1, 1, 1), F32, fill="ones"), True),
            }
        },
    }
    
    def _run_rsqrt_test(self, x, compiled):
        compare_with_cpu(lambda t: torch.rsqrt(t + self._EPS), x, compiled=compiled)


# ═════════════════════════════════════════════════════════════════════════════
#  TestCumsum  —  torch.cumsum
# ═════════════════════════════════════════════════════════════════════════════

class TestCumsum(unittest.TestCase, metaclass=ParameterizedTestMeta):
    pytestmark = pytest.mark.torch_cumsum
    torch.manual_seed(0)
    
    PARAMS = {
        ("test_torch_cumsum_bool", "_run_cumsum_test"): {
            "param_sets": {
                "prefill_zeros_eager": (make_strided_tensor((1, 38), (38, 1), torch.bool, fill="zeros"), -1, False),
                "prefill_zeros_compiled": (make_strided_tensor((1, 38), (38, 1), torch.bool, fill="zeros"), -1, True),
                "prefill_ones_eager": (make_strided_tensor((1, 38), (38, 1), torch.bool, fill="ones"), -1, False),
                "prefill_ones_compiled": (make_strided_tensor((1, 38), (38, 1), torch.bool, fill="ones"), -1, True),
                "prefill_mixed_eager": (
                    torch.tensor([[False]*10 + [True] + [False]*14 + [True] + [False]*12], dtype=torch.bool),
                    -1, False,
                ),
            }
        },
        
        ("test_torch_cumsum_int64", "_run_cumsum_test"): {
            "param_sets": {
                "int64_prefill_eager": (make_strided_tensor((1, 38), (38, 1), I64, fill="arange"), -1, False),
                "int64_prefill_compiled": (make_strided_tensor((1, 38), (38, 1), I64, fill="arange"), -1, True),
            }
        },
        
    }
    
    def _run_cumsum_test(self, x, dim, compiled):
        compare_with_cpu(lambda t: torch.cumsum(t, dim), x, compiled=compiled)


# ═════════════════════════════════════════════════════════════════════════════
#  TestTranspose  —  torch.transpose
# ═════════════════════════════════════════════════════════════════════════════

class TestTranspose(unittest.TestCase, metaclass=ParameterizedTestMeta):
    pytestmark = pytest.mark.torch_transpose
    torch.manual_seed(0)
    
    _prefill_q = make_strided_tensor((1, 38, 32, 128), (155648, 4096, 128, 1), F16)
    _prefill_k = make_strided_tensor((1, 38, 8, 128), (38912, 1024, 128, 1), F16)
    _prefill_post = make_strided_tensor((1, 32, 38, 128), (155648, 128, 4096, 1), F16)
    _decode_q = make_strided_tensor((1, 1, 32, 128), (4096, 4096, 128, 1), F16)
    _decode_k = make_strided_tensor((1, 1, 8, 128), (1024, 1024, 128, 1), F16)
    _decode_post = make_strided_tensor((1, 32, 1, 128), (4096, 128, 128, 1), F16)
    _rope_3d = make_strided_tensor((1, 64, 38), (2432, 38, 1), F32)
    
    PARAMS = {
        ("test_transpose_core_patterns", "_run_transpose_core"): {
            "param_sets": {
                "prefill_q_d12": (_prefill_q, 1, 2, False),
                "prefill_q_d12_comp": (_prefill_q, 1, 2, True),
                "prefill_q_d23": (_prefill_q, 2, 3, False),
                "prefill_k_d12": (_prefill_k, 1, 2, False),
                "prefill_post_d12": (_prefill_post, 1, 2, False),
                "decode_q_d12": (_decode_q, 1, 2, False),
                "decode_q_d12_comp": (_decode_q, 1, 2, True),
                "decode_k_d12": (_decode_k, 1, 2, False),
                "decode_post_d12": (_decode_post, 1, 2, False),
                "rope_3d_d12": (_rope_3d, 1, 2, False),
                "rope_3d_d12_comp": (_rope_3d, 1, 2, True),
            }
        },
        
        ("test_transpose_neg_dims", "_run_transpose_neg"): {
            "param_sets": {
                "prefill_q_neg": (_prefill_q, -2, -1, 2, 3, False),
                "prefill_q_neg_comp": (_prefill_q, -2, -1, 2, 3, True),
                "prefill_post_neg": (_prefill_post, -3, -1, 1, 3, False),
                "decode_post_neg": (_decode_post, -3, -1, 1, 3, False),
                "rope_3d_neg": (_rope_3d, -2, -1, 1, 2, False),
            }
        },
        
        ("test_transpose_dtype", "_run_transpose_dtype"): {
            "param_sets": {
                "prefill_q_dtype": (_prefill_q, 1, 2, False),
                "prefill_q_dtype_comp": (_prefill_q, 1, 2, True),
                "decode_q_dtype": (_decode_q, 1, 2, False),
                "rope_3d_dtype": (_rope_3d, 1, 2, False),
            }
        },
    }
    
    def _run_transpose_core(self, x, d0, d1, compiled):
        expected = list(x.shape)
        d0n, d1n = d0 % x.ndim, d1 % x.ndim
        expected[d0n], expected[d1n] = expected[d1n], expected[d0n]
        
        def fn(t):
            out = torch.transpose(t, d0, d1).contiguous()
            assert list(out.shape) == expected
            return out
        compare_with_cpu(fn, x, compiled=compiled)
    
    def _run_transpose_neg(self, x, neg0, neg1, pos0, pos1, compiled):
        def fn(t):
            a = torch.transpose(t, neg0, neg1).contiguous()
            b = torch.transpose(t, pos0, pos1).contiguous()
            torch.testing.assert_close(a, b)
            return a
        compare_with_cpu(fn, x, compiled=compiled)
    
    def _run_transpose_dtype(self, x, d0, d1, compiled):
        def fn(t):
            out = torch.transpose(t, d0, d1).contiguous()
            assert out.dtype == t.dtype
            return out
        compare_with_cpu(fn, x, compiled=compiled)

# ═════════════════════════════════════════════════════════════════════════════
#  TestAdd  —  torch.add (corrected with proper sequence lengths)
# ═════════════════════════════════════════════════════════════════════════════

class TestAdd(unittest.TestCase, metaclass=ParameterizedTestMeta):
    pytestmark = pytest.mark.torch_add
    torch.manual_seed(0)
    
    # Use traced sequence lengths: 37 for rotary patterns, 38 for residual
    PREFILL_SEQ_TRACED = 37
    PREFILL_SEQ_FULL = 38
    DECODE_SEQ = 1
    
    PARAMS = {
        # ── prefill Q rotary (traced with seq=37) ─────────────────────────────
        ("test_add_binary_prefill_q", "_run_add_binary"): {
            "param_sets": {
                "prefill_q_eager": (
                    make_strided_tensor((1, 32, 37, 128), (151552, 128, 4096, 1), F16),
                    make_strided_tensor((1, 32, 37, 128), (151552, 128, 4096, 1), F16),
                    False,
                ),
                "prefill_q_compiled": (
                    make_strided_tensor((1, 32, 37, 128), (151552, 128, 4096, 1), F16),
                    make_strided_tensor((1, 32, 37, 128), (151552, 128, 4096, 1), F16),
                    True,
                ),
            }
        },
        
        # ── prefill K rotary (traced with seq=37) ─────────────────────────────
        ("test_add_binary_prefill_k", "_run_add_binary"): {
            "param_sets": {
                "prefill_k_eager": (
                    make_strided_tensor((1, 8, 37, 128), (37888, 128, 1024, 1), F16),
                    make_strided_tensor((1, 8, 37, 128), (37888, 128, 1024, 1), F16),
                    False,
                ),
                "prefill_k_compiled": (
                    make_strided_tensor((1, 8, 37, 128), (37888, 128, 1024, 1), F16),
                    make_strided_tensor((1, 8, 37, 128), (37888, 128, 1024, 1), F16),
                    True,
                ),
            }
        },
        
        # ── decode Q rotary ───────────────────────────────────────────────────
        ("test_add_binary_decode_q", "_run_add_binary"): {
            "param_sets": {
                "decode_q_eager": (
                    make_strided_tensor((1, 32, 1, 128), (4096, 128, 4096, 1), F16),
                    make_strided_tensor((1, 32, 1, 128), (4096, 128, 4096, 1), F16),
                    False,
                ),
                "decode_q_compiled": (
                    make_strided_tensor((1, 32, 1, 128), (4096, 128, 4096, 1), F16),
                    make_strided_tensor((1, 32, 1, 128), (4096, 128, 4096, 1), F16),
                    True,
                ),
            }
        },
        
        # ── decode K rotary ───────────────────────────────────────────────────
        ("test_add_binary_decode_k", "_run_add_binary"): {
            "param_sets": {
                "decode_k_eager": (
                    make_strided_tensor((1, 8, 1, 128), (1024, 128, 1024, 1), F16),
                    make_strided_tensor((1, 8, 1, 128), (1024, 128, 1024, 1), F16),
                    False,
                ),
                "decode_k_compiled": (
                    make_strided_tensor((1, 8, 1, 128), (1024, 128, 1024, 1), F16),
                    make_strided_tensor((1, 8, 1, 128), (1024, 128, 1024, 1), F16),
                    True,
                ),
            }
        },
        
        # ── residual addition (full sequence 38) ──────────────────────────────
        ("test_add_binary_residual_prefill", "_run_add_binary"): {
            "param_sets": {
                "residual_prefill_eager": (
                    make_strided_tensor((1, 38, 5120), (194560, 5120, 1), F16),
                    make_strided_tensor((1, 38, 5120), (194560, 5120, 1), F16),
                    False,
                ),
                "residual_prefill_compiled": (
                    make_strided_tensor((1, 38, 5120), (194560, 5120, 1), F16),
                    make_strided_tensor((1, 38, 5120), (194560, 5120, 1), F16),
                    True,
                ),
            }
        },
        
        # ── scalar addition for RMSNorm (seq=38) ──────────────────────────────
        ("test_add_scalar_rmsnorm_prefill", "_run_add_scalar"): {
            "param_sets": {
                "rmsnorm_prefill_eager": (
                    make_strided_tensor((1, 38, 1), (38, 1, 1), F32),
                    1e-5, False,
                ),
                "rmsnorm_prefill_compiled": (
                    make_strided_tensor((1, 38, 1), (38, 1, 1), F32),
                    1e-5, True,
                ),
            }
        },
        
        # ── scalar addition for RMSNorm decode ────────────────────────────────
        ("test_add_scalar_rmsnorm_decode", "_run_add_scalar"): {
            "param_sets": {
                "rmsnorm_decode_eager": (
                    make_strided_tensor((1, 1, 1), (1, 1, 1), F32),
                    1e-5, False,
                ),
                "rmsnorm_decode_compiled": (
                    make_strided_tensor((1, 1, 1), (1, 1, 1), F32),
                    1e-5, True,
                ),
            }
        },
        
        # ── scalar left addition for attention scale (seq=38) ─────────────────
        ("test_add_scalar_left_attention", "_run_add_scalar_left"): {
            "param_sets": {
                "attention_prefill_eager": (
                    1,
                    make_strided_tensor((1, 38), (38, 1), F16),
                    False,
                ),
                "attention_prefill_compiled": (
                    1,
                    make_strided_tensor((1, 38), (38, 1), F16),
                    True,
                ),
                "attention_decode_eager": (
                    1,
                    make_strided_tensor((1, 1), (1, 1), F16),
                    False,
                ),
            }
        },
        
        # ── alpha addition ────────────────────────────────────────────────────
        ("test_add_alpha", "_run_add_alpha"): {
            "param_sets": {
                "alpha_2_eager": (
                    make_strided_tensor((1, 32, 37, 128), (151552, 128, 4096, 1), F16),
                    make_strided_tensor((1, 32, 37, 128), (151552, 128, 4096, 1), F16),
                    2.0, False,
                ),
                "alpha_2_compiled": (
                    make_strided_tensor((1, 32, 37, 128), (151552, 128, 4096, 1), F16),
                    make_strided_tensor((1, 32, 37, 128), (151552, 128, 4096, 1), F16),
                    2.0, True,
                ),
            }
        },
        
        # ── in-place addition (full sequence 38) ──────────────────────────────
        ("test_add_inplace", "_run_add_inplace"): {
            "param_sets": {
                "inplace_prefill_eager": (
                    make_strided_tensor((1, 38, 5120), (194560, 5120, 1), F16, fill="zeros"),
                    make_strided_tensor((1, 38, 5120), (194560, 5120, 1), F16),
                    False,
                ),
                "inplace_decode_eager": (
                    make_strided_tensor((1, 1, 5120), (5120, 5120, 1), F16, fill="zeros"),
                    make_strided_tensor((1, 1, 5120), (5120, 5120, 1), F16),
                    False,
                ),
                "inplace_prefill_compiled": (
                    make_strided_tensor((1, 38, 5120), (194560, 5120, 1), F16, fill="zeros"),
                    make_strided_tensor((1, 38, 5120), (194560, 5120, 1), F16),
                    True,
                ),
            }
        },
    }
    
    def _run_add_binary(self, a, b, compiled):
        compare_with_cpu(torch.add, a, b, compiled=compiled)
    
    def _run_add_scalar(self, a, scalar, compiled):
        compare_with_cpu(lambda x: torch.add(x, scalar), a, compiled=compiled)
    
    def _run_add_scalar_left(self, scalar, b, compiled):
        compare_with_cpu(lambda x: torch.add(scalar, x), b, compiled=compiled)
    
    def _run_add_alpha(self, a, b, alpha, compiled):
        compare_with_cpu(lambda x, y: torch.add(x, y, alpha=alpha), a, b, compiled=compiled)
    
    def _run_add_inplace(self, dst, src, compiled):
        def fn(d, s):
            d = d.clone()
            out = d.add_(s)
            assert out.data_ptr() == d.data_ptr()
            return out
        compare_with_cpu(fn, dst, src, compiled=compiled)

# ═════════════════════════════════════════════════════════════════════════════
#  TestSub  —  torch.sub
# ═════════════════════════════════════════════════════════════════════════════

class TestSub(unittest.TestCase, metaclass=ParameterizedTestMeta):
    pytestmark = pytest.mark.torch_sub
    torch.manual_seed(0)
    
    PARAMS = {
        ("test_torch_sub_binary", "_run_sub_binary"): {
            "param_sets": {
                "binary_1x1_int64_eager": (
                    torch.randint(1, 512, (1, 1), dtype=I64),
                    torch.randint(0, 256, (1, 1), dtype=I64),
                    False,
                ),
                "binary_1x1_int64_compiled": (
                    torch.randint(1, 512, (1, 1), dtype=I64),
                    torch.randint(0, 256, (1, 1), dtype=I64),
                    True,
                ),
                "binary_zero_result_eager": (
                    torch.full((1, 1), 42, dtype=I64),
                    torch.full((1, 1), 42, dtype=I64),
                    False,
                ),
                "binary_zero_result_compiled": (
                    torch.full((1, 1), 42, dtype=I64),
                    torch.full((1, 1), 42, dtype=I64),
                    True,
                ),
            }
        },
        
        ("test_torch_sub_scalar", "_run_sub_scalar"): {
            "param_sets": {
                "scalar_1x1_int64_minus1_eager": (
                    torch.randint(1, 512, (1, 1), dtype=I64),
                    1,
                    False,
                ),
                "scalar_1x1_int64_minus1_compiled": (
                    torch.randint(1, 512, (1, 1), dtype=I64),
                    1,
                    True,
                ),
                "scalar_1x1_int64_minus0_eager": (
                    torch.randint(0, 512, (1, 1), dtype=I64),
                    0,
                    False,
                ),
                "scalar_1x1_int64_minus0_compiled": (
                    torch.randint(0, 512, (1, 1), dtype=I64),
                    0,
                    True,
                ),
            }
        },
        
        ("test_torch_sub_scalar_left", "_run_sub_scalar_left"): {
            "param_sets": {
                "scalar_left_512_minus_1x1_eager": (
                    512,
                    torch.randint(0, 512, (1, 1), dtype=I64),
                    False,
                ),
                "scalar_left_512_minus_1x1_compiled": (
                    512,
                    torch.randint(0, 512, (1, 1), dtype=I64),
                    True,
                ),
            }
        },
        
        ("test_torch_sub_alpha", "_run_sub_alpha"): {
            "param_sets": {
                "alpha_2_1x1_int64_eager": (
                    torch.randint(1, 512, (1, 1), dtype=I64),
                    torch.randint(0, 64, (1, 1), dtype=I64),
                    2,
                    False,
                ),
                "alpha_2_1x1_int64_compiled": (
                    torch.randint(1, 512, (1, 1), dtype=I64),
                    torch.randint(0, 64, (1, 1), dtype=I64),
                    2,
                    True,
                ),
                "alpha_0_1x1_int64_eager": (
                    torch.randint(1, 512, (1, 1), dtype=I64),
                    torch.randint(0, 64, (1, 1), dtype=I64),
                    0,
                    False,
                ),
            }
        },
        
        ("test_torch_sub_inplace", "_run_sub_inplace"): {
            "param_sets": {
                "inplace_1x1_int64_eager": (
                    torch.randint(1, 512, (1, 1), dtype=I64),
                    torch.randint(0, 256, (1, 1), dtype=I64),
                    False,
                ),
                "inplace_1x1_int64_compiled": (
                    torch.randint(1, 512, (1, 1), dtype=I64),
                    torch.randint(0, 256, (1, 1), dtype=I64),
                    True,
                ),
            }
        },
    }
    
    def _run_sub_binary(self, a, b, compiled):
        compare_with_cpu(torch.sub, a, b, compiled=compiled)
    
    def _run_sub_scalar(self, a, scalar, compiled):
        compare_with_cpu(lambda x: torch.sub(x, scalar), a, compiled=compiled)
    
    def _run_sub_scalar_left(self, scalar, b, compiled):
        compare_with_cpu(lambda x: torch.sub(scalar, x), b, compiled=compiled)
    
    def _run_sub_alpha(self, a, b, alpha, compiled):
        compare_with_cpu(lambda x, y: torch.sub(x, y, alpha=alpha), a, b, compiled=compiled)
    
    def _run_sub_inplace(self, dst, src, compiled):
        def fn(d, s):
            d = d.clone()
            out = d.sub_(s)
            assert out.data_ptr() == d.data_ptr()
            return out
        compare_with_cpu(fn, dst, src, compiled=compiled)


# ═════════════════════════════════════════════════════════════════════════════
#  TestContiguous  —  torch.Tensor.contiguous
# ═════════════════════════════════════════════════════════════════════════════

class TestContiguous(unittest.TestCase, metaclass=ParameterizedTestMeta):
    pytestmark = pytest.mark.torch_contiguous
    torch.manual_seed(0)
    
    _tensor = make_strided_tensor((1, 38, 32, 128), (155648, 4096, 128, 1), F16)
    
    PARAMS = {
        ("test_contiguous_already", "_run_contiguous_test"): {
            "param_sets": {
                "already_contiguous_eager": (_tensor, False),
                "already_contiguous_compiled": (_tensor, True),
                "decode_contiguous_eager": (make_strided_tensor((1, 1, 32, 128), (4096, 4096, 128, 1), F16), False),
                "decode_contiguous_compiled": (make_strided_tensor((1, 1, 32, 128), (4096, 4096, 128, 1), F16), True),
            }
        },
        
        ("test_contiguous_noncontig", "_run_contiguous_noncontig_test"): {
            "param_sets": {
                "noncontig_prefill_eager": (1, 2, _tensor, False),
                "noncontig_prefill_compiled": (1, 2, _tensor, True),
                "noncontig_decode_eager": (1, 2, make_strided_tensor((1, 1, 32, 128), (4096, 4096, 128, 1), F16), False),
                "noncontig_decode_compiled": (1, 2, make_strided_tensor((1, 1, 32, 128), (4096, 4096, 128, 1), F16), True),
            }
        },
        
        ("test_contiguous_3d", "_run_contiguous_test"): {
            "param_sets": {
                "prefill_3d_eager": (make_strided_tensor((1, 38, 5120), (194560, 5120, 1), F16), False),
                "prefill_3d_compiled": (make_strided_tensor((1, 38, 5120), (194560, 5120, 1), F16), True),
                "decode_3d_eager": (make_strided_tensor((1, 1, 5120), (5120, 5120, 1), F16), False),
            }
        },
    }
    
    def _run_contiguous_test(self, x, compiled):
        def fn(t):
            out = t.contiguous()
            assert out.is_contiguous()
            return out
        compare_with_cpu(fn, x, compiled=compiled)
    
    def _run_contiguous_noncontig_test(self, d0, d1, x, compiled):
        def fn(t):
            view = torch.transpose(t, d0, d1)
            out = view.contiguous()
            assert out.is_contiguous()
            return out
        compare_with_cpu(fn, x, compiled=compiled)


# ═════════════════════════════════════════════════════════════════════════════
#  TestView  —  torch.Tensor.view
# ═════════════════════════════════════════════════════════════════════════════

class TestView(unittest.TestCase, metaclass=ParameterizedTestMeta):
    pytestmark = pytest.mark.torch_view
    torch.manual_seed(0)
    
    PARAMS = {
        ("test_view_q_split", "_run_view_test"): {
            "param_sets": {
                "q_split_prefill_eager": (
                    make_strided_tensor((1, 38, 4096), (155648, 4096, 1), F16),
                    (1, 38, NUM_Q_HEADS, HEAD_DIM),
                    False,
                ),
                "q_split_prefill_compiled": (
                    make_strided_tensor((1, 38, 4096), (155648, 4096, 1), F16),
                    (1, 38, NUM_Q_HEADS, HEAD_DIM),
                    True,
                ),
                "q_split_decode_eager": (
                    make_strided_tensor((1, 1, 4096), (4096, 4096, 1), F16),
                    (1, 1, NUM_Q_HEADS, HEAD_DIM),
                    False,
                ),
                "q_split_decode_compiled": (
                    make_strided_tensor((1, 1, 4096), (4096, 4096, 1), F16),
                    (1, 1, NUM_Q_HEADS, HEAD_DIM),
                    True,
                ),
                "q_split_batch_eager": (
                    make_strided_tensor((4, 38, 4096), (622592, 4096, 1), F16),
                    (4, 38, NUM_Q_HEADS, HEAD_DIM),
                    False,
                ),
            }
        },
        
        ("test_view_attn_merge", "_run_view_test"): {
            "param_sets": {
                "attn_merge_prefill_eager": (
                    make_strided_tensor((1, 38, 32, 128), (155648, 4096, 128, 1), F16),
                    (1, 38, 4096),
                    False,
                ),
                "attn_merge_prefill_compiled": (
                    make_strided_tensor((1, 38, 32, 128), (155648, 4096, 128, 1), F16),
                    (1, 38, 4096),
                    True,
                ),
                "attn_merge_decode_eager": (
                    make_strided_tensor((1, 1, 32, 128), (4096, 4096, 128, 1), F16),
                    (1, 1, 4096),
                    False,
                ),
                "attn_merge_decode_compiled": (
                    make_strided_tensor((1, 1, 32, 128), (4096, 4096, 128, 1), F16),
                    (1, 1, 4096),
                    True,
                ),
            }
        },
        
        ("test_view_mlp_flatten", "_run_view_test"): {
            "param_sets": {
                "mlp_flatten_prefill_eager": (
                    make_strided_tensor((1, 38, 5120), (194560, 5120, 1), F16),
                    (38, 5120),
                    False,
                ),
                "mlp_flatten_prefill_compiled": (
                    make_strided_tensor((1, 38, 5120), (194560, 5120, 1), F16),
                    (38, 5120),
                    True,
                ),
                "mlp_flatten_batch_eager": (
                    make_strided_tensor((4, 38, 5120), (778240, 5120, 1), F16),
                    (152, 5120),
                    False,
                ),
                "mlp_flatten_decode_eager": (
                    make_strided_tensor((1, 1, 5120), (5120, 5120, 1), F16),
                    (1, 5120),
                    False,
                ),
            }
        },
        
        ("test_view_mlp_unflatten", "_run_view_test"): {
            "param_sets": {
                "mlp_unflatten_prefill_eager": (
                    make_strided_tensor((38, 5120), (5120, 1), F16),
                    (1, 38, 5120),
                    False,
                ),
                "mlp_unflatten_prefill_compiled": (
                    make_strided_tensor((38, 5120), (5120, 1), F16),
                    (1, 38, 5120),
                    True,
                ),
                "mlp_unflatten_batch_eager": (
                    make_strided_tensor((152, 5120), (5120, 1), F16),
                    (4, 38, 5120),
                    False,
                ),
            }
        },
        
        ("test_view_lm_head", "_run_view_test"): {
            "param_sets": {
                "lm_head_flatten_eager": (
                    make_strided_tensor((1, 38, 5120), (194560, 5120, 1), F16),
                    (-1, 5120),
                    False,
                ),
                "lm_head_flatten_compiled": (
                    make_strided_tensor((1, 38, 5120), (194560, 5120, 1), F16),
                    (-1, 5120),
                    True,
                ),
            }
        },
    }
    def _run_view_test(self, tensor, view_shape, compiled):
        def view_fn(t):
            result = t.reshape(*view_shape)
            assert result.numel() == t.numel()
            return result

        compare_with_cpu(view_fn, tensor, compiled=compiled)

# ═════════════════════════════════════════════════════════════════════════════
#  TestTo  —  Tensor.to
# ═════════════════════════════════════════════════════════════════════════════

class TestTo(unittest.TestCase, metaclass=ParameterizedTestMeta):
    pytestmark = pytest.mark.torch_to
    torch.manual_seed(0)
    
    _tensor_f16 = make_strided_tensor((1, 38, 5120), (194560, 5120, 1), F16)
    _tensor_f32 = make_strided_tensor((1, 38, 5120), (194560, 5120, 1), F32)
    _tensor_bf16 = make_strided_tensor((1, 38, 5120), (194560, 5120, 1), F16)
    _tensor_int64 = make_strided_tensor((1, 38), (38, 1), I64, fill="arange")
    _tensor_bool = make_strided_tensor((1, 38, 128, 128), (622592, 16384, 128, 1), torch.bool, fill="ones")
    
    PARAMS = {
        ("test_to_dtype", "_run_to_test"): {
            "param_sets": {
                "fp16_to_fp32_eager": (_tensor_f16, torch.float32, None, False),
                "fp16_to_fp32_compiled": (_tensor_f16, torch.float32, None, True),
                "fp32_to_fp16_eager": (_tensor_f32, torch.float16, None, False),
                "fp32_to_fp16_compiled": (_tensor_f32, torch.float16, None, True),
                "fp32_to_bf16_eager": (_tensor_f32, torch.bfloat16, None, False),
                "fp32_to_bf16_compiled": (_tensor_f32, torch.bfloat16, None, True),
                "bf16_to_fp32_eager": (_tensor_bf16, torch.float32, None, False),
                "int32_to_int64_eager": (torch.randint(0, 1000, (1, 38), dtype=torch.int32), torch.int64, None, False),
                "bool_to_fp16_eager": (_tensor_bool, torch.float16, None, False),
                "bool_to_fp32_eager": (_tensor_bool, torch.float32, None, False),
            }
        },
        
        ("test_to_device", "_run_to_test"): {
            "param_sets": {
                "activation_to_cpu_eager": (_tensor_f16, None, "cpu", False),
                "activation_to_cpu_compiled": (_tensor_f16, None, "cpu", True),
                "kv_to_cpu_eager": (make_strided_tensor((1, 8, 38, 128), (38912, 4864, 128, 1), BF16), None, "cpu", False),
                "ids_to_cpu_eager": (_tensor_int64, None, "cpu", False),
                "pos_to_cpu_eager": (torch.arange(38, dtype=I64).unsqueeze(0), None, "cpu", False),
            }
        },
        
        ("test_to_combined", "_run_to_test"): {
            "param_sets": {
                "combined_fp32_to_fp16_cpu_eager": (_tensor_f32, torch.float16, "cpu", False),
                "combined_fp32_to_fp16_cpu_compiled": (_tensor_f32, torch.float16, "cpu", True),
                "combined_fp32_to_bf16_cpu_eager": (_tensor_f32, torch.bfloat16, "cpu", False),
                "combined_bf16_to_fp32_cpu_eager": (_tensor_bf16, torch.float32, "cpu", False),
            }
        },
    }
    
    def _run_to_test(self, tensor, dtype, device, compiled):
        def to_fn(t):
            kwargs = {}
            if dtype is not None:
                kwargs["dtype"] = dtype
            if device is not None:
                kwargs["device"] = device
            return t.to(**kwargs)
        
        compare_with_cpu(to_fn, tensor, compiled=compiled)


# ═════════════════════════════════════════════════════════════════════════════
#  TestDiff  —  torch.diff
# ═════════════════════════════════════════════════════════════════════════════

class TestDiff(unittest.TestCase, metaclass=ParameterizedTestMeta):
    pytestmark = pytest.mark.torch_diff
    torch.manual_seed(0)
    
    PARAMS = {
        ("test_diff_1d", "_run_diff_test"): {
            "param_sets": {
                "image_positions_eager": (
                    torch.tensor([12, 13, 14, 15, 50, 51, 52, 53], dtype=I64),
                    1, -1, False,
                ),
                "image_positions_compiled": (
                    torch.tensor([12, 13, 14, 15, 50, 51, 52, 53], dtype=I64),
                    1, -1, True,
                ),
                "long_positions_eager": (torch.arange(0, 128, 2, dtype=I64), 1, -1, False),
                "float_seq_eager": (torch.randn(128, dtype=F32).cumsum(0), 1, -1, False),
            }
        },
        
        ("test_diff_position_ids", "_run_diff_test"): {
            "param_sets": {
                "pos_ids_single_eager": (torch.arange(38, dtype=I64).unsqueeze(0), 1, -1, False),
                "pos_ids_single_compiled": (torch.arange(38, dtype=I64).unsqueeze(0), 1, -1, True),
                "pos_ids_batch_eager": (torch.arange(128, dtype=I64).unsqueeze(0).expand(4, -1), 1, -1, False),
                "pos_gap_eager": (torch.cat([torch.arange(64, dtype=I64), torch.arange(128, 192, dtype=I64)]).unsqueeze(0), 1, -1, False),
            }
        },
        
        ("test_diff_second_order", "_run_diff_test"): {
            "param_sets": {
                "second_order_eager": (make_strided_tensor((128,), (1,), F32), 2, -1, False),
                "second_order_compiled": (make_strided_tensor((128,), (1,), F32), 2, -1, True),
            }
        },
        
        ("test_diff_cumsum", "_run_diff_test"): {
            "param_sets": {
                "cumsum_lengths_eager": (torch.tensor([10, 30, 60, 100, 150], dtype=I64), 1, -1, False),
                "batch_cumsum_eager": (torch.randint(1, 50, (4, 8), dtype=I64).cumsum(-1), 1, -1, False),
            }
        },
        
        ("test_diff_padded", "_run_diff_test"): {
            "param_sets": {
                "padded_pos_ids_eager": (
                    torch.cat([torch.arange(128, dtype=I64), torch.zeros(128, dtype=I64)]).unsqueeze(0),
                    1, -1, False,
                ),
            }
        },
        
    }
    
    def _run_diff_test(self, tensor, n, dim, compiled):
        compare_with_cpu(lambda t: torch.diff(t, n=n, dim=dim), tensor, compiled=compiled)

# ═════════════════════════════════════════════════════════════════════════════
#  TestAll  —  torch.all
# ═════════════════════════════════════════════════════════════════════════════

class TestAll(unittest.TestCase, metaclass=ParameterizedTestMeta):
    pytestmark = pytest.mark.torch_all
    torch.manual_seed(0)
    
    PREFILL_SEQ = 38
    
    PARAMS = {
        ("test_all_global", "_run_all_test"): {
            "param_sets": {
                "eos_all_done_eager": (torch.zeros(4, dtype=torch.int32), None, False),
                "eos_all_done_compiled": (torch.zeros(4, dtype=torch.int32), None, True),
                "eos_partial_eager": (torch.tensor([0, 1, 0, 0], dtype=torch.int32), None, False),
                "eos_partial_compiled": (torch.tensor([0, 1, 0, 0], dtype=torch.int32), None, True),
                "cache_valid_eager": (torch.randint(1, 512, (NUM_KV_HEADS,), dtype=I64), None, False),
                "image_mask_full_eager": (torch.ones(64, dtype=torch.bool), None, False),
            }
        },
        
        ("test_all_dim", "_run_all_test"): {
            "param_sets": {
                "mask_rowcheck_single_eager": (torch.ones(1, 38, dtype=torch.int32), -1, False),
                "mask_rowcheck_single_compiled": (torch.ones(1, 38, dtype=torch.int32), -1, True),
                "mask_rowcheck_batch_eager": (
                    torch.cat([
                        torch.ones(2, 128, dtype=torch.int32),
                        torch.cat([torch.ones(2, 64, dtype=torch.int32), torch.zeros(2, 64, dtype=torch.int32)], dim=1),
                    ], dim=0),
                    -1, False,
                ),
                "mask_rowcheck_batch_compiled": (
                    torch.cat([
                        torch.ones(2, 128, dtype=torch.int32),
                        torch.cat([torch.ones(2, 64, dtype=torch.int32), torch.zeros(2, 64, dtype=torch.int32)], dim=1),
                    ], dim=0),
                    -1, True,
                ),
                "image_token_present_eager": (torch.ones(4, 64, dtype=torch.bool), -1, False),
                "col_valid_eager": (torch.ones(4, 128, dtype=torch.int32), 0, False),
                "head_kv_valid_eager": (torch.ones(NUM_KV_HEADS, 128, dtype=torch.bool), 1, False),
            }
        },
        
        ("test_all_bool", "_run_all_test"): {
            "param_sets": {
                "bool_global_eager": (torch.ones(64, dtype=torch.bool), None, False),
                "bool_global_compiled": (torch.ones(64, dtype=torch.bool), None, True),
            }
        },
        
        ("test_all_padded", "_run_all_test"): {
            "param_sets": {
                "padded_mask_global_eager": (
                    torch.cat([torch.ones(128, dtype=torch.int32), torch.zeros(128, dtype=torch.int32)]),
                    None, False,
                ),
                "large_batch_eos_eager": (torch.zeros(8, dtype=torch.int32), None, False),
                "medium_padded_rowcheck_eager": (
                    torch.cat([torch.ones(2, 256, dtype=torch.int32), torch.zeros(2, 256, dtype=torch.int32)], dim=1),
                    -1, False,
                ),
            }
        },
    }
    
    def _run_all_test(self, tensor, dim, compiled):
        def all_fn(t):
            if dim is None:
                return torch.all(t)
            return torch.all(t, dim=dim)
        compare_with_cpu(all_fn, tensor, compiled=compiled)


# ═════════════════════════════════════════════════════════════════════════════
#  TestFloat  —  .float(), .half(), .bfloat16()
# ═════════════════════════════════════════════════════════════════════════════

class TestFloat(unittest.TestCase, metaclass=ParameterizedTestMeta):
    pytestmark = pytest.mark.torch_float
    torch.manual_seed(0)
    
    _tensor_f16 = make_strided_tensor((1, 38, 5120), (194560, 5120, 1), F16)
    _tensor_f32 = make_strided_tensor((1, 38, 5120), (194560, 5120, 1), F32)
    _tensor_bf16 = make_strided_tensor((1, 38, 5120), (194560, 5120, 1), BF16)
    _weight = make_strided_tensor((4096, 5120), (5120, 1), F16)
    _norm_weight = make_strided_tensor((5120,), (1,), F16)
    _logits = make_strided_tensor((1, 38, 32), (1216, 32, 1), F16)
    
    PARAMS = {
        ("test_float_cast", "_run_float_test"): {
            "param_sets": {
                "fp16_to_float_eager": (_tensor_f16, "float", False),
                "fp16_to_float_compiled": (_tensor_f16, "float", True),
                "bf16_to_float_eager": (_tensor_bf16, "float", False),
                "bf16_to_float_compiled": (_tensor_bf16, "float", True),
                "fp16_to_float_batch_eager": (make_strided_tensor((4, 38, 5120), (778240, 5120, 1), F16), "float", False),
            }
        },
        
        ("test_half_cast", "_run_float_test"): {
            "param_sets": {
                "fp32_to_half_eager": (_tensor_f32, "half", False),
                "fp32_to_half_compiled": (_tensor_f32, "half", True),
                "weight_fp32_to_half_eager": (make_strided_tensor((4096, 5120), (5120, 1), F32), "half", False),
            }
        },
        
        ("test_bfloat16_cast", "_run_float_test"): {
            "param_sets": {
                "fp32_to_bf16_eager": (_tensor_f32, "bfloat16", False),
                "fp32_to_bf16_compiled": (_tensor_f32, "bfloat16", True),
                "output_fp32_to_bf16_eager": (make_strided_tensor((4, 128, 5120), (2621440, 5120, 1), F32), "bfloat16", False),
                "gate_fp32_to_bf16_eager": (make_strided_tensor((32768, 5120), (5120, 1), F32), "bfloat16", False),
                "norm_weight_fp32_to_bf16_eager": (make_strided_tensor((5120,), (1,), F32), "bfloat16", False),
            }
        },
        
        ("test_norm_weight_cast", "_run_float_test"): {
            "param_sets": {
                "norm_weight_fp16_to_float_eager": (_norm_weight, "float", False),
                "norm_weight_fp32_to_half_eager": (make_strided_tensor((5120,), (1,), F32), "half", False),
            }
        },
        
        ("test_logits_cast", "_run_float_test"): {
            "param_sets": {
                "logits_fp16_to_float_eager": (_logits, "float", False),
                "logits_batch_fp16_to_float_eager": (make_strided_tensor((4, 128, 32), (16384, 32, 1), F16), "float", False),
            }
        },
        
        ("test_padded_cast", "_run_float_test"): {
            "param_sets": {
                "padded_fp16_to_float_eager": (
                    torch.cat([
                        make_strided_tensor((1, 128, 5120), (655360, 5120, 1), F16),
                        torch.zeros(1, 128, 5120, dtype=F16),
                    ], dim=1),
                    "float", False,
                ),
            }
        },
    }
    
    def _run_float_test(self, tensor, cast, compiled):
        dtype_map = {"float": torch.float32, "half": torch.float16, "bfloat16": torch.bfloat16}
        expected_dtype = dtype_map[cast]
        
        def cast_fn(t):
            return getattr(t, cast)()
        
        def check_and_cast(t):
            result = cast_fn(t)
            assert result.shape == t.shape
            assert result.dtype == expected_dtype
            return result
        
        compare_with_cpu(check_and_cast, tensor, compiled=compiled)


# ═════════════════════════════════════════════════════════════════════════════
#  TestCos  —  torch.cos
# ═════════════════════════════════════════════════════════════════════════════

class TestCos(unittest.TestCase, metaclass=ParameterizedTestMeta):
    pytestmark = pytest.mark.torch_cos
    torch.manual_seed(0)
    
    PARAMS = {
        ("test_cos_prefill", "_run_cos_test"): {
            "param_sets": {
                "prefill_eager": (make_strided_tensor((1, 38, 128), (4864, 128, 1), F32), False),
                "prefill_compiled": (make_strided_tensor((1, 38, 128), (4864, 128, 1), F32), True),
                "prefill_ones_eager": (torch.ones(1, 38, 128, dtype=F32), False),
                "prefill_pi2_eager": (torch.full((1, 38, 128), math.pi / 2, dtype=F32), False),
            }
        },
        
        ("test_cos_decode", "_run_cos_test"): {
            "param_sets": {
                "decode_eager": (make_strided_tensor((1, 1, 128), (128, 128, 1), F32), False),
                "decode_compiled": (make_strided_tensor((1, 1, 128), (128, 128, 1), F32), True),
                "decode_ones_eager": (torch.ones(1, 1, 128, dtype=F32), False),
                "decode_pi2_eager": (torch.full((1, 1, 128), math.pi / 2, dtype=F32), False),
            }
        },
    }
    
    def _run_cos_test(self, x, compiled):
        compare_with_cpu(torch.cos, x, compiled=compiled)


# ═════════════════════════════════════════════════════════════════════════════
#  TestGetitem  —  Tensor.__getitem__
# ═════════════════════════════════════════════════════════════════════════════

class TestGetitem(unittest.TestCase, metaclass=ParameterizedTestMeta):
    pytestmark = pytest.mark.torch_getitem
    torch.manual_seed(0)
    
    _tensor_1d = make_strided_tensor((64,), (1,), F32)
    _tensor_2d = make_strided_tensor((1, 38), (38, 1), I64)
    _tensor_3d = make_strided_tensor((1, 38, 5120), (194560, 5120, 1), F16)
    _tensor_4d_q = make_strided_tensor((1, 32, 38, 128), (155648, 128, 4096, 1), F16)
    _tensor_4d_k = make_strided_tensor((1, 8, 38, 128), (38912, 128, 1024, 1), F16)
    
    PARAMS = {
        ("test_getitem_slice", "_run_getitem_test"): {
            "param_sets": {
                "slice_first_half_eager": (_tensor_1d, S(None, 32), False),
                "slice_first_half_compiled": (_tensor_1d, S(None, 32), True),
                "slice_second_half_eager": (_tensor_1d, S(32, None), False),
                "slice_second_half_compiled": (_tensor_1d, S(32, None), True),
            }
        },
        
        ("test_getitem_index", "_run_getitem_test"): {
            "param_sets": {
                "index_0_eager": (_tensor_3d, 0, False),
                "index_0_compiled": (_tensor_3d, 0, True),
                "index_neg1_eager": (_tensor_3d, -1, False),
                "index_neg1_compiled": (_tensor_3d, -1, True),
                "index_0_2d_eager": (_tensor_2d, 0, False),
            }
        },
        
        ("test_getitem_tuple", "_run_getitem_test"): {
            "param_sets": {
                "first_token_eager": (_tensor_3d, (_, 0, _), False),
                "first_token_compiled": (_tensor_3d, (_, 0, _), True),
                "last_token_eager": (_tensor_3d, (_, -1, _), False),
                "last_token_compiled": (_tensor_3d, (_, -1, _), True),
                "first_head_eager": (_tensor_4d_q, (_, _, 0, _), False),
                "first_head_compiled": (_tensor_4d_q, (_, _, 0, _), True),
                "first_kv_head_eager": (_tensor_4d_k, (_, _, 0, _), False),
                "decode_slice_eager": (_tensor_4d_q, (_, _, S(None, 1), _), False),
            }
        },
        
        ("test_getitem_ellipsis", "_run_getitem_test"): {
            "param_sets": {
                "ellipsis_eager": (_tensor_4d_q, (..., 0), False),
                "ellipsis_compiled": (_tensor_4d_q, (..., 0), True),
            }
        },
    }
    
    def _run_getitem_test(self, x, idx, compiled):
        compare_with_cpu(lambda t: t[idx], x, compiled=compiled)


# ═════════════════════════════════════════════════════════════════════════════
#  TestMatmul  —  torch.matmul
# ═════════════════════════════════════════════════════════════════════════════

class TestMatmul(unittest.TestCase, metaclass=ParameterizedTestMeta):
    pytestmark = pytest.mark.torch_matmul
    torch.manual_seed(0)
    
    _attn_score_prefill = (make_strided_tensor((1, 64, 1), (64, 1, 1), F32),
                           make_strided_tensor((1, 1, 38), (38, 38, 1), F32))
    _attn_score_decode = (make_strided_tensor((1, 64, 1), (64, 1, 1), F32),
                          make_strided_tensor((1, 1, 1), (1, 1, 1), F32))
    
    PARAMS = {
        ("test_torch_matmul_prefill", "_run_matmul_test"): {
            "param_sets": {
                "prefill_eager":    (*_attn_score_prefill, lambda a, b: torch.matmul(a, b), False),
                "prefill_compiled": (*_attn_score_prefill, lambda a, b: torch.matmul(a, b), True),
            }
        },
        
        ("test_torch_matmul_decode", "_run_matmul_test"): {
            "param_sets": {
                "decode_eager":    (*_attn_score_decode, lambda a, b: torch.matmul(a, b), False),
                "decode_compiled": (*_attn_score_decode, lambda a, b: torch.matmul(a, b), True),
            }
        },
        
        ("test_torch_matmul_method", "_run_matmul_test"): {
            "param_sets": {
                "method_prefill_eager":    (*_attn_score_prefill, lambda a, b: a.matmul(b), False),
                "method_prefill_compiled": (*_attn_score_prefill, lambda a, b: a.matmul(b), True),
            }
        },
        
        ("test_torch_matmul_operator", "_run_matmul_test"): {
            "param_sets": {
                "bmm_op_prefill_eager":    (*_attn_score_prefill, lambda a, b: a @ b, False),
                "bmm_op_prefill_compiled": (*_attn_score_prefill, lambda a, b: a @ b, True),
            }
        },
        
        ("test_torch_matmul_zeros", "_run_matmul_test"): {
            "param_sets": {
                "zeros_a_eager":    (torch.zeros(1, 64, 1, dtype=F32), _attn_score_prefill[1], lambda a, b: torch.matmul(a, b), False),
                "zeros_a_compiled": (torch.zeros(1, 64, 1, dtype=F32), _attn_score_prefill[1], lambda a, b: torch.matmul(a, b), True),
                "zeros_b_eager":    (_attn_score_prefill[0], torch.zeros(1, 1, 38, dtype=F32), lambda a, b: torch.matmul(a, b), False),
            }
        },
        
        ("test_torch_matmul_ones", "_run_matmul_test"): {
            "param_sets": {
                "ones_prefill_eager":    (torch.ones(1, 64, 1, dtype=F32), torch.ones(1, 1, 38, dtype=F32), lambda a, b: torch.matmul(a, b), False),
                "ones_prefill_compiled": (torch.ones(1, 64, 1, dtype=F32), torch.ones(1, 1, 38, dtype=F32), lambda a, b: torch.matmul(a, b), True),
                "ones_decode_eager":     (torch.ones(1, 64, 1, dtype=F32), torch.ones(1, 1, 1, dtype=F32), lambda a, b: torch.matmul(a, b), False),
            }
        },

        ("test_torch_matmul_with_bias", "_run_matmul_test"): {
            "param_sets": {
                "with_bias_prefill_eager": (
                    _attn_score_prefill[0], _attn_score_prefill[1],lambda a, b: torch.matmul(a, b) + torch.ones(1, 64, 38, dtype=F32, device=a.device),False,),
                "with_bias_prefill_compiled": (
                    _attn_score_prefill[0], _attn_score_prefill[1],lambda a, b: torch.matmul(a, b) + torch.ones(1, 64, 38, dtype=F32, device=a.device),True,),
                "with_bias_decode_eager": (
                    _attn_score_decode[0], _attn_score_decode[1],lambda a, b: torch.matmul(a, b) + torch.ones(1, 64, 1, dtype=F32, device=a.device),False,),
            }
        },
    }
    
    def _run_matmul_test(self, a, b, op, compiled):
        compare_with_cpu(op, a, b, compiled=compiled)


# ═════════════════════════════════════════════════════════════════════════════
#  TestMean  —  torch.mean
# ═════════════════════════════════════════════════════════════════════════════

class TestMean(unittest.TestCase, metaclass=ParameterizedTestMeta):
    pytestmark = pytest.mark.torch_mean
    torch.manual_seed(0)
    
    _hidden_prefill = make_strided_tensor((1, 38, 5120), (194560, 5120, 1), F32)
    _hidden_decode = make_strided_tensor((1, 1, 5120), (5120, 5120, 1), F32)
    
    PARAMS = {
        ("test_torch_mean_global", "_run_mean_test"): {
            "param_sets": {
                "prefill_global_eager":    (_hidden_prefill, lambda t: torch.mean(t), False),
                "prefill_global_compiled": (_hidden_prefill, lambda t: torch.mean(t), True),
                "decode_global_eager":     (_hidden_decode, lambda t: torch.mean(t), False),
                "decode_global_compiled":  (_hidden_decode, lambda t: torch.mean(t), True),
            }
        },
        
        ("test_torch_mean_dim_last", "_run_mean_test"): {
            "param_sets": {
                "prefill_dim_last_eager":    (_hidden_prefill, lambda t: torch.mean(t, dim=-1), False),
                "prefill_dim_last_compiled": (_hidden_prefill, lambda t: torch.mean(t, dim=-1), True),
                "decode_dim_last_eager":     (_hidden_decode, lambda t: torch.mean(t, dim=-1), False),
                "decode_dim_last_compiled":  (_hidden_decode, lambda t: torch.mean(t, dim=-1), True),
            }
        },
        
        ("test_torch_mean_keepdim", "_run_mean_test"): {
            "param_sets": {
                "prefill_keepdim_eager":    (_hidden_prefill, lambda t: torch.mean(t, dim=-1, keepdim=True), False),
                "prefill_keepdim_compiled": (_hidden_prefill, lambda t: torch.mean(t, dim=-1, keepdim=True), True),
                "decode_keepdim_eager":     (_hidden_decode, lambda t: torch.mean(t, dim=-1, keepdim=True), False),
                "decode_keepdim_compiled":  (_hidden_decode, lambda t: torch.mean(t, dim=-1, keepdim=True), True),
            }
        },
        
        ("test_torch_mean_method", "_run_mean_test"): {
            "param_sets": {
                "prefill_method_eager":    (_hidden_prefill, lambda t: t.mean(dim=-1), False),
                "prefill_method_compiled": (_hidden_prefill, lambda t: t.mean(dim=-1), True),
                "decode_method_eager":     (_hidden_decode, lambda t: t.mean(dim=-1), False),
                "decode_method_compiled":  (_hidden_decode, lambda t: t.mean(dim=-1), True),
            }
        },
        
        ("test_torch_mean_dim_seq", "_run_mean_test"): {
            "param_sets": {
                "prefill_dim1_eager":    (_hidden_prefill, lambda t: torch.mean(t, dim=1), False),
                "prefill_dim0_eager":    (_hidden_prefill, lambda t: torch.mean(t, dim=0), False),
            }
        },
        
        ("test_torch_mean_zeros", "_run_mean_test"): {
            "param_sets": {
                "zeros_prefill_eager":    (torch.zeros(1, 38, 5120, dtype=F32), lambda t: torch.mean(t, dim=-1), False),
                "zeros_prefill_compiled": (torch.zeros(1, 38, 5120, dtype=F32), lambda t: torch.mean(t, dim=-1), True),
            }
        },
        
        ("test_torch_mean_ones", "_run_mean_test"): {
            "param_sets": {
                "ones_prefill_eager":    (torch.ones(1, 38, 5120, dtype=F32), lambda t: torch.mean(t, dim=-1), False),
                "ones_prefill_compiled": (torch.ones(1, 38, 5120, dtype=F32), lambda t: torch.mean(t, dim=-1), True),
            }
        },
        
        ("test_torch_mean_cast", "_run_mean_test"): {
            "param_sets": {
                "cast_fp16_prefill_eager":    (_hidden_prefill, lambda t: torch.mean(t, dim=-1).to(F16), False),
                "cast_fp16_prefill_compiled": (_hidden_prefill, lambda t: torch.mean(t, dim=-1).to(F16), True),
                "cast_fp16_decode_eager":     (_hidden_decode, lambda t: torch.mean(t, dim=-1).to(F16), False),
            }
        },
    }
    
    def _run_mean_test(self, tensor, op, compiled):
        compare_with_cpu(op, tensor, compiled=compiled)


# ═════════════════════════════════════════════════════════════════════════════
#  TestNe  —  torch.ne
# ═════════════════════════════════════════════════════════════════════════════

class TestNe(unittest.TestCase, metaclass=ParameterizedTestMeta):
    pytestmark = pytest.mark.torch_ne
    torch.manual_seed(0)
    
    PARAMS = {
        ("test_torch_ne_scalar", "_run_ne_scalar_test"): {
            "param_sets": {
                "ne_1x38_int64_contiguous_eager": (torch.arange(38, dtype=I64).unsqueeze(0), 1, False),
                "ne_1x38_int64_contiguous_compiled": (torch.arange(38, dtype=I64).unsqueeze(0), 1, True),
                "ne_1x38_int64_noncontig_eager": (
                    torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                   21, 22, 23, 24, 25, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]],
                                 dtype=I64),
                    1, False,
                ),
                "ne_1x38_f32_scalar0_eager": (make_strided_tensor((1, 38), (38, 1), F32), 0.0, False),
                "ne_1x38_f32_scalar0_compiled": (make_strided_tensor((1, 38), (38, 1), F32), 0.0, True),
            }
        },
        
        ("test_torch_ne_tensor", "_run_ne_tensor_test"): {
            "param_sets": {
                "ne_1x38_int64_tensor_eager": (
                    torch.randint(0, 5, (1, 38), dtype=I64),
                    torch.randint(0, 5, (1, 38), dtype=I64),
                    False,
                ),
                "ne_1x38_int64_tensor_compiled": (
                    torch.randint(0, 5, (1, 38), dtype=I64),
                    torch.randint(0, 5, (1, 38), dtype=I64),
                    True,
                ),
            }
        },
    }
    
    def _run_ne_scalar_test(self, a, scalar, compiled):
        compare_with_cpu(lambda x: torch.ne(x, scalar), a, compiled=compiled)
    
    def _run_ne_tensor_test(self, a, b, compiled):
        compare_with_cpu(torch.ne, a, b, compiled=compiled)


# ═════════════════════════════════════════════════════════════════════════════
#  TestSin  —  torch.sin
# ═════════════════════════════════════════════════════════════════════════════

class TestSin(unittest.TestCase, metaclass=ParameterizedTestMeta):
    pytestmark = pytest.mark.torch_sin
    torch.manual_seed(0)
    
    PARAMS = {
        ("test_torch_sin_prefill", "_run_sin_test"): {
            "param_sets": {
                "prefill_eager":    (make_strided_tensor((1, 38, 128), (4864, 128, 1), F32), False),
                "prefill_compiled": (make_strided_tensor((1, 38, 128), (4864, 128, 1), F32), True),
            },
        },
        
        ("test_torch_sin_decode", "_run_sin_test"): {
            "param_sets": {
                "decode_eager":    (make_strided_tensor((1, 1, 128), (128, 128, 1), F32), False),
                "decode_compiled": (make_strided_tensor((1, 1, 128), (128, 128, 1), F32), True),
            },
        },
        
        ("test_torch_sin_out", "_run_sin_out_test"): {
            "param_sets": {
                "prefill_out_eager":    (make_strided_tensor((1, 38, 128), (4864, 128, 1), F32), False),
                "prefill_out_compiled": (make_strided_tensor((1, 38, 128), (4864, 128, 1), F32), True),
                "decode_out_eager":     (make_strided_tensor((1, 1, 128), (128, 128, 1), F32), False),
            },
        },
    }
    
    def _run_sin_test(self, input_tensor, compiled):
        compare_with_cpu(torch.sin, input_tensor, compiled=compiled)
    
    def _run_sin_out_test(self, input_tensor, compiled):
        def sin_out_fn(x):
            out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
            result = torch.sin(x, out=out)
            assert result is out
            return result
        compare_with_cpu(sin_out_fn, input_tensor, compiled=compiled)


# ═════════════════════════════════════════════════════════════════════════════
#  TestEq  —  torch.eq
# ═════════════════════════════════════════════════════════════════════════════

class TestEq(unittest.TestCase, metaclass=ParameterizedTestMeta):
    pytestmark = pytest.mark.torch_eq
    torch.manual_seed(0)
    
    PARAMS = {
        ("test_torch_eq_tensor", "_run_eq_tensor_test"): {
            "param_sets": {
                "1_tensor_eager": (torch.randn(1, dtype=F16), torch.randn(1, dtype=F16), False),
                "1_tensor_compiled": (torch.randn(1, dtype=F16), torch.randn(1, dtype=F16), True),
                "38_tensor_eager": (torch.randn(38, dtype=F16), torch.randn(38, dtype=F16), False),
            },
        },
        
        ("test_torch_eq_scalar", "_run_eq_scalar_test"): {
            "param_sets": {
                "1_scalar_eager": (torch.randn(1, dtype=F16), torch.randn(1).item(), False),
                "1_scalar_compiled": (torch.randn(1, dtype=F16), torch.randn(1).item(), True),
                "38_scalar_eager": (torch.randn(38, dtype=F16), 0.0, False),
            },
        },
        
        ("test_torch_eq_out", "_run_eq_out_test"): {
            "param_sets": {
                "1_out_eager": (torch.randn(1, dtype=F16), torch.randn(1, dtype=F16), False),
                "1_out_compiled": (torch.randn(1, dtype=F16), torch.randn(1, dtype=F16), True),
            },
        },
    }
    
    def _run_eq_tensor_test(self, input_tensor, other_tensor, compiled):
        compare_with_cpu(torch.eq, input_tensor, other_tensor, compiled=compiled)
    
    def _run_eq_scalar_test(self, input_tensor, scalar, compiled):
        compare_with_cpu(lambda x: torch.eq(x, scalar), input_tensor, compiled=compiled)
    
    def _run_eq_out_test(self, input_tensor, other_tensor, compiled):
        def eq_out_fn(x, y):
            out = torch.empty(x.shape, dtype=torch.bool, device=x.device)
            result = torch.eq(x, y, out=out)
            assert result is out
            return result
        compare_with_cpu(eq_out_fn, input_tensor, other_tensor, compiled=compiled)


# ═════════════════════════════════════════════════════════════════════════════
#  TestArange  —  torch.arange
# ═════════════════════════════════════════════════════════════════════════════

class TestArange(unittest.TestCase, metaclass=ParameterizedTestMeta):
    pytestmark = pytest.mark.torch_arange
    torch.manual_seed(0)
    
    PARAMS = {
        ("test_torch_arange_basic", "_run_arange_test"): {
            "param_sets": {
                "shape_1_eager": (1, False),
                "shape_1_compiled": (1, True),
                "shape_38_eager": (38, False),
                "shape_38_compiled": (38, True),
            },
        },
        
        ("test_torch_arange_step", "_run_arange_step_test"): {
            "param_sets": {
                "shape_38_step_2_eager": (0, 38, 2, False),
                "shape_38_step_2_compiled": (0, 38, 2, True),
                "shape_38_step_3_eager": (0, 38, 3, False),
            },
        },
        
        ("test_torch_arange_out", "_run_arange_out_test"): {
            "param_sets": {
                "shape_38_out_eager": (38, False),
                "shape_38_out_compiled": (38, True),
                "shape_1_out_eager": (1, False),
            },
        },
    }
    
    def _run_arange_test(self, end, compiled):
        def arange_fn(device=None):
            return torch.arange(end, dtype=I64, device=device)
        compare_with_cpu(arange_fn, compiled=compiled, needs_device=True)
    
    def _run_arange_step_test(self, start, end, step, compiled):
        def arange_fn(device=None):
            return torch.arange(start, end, step, dtype=I64, device=device)
        compare_with_cpu(arange_fn, compiled=compiled, needs_device=True)
    
    def _run_arange_out_test(self, end, compiled):
        cpu_out = torch.empty(end, dtype=I64, device="cpu")
        cpu_result = torch.arange(end, dtype=I64, out=cpu_out)
        assert cpu_result is cpu_out
        
        alt_out = torch.empty(end, dtype=I64, device=DEVICE)
        alt_result = torch.arange(end, dtype=I64, out=alt_out)
        assert alt_result is alt_out
        
        torch.testing.assert_close(alt_result.to("cpu"), cpu_result)


# ═════════════════════════════════════════════════════════════════════════════
#  TestLinear  —  torch.nn.functional.linear
# ═════════════════════════════════════════════════════════════════════════════
class TestLinear(unittest.TestCase, metaclass=ParameterizedTestMeta):
    pytestmark = pytest.mark.torch_linear
    torch.manual_seed(0)
    
    PREFILL_SEQ = 38
    DECODE_SEQ = 1
    
    _input_prefill = make_strided_tensor((1, 38, 5120), (194560, 5120, 1), F16)
    _input_decode = make_strided_tensor((1, 1, 5120), (5120, 5120, 1), F16)
    _weight_4096 = make_strided_tensor((4096, 5120), (5120, 1), F16)
    _weight_1024 = make_strided_tensor((1024, 5120), (5120, 1), F16)
    _weight_32768 = make_strided_tensor((32768, 5120), (5120, 1), F16)
    _weight_131072 = make_strided_tensor((131072, 5120), (5120, 1), F16)
    _bias_4096 = make_strided_tensor((4096,), (1,), F16)
    _bias_1024 = make_strided_tensor((1024,), (1,), F16)
    _bias_32768 = make_strided_tensor((32768,), (1,), F16)
    _bias_131072 = make_strided_tensor((131072,), (1,), F16)
    
    PARAMS = {
        ("test_linear_prefill", "_run_linear_test"): {
            "param_sets": {
                "prefill_4096x5120_eager": (_input_prefill, _weight_4096, False),
                "prefill_4096x5120_compiled": (_input_prefill, _weight_4096, True),
                "prefill_1024x5120_eager": (_input_prefill, _weight_1024, False),
                "prefill_32768x5120_eager": (_input_prefill, _weight_32768, False),
                "prefill_131072x5120_eager": (_input_prefill, _weight_131072, False),
            }
        },
        
        ("test_linear_decode", "_run_linear_test"): {
            "param_sets": {
                "decode_4096x5120_eager": (_input_decode, _weight_4096, False),
                "decode_4096x5120_compiled": (_input_decode, _weight_4096, True),
                "decode_1024x5120_eager": (_input_decode, _weight_1024, False),
                "decode_32768x5120_eager": (_input_decode, _weight_32768, False),
            }
        },
        
        ("test_linear_with_bias", "_run_linear_bias_test"): {
            "param_sets": {
                "prefill_with_bias_4096_eager": (_input_prefill, _weight_4096, _bias_4096, False),
                "prefill_with_bias_4096_compiled": (_input_prefill, _weight_4096, _bias_4096, True),
                "prefill_with_bias_1024_eager": (_input_prefill, _weight_1024, _bias_1024, False),
                "prefill_with_bias_32768_eager": (_input_prefill, _weight_32768, _bias_32768, False),
                "decode_with_bias_4096_eager": (_input_decode, _weight_4096, _bias_4096, False),
                "decode_with_bias_4096_compiled": (_input_decode, _weight_4096, _bias_4096, True),
            }
        },
        
        ("test_linear_transposed", "_run_linear_test"): {
            "param_sets": {
                "prefill_transposed_eager": (
                    make_strided_tensor((38, 5120), (5120, 1), F16),
                    make_strided_tensor((4096, 5120), (5120, 1), F16),
                    False,
                ),
            }
        },
    }
    
    def _run_linear_test(self, input_tensor, weight, compiled):
        compare_with_cpu(lambda x, w: F.linear(x, w), input_tensor, weight, compiled=compiled)
    
    def _run_linear_bias_test(self, input_tensor, weight, bias, compiled):
        compare_with_cpu(lambda x, w, b: F.linear(x, w, b), input_tensor, weight, bias, compiled=compiled)

# ═════════════════════════════════════════════════════════════════════════════
#  TestCat  —  torch.cat 
# ═════════════════════════════════════════════════════════════════════════════

class TestCat(unittest.TestCase, metaclass=ParameterizedTestMeta):
    pytestmark = pytest.mark.torch_cat
    torch.manual_seed(0)
    
    # Use traced sequence lengths: 37 for specific traced patterns
    PREFILL_SEQ_TRACED = 37
    PREFILL_SEQ_FULL = 38
    DECODE_SEQ = 1
    
    PARAMS = {
        # ── prefill rope (traced with seq=37) ─────────────────────────────────
        ("test_torch_cat_prefill_rope", "_run_cat_test"): {
            "param_sets": {
                "eager": (
                    [make_strided_tensor((1, 37, 64), (2368, 1, 37), F32),
                     make_strided_tensor((1, 37, 64), (2368, 1, 37), F32)],
                    -1, False),
                "compiled": (
                    [make_strided_tensor((1, 37, 64), (2368, 1, 37), F32),
                     make_strided_tensor((1, 37, 64), (2368, 1, 37), F32)],
                    -1, True),
            }
        },

        # ── prefill Q rotary (traced with seq=37) ─────────────────────────────
        ("test_torch_cat_prefill_q_rotary", "_run_cat_test"): {
            "param_sets": {
                "eager": (
                    [make_strided_tensor((1, 32, 37, 64), (75776, 64, 2048, 1), F16),
                     make_strided_tensor((1, 32, 37, 64), (151552, 128, 4096, 1), F16)],
                    -1, False),
                "compiled": (
                    [make_strided_tensor((1, 32, 37, 64), (75776, 64, 2048, 1), F16),
                     make_strided_tensor((1, 32, 37, 64), (151552, 128, 4096, 1), F16)],
                    -1, True),
            }
        },

        # ── prefill K rotary (traced with seq=37) ─────────────────────────────
        ("test_torch_cat_prefill_k_rotary", "_run_cat_test"): {
            "param_sets": {
                "eager": (
                    [make_strided_tensor((1, 8, 37, 64), (18944, 64, 512, 1), F16),
                     make_strided_tensor((1, 8, 37, 64), (37888, 128, 1024, 1), F16)],
                    -1, False),
                "compiled": (
                    [make_strided_tensor((1, 8, 37, 64), (18944, 64, 512, 1), F16),
                     make_strided_tensor((1, 8, 37, 64), (37888, 128, 1024, 1), F16)],
                    -1, True),
            }
        },

        # ── decode rope ──────────────────────────────────────────────────────
        ("test_torch_cat_decode_rope", "_run_cat_test"): {
            "param_sets": {
                "eager": (
                    [make_strided_tensor((1, 1, 64), (64, 1, 1), F32),
                     make_strided_tensor((1, 1, 64), (64, 1, 1), F32)],
                    -1, False),
                "compiled": (
                    [make_strided_tensor((1, 1, 64), (64, 1, 1), F32),
                     make_strided_tensor((1, 1, 64), (64, 1, 1), F32)],
                    -1, True),
            }
        },

        # ── decode Q rotary ───────────────────────────────────────────────────
        ("test_torch_cat_decode_q_rotary", "_run_cat_test"): {
            "param_sets": {
                "eager": (
                    [make_strided_tensor((1, 32, 1, 64), (2048, 64, 2048, 1), F16),
                     make_strided_tensor((1, 32, 1, 64), (4096, 128, 4096, 1), F16)],
                    -1, False),
                "compiled": (
                    [make_strided_tensor((1, 32, 1, 64), (2048, 64, 2048, 1), F16),
                     make_strided_tensor((1, 32, 1, 64), (4096, 128, 4096, 1), F16)],
                    -1, True),
            }
        },

        # ── decode K rotary ───────────────────────────────────────────────────
        ("test_torch_cat_decode_k_rotary", "_run_cat_test"): {
            "param_sets": {
                "eager": (
                    [make_strided_tensor((1, 8, 1, 64), (512, 64, 512, 1), F16),
                     make_strided_tensor((1, 8, 1, 64), (1024, 128, 1024, 1), F16)],
                    -1, False),
                "compiled": (
                    [make_strided_tensor((1, 8, 1, 64), (512, 64, 512, 1), F16),
                     make_strided_tensor((1, 8, 1, 64), (1024, 128, 1024, 1), F16)],
                    -1, True),
            }
        },

        # ── decode KV cache append (traced with seq=37 for past cache) ────────
        ("test_torch_cat_decode_kv_append", "_run_cat_test"): {
            "param_sets": {
                "eager": (
                    [make_strided_tensor((1, 8, 37, 128), (37888, 4736, 128, 1), F16),
                     make_strided_tensor((1, 8, 1, 128), (1024, 128, 128, 1), F16)],
                    2, False),
                "compiled": (
                    [make_strided_tensor((1, 8, 37, 128), (37888, 4736, 128, 1), F16),
                     make_strided_tensor((1, 8, 1, 128), (1024, 128, 128, 1), F16)],
                    2, True),
            }
        },
    }
    
    def _run_cat_test(self, tensors, dim, compiled):
        def cat_fn(*ts):
            return torch.cat(list(ts), dim=dim)
        compare_with_cpu(cat_fn, *tensors, compiled=compiled)

# ═════════════════════════════════════════════════════════════════════════════
#  TestMul  —  torch.mul 
# ═════════════════════════════════════════════════════════════════════════════

class TestMul(unittest.TestCase, metaclass=ParameterizedTestMeta):
    pytestmark = pytest.mark.torch_mul
    torch.manual_seed(0)
    
    # Use traced sequence lengths: 37 for specific traced patterns
    PREFILL_SEQ_TRACED = 37
    PREFILL_SEQ_FULL = 38
    DECODE_SEQ = 1
    
    PARAMS = {
        # ── prefill rope freq × scalar (traced with seq=37) ───────────────────
        ("test_torch_mul_prefill_rope_scalar", "_run_mul_scalar"): {
            "param_sets": {
                "eager": (make_strided_tensor((1, 37, 128), (4736, 128, 1), F32), 1.0, False),
                "compiled": (make_strided_tensor((1, 37, 128), (4736, 128, 1), F32), 1.0, True),
            }
        },

        # ── prefill RMSNorm scale (broadcast) with seq=37 ─────────────────────
        ("test_torch_mul_prefill_rmsnorm", "_run_mul_binary"): {
            "param_sets": {
                "eager": (
                    make_strided_tensor((1, 37, 5120), (189440, 5120, 1), F32),
                    make_strided_tensor((1, 37, 1), (37, 1, 1), F32),
                    False),
                "compiled": (
                    make_strided_tensor((1, 37, 5120), (189440, 5120, 1), F32),
                    make_strided_tensor((1, 37, 1), (37, 1, 1), F32),
                    True),
            }
        },

        # ── prefill weight × hidden (1-D weight broadcast) with seq=37 ────────
        ("test_torch_mul_prefill_weight_hidden", "_run_mul_binary"): {
            "param_sets": {
                "eager": (
                    make_strided_tensor((5120,), (1,), F16),
                    make_strided_tensor((1, 37, 5120), (189440, 5120, 1), F16),
                    False),
                "compiled": (
                    make_strided_tensor((5120,), (1,), F16),
                    make_strided_tensor((1, 37, 5120), (189440, 5120, 1), F16),
                    True),
            }
        },

        # ── prefill Q rotary (traced with seq=37) ─────────────────────────────
        ("test_torch_mul_prefill_q_rotary", "_run_mul_binary"): {
            "param_sets": {
                "eager": (
                    make_strided_tensor((1, 32, 37, 128), (151552, 128, 4096, 1), F16),
                    make_strided_tensor((1, 1, 37, 128), (4736, 4736, 128, 1), F16),
                    False),
                "compiled": (
                    make_strided_tensor((1, 32, 37, 128), (151552, 128, 4096, 1), F16),
                    make_strided_tensor((1, 1, 37, 128), (4736, 4736, 128, 1), F16),
                    True),
            }
        },

        # ── prefill K rotary (traced with seq=37) ─────────────────────────────
        ("test_torch_mul_prefill_k_rotary", "_run_mul_binary"): {
            "param_sets": {
                "eager": (
                    make_strided_tensor((1, 8, 37, 128), (37888, 128, 1024, 1), F16),
                    make_strided_tensor((1, 1, 37, 128), (4736, 4736, 128, 1), F16),
                    False),
                "compiled": (
                    make_strided_tensor((1, 8, 37, 128), (37888, 128, 1024, 1), F16),
                    make_strided_tensor((1, 1, 37, 128), (4736, 4736, 128, 1), F16),
                    True),
            }
        },

        # ── prefill FFN gate × up (full sequence 38) ──────────────────────────
        ("test_torch_mul_prefill_ffn_gate", "_run_mul_binary"): {
            "param_sets": {
                "eager": (
                    make_strided_tensor((1, 38, 32768), (1245184, 32768, 1), F16),
                    make_strided_tensor((1, 38, 32768), (1245184, 32768, 1), F16),
                    False),
                "compiled": (
                    make_strided_tensor((1, 38, 32768), (1245184, 32768, 1), F16),
                    make_strided_tensor((1, 38, 32768), (1245184, 32768, 1), F16),
                    True),
            }
        },

        # ── decode rope freq × scalar ─────────────────────────────────────────
        ("test_torch_mul_decode_rope_scalar", "_run_mul_scalar"): {
            "param_sets": {
                "eager": (make_strided_tensor((1, 1, 128), (128, 128, 1), F32), 1.0, False),
                "compiled": (make_strided_tensor((1, 1, 128), (128, 128, 1), F32), 1.0, True),
            }
        },

        # ── decode RMSNorm ────────────────────────────────────────────────────
        ("test_torch_mul_decode_rmsnorm", "_run_mul_binary"): {
            "param_sets": {
                "eager": (
                    make_strided_tensor((1, 1, 5120), (5120, 5120, 1), F32),
                    make_strided_tensor((1, 1, 1), (1, 1, 1), F32),
                    False),
                "compiled": (
                    make_strided_tensor((1, 1, 5120), (5120, 5120, 1), F32),
                    make_strided_tensor((1, 1, 1), (1, 1, 1), F32),
                    True),
            }
        },

        # ── decode weight × hidden ────────────────────────────────────────────
        ("test_torch_mul_decode_weight_hidden", "_run_mul_binary"): {
            "param_sets": {
                "eager": (
                    make_strided_tensor((5120,), (1,), F16),
                    make_strided_tensor((1, 1, 5120), (5120, 5120, 1), F16),
                    False),
                "compiled": (
                    make_strided_tensor((5120,), (1,), F16),
                    make_strided_tensor((1, 1, 5120), (5120, 5120, 1), F16),
                    True),
            }
        },

        # ── decode Q rotary ───────────────────────────────────────────────────
        ("test_torch_mul_decode_q_rotary", "_run_mul_binary"): {
            "param_sets": {
                "eager": (
                    make_strided_tensor((1, 32, 1, 128), (4096, 128, 4096, 1), F16),
                    make_strided_tensor((1, 1, 1, 128), (128, 128, 128, 1), F16),
                    False),
                "compiled": (
                    make_strided_tensor((1, 32, 1, 128), (4096, 128, 4096, 1), F16),
                    make_strided_tensor((1, 1, 1, 128), (128, 128, 128, 1), F16),
                    True),
            }
        },

        # ── decode K rotary ───────────────────────────────────────────────────
        ("test_torch_mul_decode_k_rotary", "_run_mul_binary"): {
            "param_sets": {
                "eager": (
                    make_strided_tensor((1, 8, 1, 128), (1024, 128, 1024, 1), F16),
                    make_strided_tensor((1, 1, 1, 128), (128, 128, 128, 1), F16),
                    False),
                "compiled": (
                    make_strided_tensor((1, 8, 1, 128), (1024, 128, 1024, 1), F16),
                    make_strided_tensor((1, 1, 1, 128), (128, 128, 128, 1), F16),
                    True),
            }
        },

        # ── decode FFN gate × up ──────────────────────────────────────────────
        ("test_torch_mul_decode_ffn_gate", "_run_mul_binary"): {
            "param_sets": {
                "eager": (
                    make_strided_tensor((1, 1, 32768), (32768, 32768, 1), F16),
                    make_strided_tensor((1, 1, 32768), (32768, 32768, 1), F16),
                    False),
                "compiled": (
                    make_strided_tensor((1, 1, 32768), (32768, 32768, 1), F16),
                    make_strided_tensor((1, 1, 32768), (32768, 32768, 1), F16),
                    True),
            }
        },
    }
    
    def _run_mul_binary(self, a, b, compiled):
        compare_with_cpu(torch.mul, a, b, compiled=compiled)
    
    def _run_mul_scalar(self, a, scalar, compiled):
        compare_with_cpu(lambda x: torch.mul(x, scalar), a, compiled=compiled)


# ═════════════════════════════════════════════════════════════════════════════
#  TestReshape  —  torch.reshape 
# ═════════════════════════════════════════════════════════════════════════════

class TestReshape(unittest.TestCase, metaclass=ParameterizedTestMeta):
    pytestmark = pytest.mark.torch_reshape
    torch.manual_seed(0)
    
    # Use traced sequence lengths: 37 for specific traced patterns
    PREFILL_SEQ_TRACED = 37
    PREFILL_SEQ_FULL = 38
    
    PARAMS = {
        # ── standard head-merge shapes (decode) ──────────────────────────────
        ("test_reshape_A000", "_run_reshape_test"): {
            "param_sets": {
                "decode_1x1x32x128_eager":    (_t((1, 1, 32, 128)), (1, 1, -1), False),
                "decode_1x1x32x128_compiled": (_t((1, 1, 32, 128)), (1, 1, -1), True),
            }
        },

        # ── sequence head-merge (prefill with seq=38) ─────────────────────────
        ("test_reshape_B000", "_run_reshape_test"): {
            "param_sets": {
                "1x38x32x128_eager":    (_t((1, 38, 32, 128)), (1, 38, -1), False),
                "1x38x32x128_compiled": (_t((1, 38, 32, 128)), (1, 38, -1), True),
            }
        },

        # ── TRACED: GQA expand → reshape (zero stride on dim 2, seq=37) ──────
        ("test_reshape_gqa_expand_prefill", "_run_reshape_test"): {
            "param_sets": {
                "eager": (
                    make_strided_tensor(
                        (1, 8, 4, 37, 128),
                        (37888, 128, 0, 1024, 1),
                        F16,
                    ),
                    (1, 32, 37, 128), False),
                "compiled": (
                    make_strided_tensor(
                        (1, 8, 4, 37, 128),
                        (37888, 128, 0, 1024, 1),
                        F16,
                    ),
                    (1, 32, 37, 128), True),
            }
        },

        # ── TRACED: prefill merge heads (non-contiguous input, seq=37) ───────
        ("test_reshape_merge_heads_prefill", "_run_reshape_test"): {
            "param_sets": {
                "eager": (
                    make_strided_tensor((1, 37, 32, 128), (151552, 4096, 128, 1), F16),
                    (1, 37, 4096), False),
                "compiled": (
                    make_strided_tensor((1, 37, 32, 128), (151552, 4096, 128, 1), F16),
                    (1, 37, 4096), True),
            }
        },

        # ── TRACED: decode merge heads ────────────────────────────────────────
        ("test_reshape_merge_heads_decode", "_run_reshape_test"): {
            "param_sets": {
                "eager": (
                    make_strided_tensor((1, 1, 32, 128), (4096, 128, 128, 1), F16),
                    (1, 1, 4096), False),
                "compiled": (
                    make_strided_tensor((1, 1, 32, 128), (4096, 128, 128, 1), F16),
                    (1, 1, 4096), True),
            }
        },

        # ── non-contiguous via transpose then reshape (seq=38) ────────────────
        ("test_reshape_E000", "_run_reshape_noncontig_test"): {
            "param_sets": {
                "transpose_reshape_eager":    (_t((1, 32, 38, 128)), 1, 2, (1, 38, -1), False),
                "transpose_reshape_compiled": (_t((1, 32, 38, 128)), 1, 2, (1, 38, -1), True),
            }
        },

        # ── reshape + contiguous chain (seq=38) ───────────────────────────────
        ("test_reshape_F000", "_run_reshape_chain_test"): {
            "param_sets": {
                "reshape_contiguous_eager":    (_t((1, 38, 32, 128)), (1, 38, -1), False),
                "reshape_contiguous_compiled": (_t((1, 38, 32, 128)), (1, 38, -1), True),
            }
        },
    }
    
    def _run_reshape_test(self, tensor, shape, compiled):
        def fn(t):
            out = t.reshape(*shape)
            if t.dim() == 5 and t.stride(2) == 0 and shape[1] == 32:
                assert out.is_contiguous()
            return out
        compare_with_cpu(fn, tensor, compiled=compiled)
    
    def _run_reshape_noncontig_test(self, tensor, d0, d1, shape, compiled):
        compare_with_cpu(
            lambda t: t.transpose(d0, d1).reshape(*shape),
            tensor, compiled=compiled,
        )
    
    def _run_reshape_chain_test(self, tensor, shape, compiled):
        compare_with_cpu(
            lambda t: t.reshape(*shape).contiguous(),
            tensor, compiled=compiled,
        )
    
    def test_reshape_numel_preserved(self):
        t = _t((1, 38, 32, 128))
        r = t.reshape(1, 38, -1)
        assert r.numel() == t.numel()
    
    def test_gqa_reshape_contiguity(self):
        """GQA expand (zero stride) reshape should produce contiguous output."""
        t = make_strided_tensor(
            (1, 8, 4, 37, 128),
            (37888, 128, 0, 1024, 1),
            F16,
        )
        r = t.reshape(1, 32, 37, 128)
        assert r.is_contiguous()
        assert r.stride() == (151552, 4096, 128, 1)


# ═════════════════════════════════════════════════════════════════════════════
#  TestNeg  —  torch.neg 
# ═════════════════════════════════════════════════════════════════════════════

class TestNeg(unittest.TestCase, metaclass=ParameterizedTestMeta):
    pytestmark = pytest.mark.torch_neg
    torch.manual_seed(0)
    
    # Use traced sequence lengths: 37 for prefill (from traced patterns)
    PREFILL_SEQ_TRACED = 37
    DECODE_SEQ = 1
    
    PARAMS = {
        ("test_torch_neg_prefill", "_run_neg_test"): {
            "param_sets": {
                "prefill_q_eager": (make_strided_tensor((1, 32, 37, 64), (75776, 64, 2048, 1), F16), False),
                "prefill_q_compiled": (make_strided_tensor((1, 32, 37, 64), (75776, 64, 2048, 1), F16), True),
                "prefill_k_eager": (make_strided_tensor((1, 8, 37, 64), (18944, 64, 512, 1), F16), False),
                "prefill_k_compiled": (make_strided_tensor((1, 8, 37, 64), (18944, 64, 512, 1), F16), True),
            }
        },
        
        ("test_torch_neg_decode", "_run_neg_test"): {
            "param_sets": {
                "decode_q_eager": (make_strided_tensor((1, 32, 1, 64), (2048, 64, 2048, 1), F16), False),
                "decode_q_compiled": (make_strided_tensor((1, 32, 1, 64), (2048, 64, 2048, 1), F16), True),
                "decode_k_eager": (make_strided_tensor((1, 8, 1, 64), (512, 64, 512, 1), F16), False),
                "decode_k_compiled": (make_strided_tensor((1, 8, 1, 64), (512, 64, 512, 1), F16), True),
            }
        },
    }
    
    def _run_neg_test(self, x, compiled):
        compare_with_cpu(torch.neg, x, compiled=compiled)


# ═════════════════════════════════════════════════════════════════════════════
#  TestExpand  —  torch.expand
# ═════════════════════════════════════════════════════════════════════════════

class TestExpand(unittest.TestCase, metaclass=ParameterizedTestMeta):
    pytestmark = pytest.mark.torch_expand
    torch.manual_seed(0)
    
    PARAMS = {
        ("test_torch_expand_noop", "_run_expand_test"): {
            "param_sets": {
                "explicit_eager": (make_strided_tensor((1, 64, 1), (64, 1, 1), F32), (1, 64, 1), False),
                "explicit_compiled": (make_strided_tensor((1, 64, 1), (64, 1, 1), F32), (1, 64, 1), True),
                "neg1_eager": (make_strided_tensor((1, 64, 1), (64, 1, 1), F32), (-1, 64, -1), False),
                "neg1_compiled": (make_strided_tensor((1, 64, 1), (64, 1, 1), F32), (-1, 64, -1), True),
            }
        },
        
        # ── GQA broadcast (traced with seq=37) ─────────────────────────────────
        ("test_torch_expand_gqa_broadcast", "_run_expand_test"): {
            "param_sets": {
                "gqa_eager": (
                    make_strided_tensor(
                        (1, 8, 1, 37, 128),
                        (37888, 128, 37888, 1024, 1),  
                        F16,
                    ),
                    (1, 8, 4, 37, 128), False,
                ),
                "gqa_compiled": (
                    make_strided_tensor(
                        (1, 8, 1, 37, 128),
                        (37888, 128, 37888, 1024, 1),  # CORRECTED strides
                        F16,
                    ),
                    (1, 8, 4, 37, 128), True,
                ),
            }
        },
    }
    
    def _run_expand_test(self, x, size, compiled):
        def fn(t):
            out = t.expand(*size)
            if out.dim() == 5 and out.shape[2] > 1:
                assert out.stride(2) == 0
            return out
        compare_with_cpu(fn, x, compiled=compiled)

# ═════════════════════════════════════════════════════════════════════════════
#  TestUnsqueeze  —  torch.unsqueeze
# ═════════════════════════════════════════════════════════════════════════════

class TestUnsqueeze(unittest.TestCase, metaclass=ParameterizedTestMeta):
    pytestmark = pytest.mark.torch_unsqueeze
    torch.manual_seed(0)
    
    PARAMS = {
        ("test_torch_unsqueeze_decode_cos_sin", "_run_unsqueeze_test"): {
            "param_sets": {
                "decode_cos_sin_eager":    (make_strided_tensor((1, 1, 128), (128, 128, 1), F16), 1, False),
                "decode_cos_sin_compiled": (make_strided_tensor((1, 1, 128), (128, 128, 1), F16), 1, True),
            }
        },
        
        ("test_torch_unsqueeze_prefill_cos_sin", "_run_unsqueeze_test"): {
            "param_sets": {
                "prefill_cos_sin_eager":    (make_strided_tensor((1, 38, 128), (4864, 128, 1), F16), 1, False),
                "prefill_cos_sin_compiled": (make_strided_tensor((1, 38, 128), (4864, 128, 1), F16), 1, True),
            }
        },
        
        ("test_torch_unsqueeze_decode_position_ids", "_run_unsqueeze_test"): {
            "param_sets": {
                "decode_position_ids_eager":    (torch.randint(0, 2048, (1,), dtype=I64), 0, False),
                "decode_position_ids_compiled": (torch.randint(0, 2048, (1,), dtype=I64), 0, True),
            }
        },
        
        ("test_torch_unsqueeze_prefill_position_ids", "_run_unsqueeze_test"): {
            "param_sets": {
                "prefill_position_ids_eager":    (torch.arange(38, dtype=I64).unsqueeze(0), 0, False),
                "prefill_position_ids_compiled": (torch.arange(38, dtype=I64).unsqueeze(0), 0, True),
            }
        },
        
        ("test_torch_unsqueeze_negative_dim", "_run_unsqueeze_test"): {
            "param_sets": {
                "negative_dim_eager":    (make_strided_tensor((1, 38, 5120), (194560, 5120, 1), F16), -1, False),
                "negative_dim_compiled": (make_strided_tensor((1, 38, 5120), (194560, 5120, 1), F16), -1, True),
            }
        },
    }
    
    def _run_unsqueeze_test(self, tensor, dim, compiled):
        compare_with_cpu(lambda x: torch.unsqueeze(x, dim), tensor, compiled=compiled)


# ═════════════════════════════════════════════════════════════════════════════
#  TestEmbedding  —  torch.nn.functional.embedding
# ═════════════════════════════════════════════════════════════════════════════

class TestEmbedding(unittest.TestCase, metaclass=ParameterizedTestMeta):
    pytestmark = pytest.mark.torch_embedding
    torch.manual_seed(0)
    
    # Weight: (131072,5120) stride (5120,1) bfloat16 — from trace
    _W = make_strided_tensor((131072, 5120), (5120, 1), F16)
    
    @staticmethod
    def _last_idx(shape, stride):
        t = make_strided_tensor(shape, stride, I64, fill="zeros")
        t.fill_(131071)
        return t
    
    PARAMS = {
        ("test_embedding_shape_prefill", "_run_embedding_shape_test"): {
            "param_sets": {
                "prefill_eager": (
                    make_strided_tensor((1, 38), (38, 1), I64, fill="zeros"),
                    _W, False),
                "prefill_compiled": (
                    make_strided_tensor((1, 38), (38, 1), I64, fill="zeros"),
                    _W, True),
            }
        },
        
        ("test_embedding_shape_decode", "_run_embedding_shape_test"): {
            "param_sets": {
                "decode_eager": (
                    make_strided_tensor((1, 1), (1, 1), I64, fill="zeros"),
                    _W, False),
                "decode_compiled": (
                    make_strided_tensor((1, 1), (1, 1), I64, fill="zeros"),
                    _W, True),
            }
        },
        
        ("test_embedding_values_zero_index", "_run_embedding_values_test"): {
            "param_sets": {
                "zero_prefill": (
                    make_strided_tensor((1, 38), (38, 1), I64, fill="zeros"),
                    _W, False),
                "zero_decode": (
                    make_strided_tensor((1, 1), (1, 1), I64, fill="zeros"),
                    _W, False),
            }
        },
        
        ("test_embedding_values_last_index", "_run_embedding_values_test"): {
            "param_sets": {
                "last_prefill": (
                    _last_idx((1, 38), (38, 1)),
                    _W, False),
                "last_decode": (
                    _last_idx((1, 1), (1, 1)),
                    _W, False),
            }
        },
        
        ("test_embedding_values_arange_index", "_run_embedding_values_test"): {
            "param_sets": {
                "arange_prefill": (
                    make_strided_tensor((1, 38), (38, 1), I64, fill="arange") % VOCAB_SIZE,
                    _W, False),
            }
        },
        
        ("test_embedding_dtype", "_run_embedding_dtype_test"): {
            "param_sets": {
                "dtype_prefill": (
                    make_strided_tensor((1, 38), (38, 1), I64, fill="zeros"),
                    _W, False),
                "dtype_decode": (
                    make_strided_tensor((1, 1), (1, 1), I64, fill="zeros"),
                    _W, False),
            }
        },
    }
    
    def _run_embedding_shape_test(self, idx, w, compiled):
        expected = list(idx.shape) + [w.shape[1]]
        def fn(i, wt):
            out = F.embedding(i, wt)
            assert list(out.shape) == expected
            return out
        compare_with_cpu(fn, idx, w, compiled=compiled)
    
    def _run_embedding_values_test(self, idx, w, compiled):
        compare_with_cpu(lambda i, wt: F.embedding(i, wt), idx, w, compiled=compiled)
    
    def _run_embedding_dtype_test(self, idx, w, compiled):
        def fn(i, wt):
            out = F.embedding(i, wt)
            assert out.dtype == wt.dtype
            return out
        compare_with_cpu(fn, idx, w, compiled=compiled)
# ═════════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    unittest.main()