# utils_mistral_small_3_2_24b_instruct_2506.py
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
Self-contained test utilities for Mistral-Small-3.2-24B-Instruct-2506 operator
tests — specifically for the three new ops:

    • torch.nn.functional.scaled_dot_product_attention  (SDPA)
    • torch.reshape
    • torch.nn.functional.silu

Architecture values are sourced directly from the HuggingFace config.json:
    https://huggingface.co/mistralai/Mistral-Small-3.2-24B-Instruct-2506/blob/main/config.json

Config snapshot (relevant fields only)
---------------------------------------
    "architectures":         ["MistralForCausalLM"],
    "hidden_size":           5120,
    "intermediate_size":     32768,
    "num_attention_heads":   32,        # Q heads
    "num_key_value_heads":   8,         # KV heads (GQA, groups = 32 / 8 = 4)
    "head_dim":              128,       # explicit — NOT hidden_size // num_heads
    "num_hidden_layers":     40,
    "sliding_window":        null,      # no sliding-window attention
    "rope_theta":            100000000.0,
    "vocab_size":            131072,
    "torch_dtype":           "float16",  # Updated to float16
    "rms_norm_eps":          1e-05,
    "max_position_embeddings": 32768,

Public API (consumed by test_mistral_small_3_2_24b_instruct_2506.py)
----------------------------------------------------------------------

--- Architecture constants ---
NUM_Q_HEADS         Query heads = 32.
NUM_KV_HEADS        Key/Value heads = 8 (GQA).
HEAD_DIM            Per-head dimension = 128 (explicit in config, != hidden_size // Q_heads).
GQA_GROUPS          Expansion factor = NUM_Q_HEADS // NUM_KV_HEADS = 4.
SCALE               Attention scale = 1 / sqrt(HEAD_DIM).
SLIDING_WINDOW      None — this model has no sliding-window attention.
NUM_LAYERS          Decoder layers = 40.
DEFAULT_DTYPE       torch.float16 (native compute dtype for this model).
VOCAB_SIZE          131_072.
ROPE_THETA          100_000_000.0.

--- Dimension constants ---
HIDDEN_SIZE         5120  — hidden/residual-stream dimension.
INTERMEDIATE_SIZE   32768 — FFN gate/up/down dimension (≈ 6.4 × HIDDEN_SIZE).

--- Tolerances ---
EAGER_ATOL/RTOL     Eager-mode CPU-vs-Spyre tolerance.
COMPILED_ATOL/RTOL  Compiled-mode CPU-vs-Spyre tolerance (wider; fp16 rounds more).
TOLERANCES          Per-dtype dict for SDPA compare_sdpa().

--- SDPA param dicts (built at import time) ---
PREFILL_PARAMS      Causal prefill cases: seq_q == seq_kv.
DECODE_PARAMS       Decode cases: seq_q == 1, growing KV cache.
DTYPE_PARAMS        Multi-dtype prefill (fp16, bf16, fp32).
NUMERIC_COVERAGE_PARAMS  40-seed numerical sweep.
GROWING_KV_PARAMS   Autoregressive decode with growing KV length.

--- Tensor factories ---
make_tensor / _t    Contiguous tensor factory; dtype-dispatched.
make_qkv            Build (q, k, v) tuples for SDPA param sets.
cached_randn        LRU-cached random tensor factory.
expand_kv           GQA head expansion (on CPU, then moved back).

--- Mask builders ---
causal_mask         Upper-triangular causal additive mask [1,1,S,S].
sdpa_fn             F.scaled_dot_product_attention wrapper (eager, GQA).

--- Comparison helpers ---
compare_sdpa        Eager CPU-vs-Spyre SDPA comparison.
compare_with_cpu    Eager or compiled CPU-vs-Spyre comparison.

--- Test infrastructure ---
ParameterizedTestMeta   Metaclass that expands PARAMS dicts into test methods.
                        functools.wraps is NOT used (avoids pytest deselection).
"""

import functools
import math

import pytest
import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────────────────────────────────────

DEVICE = torch.device("spyre")


# ─────────────────────────────────────────────────────────────────────────────
# Generic tolerances  (eager / compiled paths)
# ─────────────────────────────────────────────────────────────────────────────
# float16 has 10 mantissa bits so tolerances are tighter than bf16

EAGER_ATOL,    EAGER_RTOL    = 1e-3, 1e-3
COMPILED_ATOL, COMPILED_RTOL = 5e-2, 5e-2  # Wider for compiled due to fusion reordering


# ─────────────────────────────────────────────────────────────────────────────
# Model architecture constants  (Mistral-Small-3.2-24B-Instruct-2506)
# ─────────────────────────────────────────────────────────────────────────────

# --- Attention ---
NUM_Q_HEADS    = 32      # Q heads used in SDPA / runtime attention tensors.
NUM_KV_HEADS   = 8       # KV heads (Grouped Query Attention).
GQA_GROUPS     = NUM_Q_HEADS // NUM_KV_HEADS   # 4 — expansion factor.
HEAD_DIM       = 128     # Explicit in config (≠ hidden_size // num_q_heads).
SCALE          = 1.0 / math.sqrt(HEAD_DIM)    # ≈ 0.0884
SLIDING_WINDOW = None    # No sliding-window attention in this model.
NUM_LAYERS     = 40
VOCAB_SIZE     = 131_072
ROPE_THETA     = 100_000_000.0
DEFAULT_DTYPE  = torch.float16   # Native compute dtype (float16).

# --- Dimensions ---
HIDDEN_SIZE       = 5120    # Residual-stream / hidden dimension.
INTERMEDIATE_SIZE = 32768   # FFN gate/up/down intermediate dimension.
EMBED_DIM           = 5120    # Embedding / projection dimension (= 40 * 128).
NUM_ATTENTION_HEADS = 40      # Used for weight-shape arithmetic only.
# --- Weight-matrix shape arithmetic ---
# NUM_Q_HEADS * HEAD_DIM = 32 * 128 = 4096  (Q projection output width).
# NUM_KV_HEADS * HEAD_DIM = 8 * 128 = 1024  (K/V projection output width).
# These are smaller than HIDDEN_SIZE (5120), which is unusual — the model
# projects into a narrower attention space and back.


# ─────────────────────────────────────────────────────────────────────────────
# Per-dtype SDPA comparison tolerances
# ─────────────────────────────────────────────────────────────────────────────

TOLERANCES: dict = {
    torch.float32:  dict(atol=1e-4, rtol=1e-3),
    torch.float16:  dict(atol=1e-2, rtol=1e-2),   # float16: 10-bit mantissa
    torch.bfloat16: dict(atol=2e-2, rtol=2e-2),   # bfloat16: 7-bit mantissa
}


# ─────────────────────────────────────────────────────────────────────────────
# Shorthand dtype constants
# ─────────────────────────────────────────────────────────────────────────────

BF16 = torch.bfloat16
F16  = torch.float16
F32  = torch.float32
I64  = torch.int64


# ─────────────────────────────────────────────────────────────────────────────
# ParameterizedTestMeta
# ─────────────────────────────────────────────────────────────────────────────
# functools.wraps is deliberately NOT used — it sets __wrapped__ which causes
# pytest to misidentify test source locations and silently deselect tests.
_W = torch.randn(VOCAB_SIZE, EMBED_DIM, dtype=F16)

class ParameterizedTestMeta(type):
    """
    Metaclass that expands a PARAMS dict on a TestCase subclass into individual
    test_* methods.

    Expected PARAMS structure::

        PARAMS = {
            (test_name_prefix, base_func_name): {
                "ops_dict":   {"op_name": op_callable, ...},  # optional
                "param_sets": {case_name: (arg0, arg1, ...), ...},
            },
        }

    Pytest mark derivation
    ----------------------
    When the class does NOT declare pytestmark, a mark is derived from
    base_func_name::

        _run_sdpa_test  → @pytest.mark.torch_sdpa
        _run_reshape_test → @pytest.mark.torch_reshape

    When pytestmark IS set on the class, that mark stamps every generated
    method and no additional derived mark is created.
    """

    def __new__(mcs, name, bases, namespace):
        param_map = namespace.get("PARAMS", {})
        to_delete = set()

        for (test_name_prefix, base_func_name), cases in param_map.items():
            base_func = namespace.get(base_func_name)
            if base_func is None:
                continue

            class_has_pytestmark = "pytestmark" in namespace
            if not class_has_pytestmark:
                raw = base_func_name
                if raw.startswith("_run_") and raw.endswith("_test"):
                    op_label = raw.removeprefix("_run_").removesuffix("_test")
                else:
                    op_label = raw.removeprefix("test_")
                mark = getattr(pytest.mark, f"torch_{op_label}")
            else:
                mark = None

            ops_dict   = cases.get("ops_dict", None)
            param_sets = cases["param_sets"]

            for test_case, params in param_sets.items():
                if ops_dict:
                    for op_name, op in ops_dict.items():
                        test_name = f"{test_name_prefix}_{op_name}_{test_case}"
                        assert test_name not in namespace, f"Test name conflict: {test_name}"

                        def _make_ops_test(_bf, _op, _params, _tname, _mark):
                            def test(self):
                                _bf(self, _op, *_params)
                            test.__name__     = _tname
                            test.__qualname__ = f"{name}.{_tname}"
                            test.__doc__      = _bf.__doc__
                            if getattr(_bf, "__unittest_skip__", False):
                                test.__unittest_skip__     = True
                                test.__unittest_skip_why__ = getattr(_bf, "__unittest_skip_why__", "")
                            return _mark(test) if _mark is not None else test

                        namespace[test_name] = _make_ops_test(base_func, op, params, test_name, mark)
                else:
                    test_name = f"{test_name_prefix}_{test_case}"
                    assert test_name not in namespace, f"Test name conflict: {test_name}"

                    def _make_test(_bf, _params, _tname, _mark):
                        def test(self):
                            _bf(self, *_params)
                        test.__name__     = _tname
                        test.__qualname__ = f"{name}.{_tname}"
                        test.__doc__      = _bf.__doc__
                        if getattr(_bf, "__unittest_skip__", False):
                            test.__unittest_skip__     = True
                            test.__unittest_skip_why__ = getattr(_bf, "__unittest_skip_why__", "")
                        return _mark(test) if _mark is not None else test

                    namespace[test_name] = _make_test(base_func, params, test_name, mark)

            to_delete.add(base_func_name)

        for key in to_delete:
            namespace.pop(key, None)

        return super().__new__(mcs, name, bases, namespace)


# ─────────────────────────────────────────────────────────────────────────────
# cached_randn
# ─────────────────────────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=None)
def cached_randn(
    shape,
    differentiation=None,
    abs=False,
    dtype=DEFAULT_DTYPE,
    scale=1.0,
) -> torch.Tensor:
    """LRU-cached random tensor factory.

    Repeated calls with identical arguments return the *same* tensor object,
    making test-input construction at module import time cheap and deterministic.
    """
    out = torch.randn(shape, dtype=dtype) * scale
    return out if not abs else torch.abs(out)


# ─────────────────────────────────────────────────────────────────────────────
# General-purpose contiguous tensor factory
# ─────────────────────────────────────────────────────────────────────────────

def make_tensor(
    shape: tuple,
    dtype: torch.dtype = DEFAULT_DTYPE,
) -> torch.Tensor:
    """Return a fresh, contiguous random tensor of *shape* and *dtype*.

    Dispatch table
    --------------
    float16 / bfloat16 / float32  → ``torch.randn(shape, dtype=dtype) * 10.0``
    int32 / int64                 → ``torch.randint(-100, 100, shape, dtype=dtype)``
    bool                          → ``torch.randint(0, 2, shape, dtype=dtype)``
    """
    if dtype in (torch.int32, torch.int64):
        return torch.randint(-100, 100, shape, dtype=dtype)
    if dtype == torch.bool:
        return torch.randint(0, 2, shape, dtype=dtype)
    return torch.randn(shape, dtype=dtype) * 10.0


# Single-letter alias — used inside PARAMS dicts for brevity.
_t = make_tensor


# ─────────────────────────────────────────────────────────────────────────────
# SDPA tensor factory
# ─────────────────────────────────────────────────────────────────────────────

def make_qkv(
    batch: int,
    seq_q: int,
    seq_kv: int,
    dtype: torch.dtype = DEFAULT_DTYPE,
    diff=0,
) -> tuple:
    """Return a param-set tuple ``(q, k, v)`` backed by ``cached_randn``.

    Shapes (Mistral-Small-3.2-24B-Instruct-2506)
    ---------------------------------------------
    q : [batch, NUM_Q_HEADS,  seq_q,  HEAD_DIM]   e.g. [1, 32,   1, 128]
    k : [batch, NUM_KV_HEADS, seq_kv, HEAD_DIM]   e.g. [1,  8, 128, 128]
    v : [batch, NUM_KV_HEADS, seq_kv, HEAD_DIM]   e.g. [1,  8, 128, 128]

    ``diff`` is forwarded to ``cached_randn(differentiation=...)`` to produce
    distinct tensors for param-set entries that share the same shape and dtype.
    """
    q = cached_randn(
        (batch, NUM_Q_HEADS,  seq_q,  HEAD_DIM),
        differentiation=("q", diff),
        dtype=dtype,
    )
    k = cached_randn(
        (batch, NUM_KV_HEADS, seq_kv, HEAD_DIM),
        differentiation=("k", diff),
        dtype=dtype,
    )
    v = cached_randn(
        (batch, NUM_KV_HEADS, seq_kv, HEAD_DIM),
        differentiation=("v", diff),
        dtype=dtype,
    )
    return (q, k, v)


# ─────────────────────────────────────────────────────────────────────────────
# SDPA pre-built param dicts
# ─────────────────────────────────────────────────────────────────────────────

_SEED_S = NUM_LAYERS // 3
_SEED_M = NUM_LAYERS - 2 * _SEED_S
_SEED_L = _SEED_S

PREFILL_PARAMS: dict = {
    f"bs{b}_seq{s}": make_qkv(b, s, s, diff=(b, s))
    for b, s in [
        (1, 38),  (1, 128),  (1, 256),  (1, 512),  (1, 1024), (1, 2048),
        (2, 38),  (2, 128),  (2, 256),
        (4, 38),  (4, 128),
        (8, 38),
    ]
}

DECODE_PARAMS: dict = {
    f"bs{b}_kv{kv}": make_qkv(b, 1, kv, diff=(b, kv, "dec"))
    for b, kv in [
        (1, 38),  (1, 128),  (1, 256),  (1, 512),  (1, 1024), (1, 2048),
        (2, 38),  (2, 128),
        (4, 38),  (4, 128),
        (8, 38),
    ]
}

DTYPE_PARAMS: dict = {
    "fp16": make_qkv(1, 38,  38,  dtype=torch.float16,  diff="fp16"),
    "bf16": make_qkv(1, 38,  38,  dtype=torch.bfloat16, diff="bf16"),
    "fp32": make_qkv(1, 38,  38,  dtype=torch.float32,  diff="fp32"),
}

NUMERIC_COVERAGE_PARAMS: dict = {
    **{
        f"seed{i:02d}_seq38":   make_qkv(1, 38,   38,   diff=("cov", i, "seq38"))
        for i in range(_SEED_S)
    },
    **{
        f"seed{i:02d}_seq128":  make_qkv(1, 128,  128,  diff=("cov", i, "seq128"))
        for i in range(_SEED_S, _SEED_S + _SEED_M)
    },
    **{
        f"seed{i:02d}_seq2048": make_qkv(1, 2048, 2048, diff=("cov", i, "seq2048"))
        for i in range(_SEED_S + _SEED_M, NUM_LAYERS)
    },
}

GROWING_KV_PARAMS: dict = {
    f"kv{kv}": make_qkv(1, 1, kv, diff=("grow", kv))
    for kv in [1, 2, 4, 8, 16, 32, 38, 64, 128]
}


# ─────────────────────────────────────────────────────────────────────────────
# GQA helper
# ─────────────────────────────────────────────────────────────────────────────

def expand_kv(
    k: torch.Tensor,
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand K/V from NUM_KV_HEADS → NUM_Q_HEADS via repeat_interleave.

    Expansion is performed on CPU (Spyre backend may raise in reshape/view
    during unsafe_view malloc).  Expanded tensors are moved back to the
    original device before being returned.
    """
    device = k.device
    gqa_groups = NUM_Q_HEADS // NUM_KV_HEADS
    k_exp = k.cpu().repeat_interleave(gqa_groups, dim=1).to(device)
    v_exp = v.cpu().repeat_interleave(gqa_groups, dim=1).to(device)
    return k_exp, v_exp


# ─────────────────────────────────────────────────────────────────────────────
# Mask builders
# ─────────────────────────────────────────────────────────────────────────────

def causal_mask(
    seq_len: int,
    dtype: torch.dtype,
    device,
) -> torch.Tensor:
    """Return an upper-triangular causal additive mask of shape [1,1,S,S].

    Future positions receive ``-inf`` so they are zeroed out by softmax.
    """
    mask = torch.zeros(seq_len, seq_len, dtype=dtype)
    mask.masked_fill_(
        torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1),
        float("-inf"),
    )
    return mask.unsqueeze(0).unsqueeze(0).to(device)


# ─────────────────────────────────────────────────────────────────────────────
# Core SDPA wrapper (eager, GQA-aware)
# ─────────────────────────────────────────────────────────────────────────────

def sdpa_fn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    is_causal: bool = False,
) -> torch.Tensor:
    """Wrapper around F.scaled_dot_product_attention with GQA expansion.

    K and V are expanded from NUM_KV_HEADS to NUM_Q_HEADS on CPU before
    being passed to SDPA.  See expand_kv for the rationale.
    """
    k_exp, v_exp = expand_kv(k, v)
    return F.scaled_dot_product_attention(
        q, k_exp, v_exp,
        attn_mask=attn_mask,
        is_causal=is_causal,
        scale=SCALE,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Internal execution helpers
# ─────────────────────────────────────────────────────────────────────────────

def _run_on_device(fn, args, device, compiled=False, needs_device=False):
    """Run *fn* with *args* placed on *device*, optionally under torch.compile."""
    if compiled:
        torch._dynamo.reset_code_caches()

    device      = torch.device(device) if isinstance(device, str) else device
    device_args = [a.to(device) if isinstance(a, torch.Tensor) else a for a in args]
    device_kwargs = {"device": device} if needs_device else {}
    runner = torch.compile(fn) if compiled else fn

    with torch.no_grad():
        return runner(*device_args, **device_kwargs)


def _to_cpu(result):
    """Recursively move tensors to CPU; pass scalars and None through."""
    if isinstance(result, torch.Tensor):
        return result.cpu()
    if isinstance(result, (tuple, list)):
        return type(result)(_to_cpu(r) for r in result)
    return result


def _assert_close(actual, expected, atol, rtol, label):
    """Recursively assert numerical closeness with a descriptive label."""
    if isinstance(actual, (tuple, list)):
        assert type(actual) == type(expected) and len(actual) == len(expected), (
            f"{label}: result structure mismatch — "
            f"actual {type(actual).__name__}[{len(actual)}] vs "
            f"expected {type(expected).__name__}[{len(expected)}]"
        )
        for i, (a, e) in enumerate(zip(actual, expected)):
            _assert_close(a, e, atol, rtol, f"{label}[{i}]")
    elif isinstance(actual, torch.Tensor):
        torch.testing.assert_close(
            actual.cpu(),
            expected.cpu(),
            equal_nan=True,
            atol=atol,
            rtol=rtol,
            msg=lambda msg: f"{label} mismatch\n\n{msg}\n",
        )
    else:
        assert actual == expected, f"{label}: scalar mismatch: {actual} != {expected}"


# ─────────────────────────────────────────────────────────────────────────────
# compare_sdpa  — eager-only SDPA comparison (CPU vs Spyre)
# ─────────────────────────────────────────────────────────────────────────────

def compare_sdpa(
    fn,
    *cpu_args,
    dtype: torch.dtype = DEFAULT_DTYPE,
    atol: float | None = None,
    rtol: float | None = None,
):
    """Run ``fn(*cpu_args)`` eagerly on CPU and on Spyre; assert closeness.

    Tolerances default to the per-dtype values in ``TOLERANCES``.
    """
    tol   = TOLERANCES[dtype]
    _atol = atol if atol is not None else tol["atol"]
    _rtol = rtol if rtol is not None else tol["rtol"]

    cpu_out = fn(*cpu_args)

    spyre_args = tuple(
        arg.to(DEVICE) if isinstance(arg, torch.Tensor) else arg
        for arg in cpu_args
    )
    spyre_out = fn(*spyre_args)

    _assert_close(
        _to_cpu(spyre_out),
        _to_cpu(cpu_out),
        _atol, _rtol,
        "spyre (eager) <-> cpu (eager)",
    )


# ─────────────────────────────────────────────────────────────────────────────
# compare_with_cpu  — eager or compiled comparison (CPU vs Spyre)
# ─────────────────────────────────────────────────────────────────────────────

def compare_with_cpu(
    fn,
    *args,
    compiled: bool = True,
    needs_device: bool = False,
    atol: float | None = None,
    rtol: float | None = None,
):
    """Run ``fn(*args)`` on both CPU and Spyre and assert the outputs match.

    Tensor args are moved to the target device automatically.  Pass
    ``needs_device=True`` for factory functions (e.g. ``torch.zeros``) that
    accept a ``device=`` keyword argument.
    """
    _atol = atol if atol is not None else (COMPILED_ATOL if compiled else EAGER_ATOL)
    _rtol = rtol if rtol is not None else (COMPILED_RTOL if compiled else EAGER_RTOL)

    cpu_out   = _run_on_device(fn, args, device="cpu",   compiled=compiled, needs_device=needs_device)
    spyre_out = _run_on_device(fn, args, device=DEVICE,  compiled=compiled, needs_device=needs_device)

    _assert_close(spyre_out, cpu_out, atol=_atol, rtol=_rtol, label="spyre vs cpu")