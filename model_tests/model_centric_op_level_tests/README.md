# Model-Centric Operator-Level Tests

CPU vs Spyre comparison tests for every operator pattern observed during a
forward pass of the `mistralai/Mistral-Small-3.2-24B-Instruct-2506` model.

Each test executes an operation on CPU and on the Spyre accelerator with
identical inputs, then asserts that the outputs are numerically close within
configurable tolerances.  Both **eager** (uncompiled) and **compiled**
(`torch.compile`) execution paths are covered for every operator.

---

## Table of Contents

- [Model Architecture](#model-architecture)
- [Requirements](#requirements)
- [Running Tests](#running-tests)
  - [Full Suite](#run-the-full-suite)
  - [By Operator](#run-by-operator-mark)
  - [By Execution Mode](#run-by-execution-mode)
  - [Combined Filters](#combining-filters)
  - [Useful Flags](#useful-flags)
- [Operators Covered](#operators-covered)
- [Tolerances](#tolerances)
- [Debugging Failures](#debugging-failures)
- [Known Constraints](#known-constraints)
- [How Test Expansion Works](#how-test-expansion-works--parameterizedtestmeta)
- [License](#license)

---

## Model Architecture

| Parameter | Constant | Value |
|---|---|---|
| `num_attention_heads` (query heads) | `NUM_Q_HEADS` | `32` |
| `num_key_value_heads` (KV heads, GQA) | `NUM_KV_HEADS` | `8` |
| `head_dim` | `HEAD_DIM` | `128` |
| `gqa_groups` | `GQA_GROUPS` | `4` |
| `num_hidden_layers` | `NUM_LAYERS` | `40` |
| Hidden / residual-stream size | `HIDDEN_SIZE` | `4096` |
| Embedding / projection size | `EMBED_DIM` | `5120` |
| `intermediate_size` (FFN) | `INTERMEDIATE_SIZE` | `16384` |
| `vocab_size` | `VOCAB_SIZE` | `131_072` |
| `rope_theta` | `ROPE_THETA` | `1_000_000.0` |
| `sliding_window` | `SLIDING_WINDOW` | `4096` |
| Default dtype | `DEFAULT_DTYPE` | `torch.float16` |

---

## Requirements

Set up the torch-spyre environment by following the official installation guide:
https://github.ibm.com/ai-foundation/torch-spyre-docs/blob/main/docs/basic_install.md

```bash
cd torch-spyre

# Create a virtual environment with access to system site packages
uv venv --system-site-packages

# Activate the virtual environment
source .venv/bin/activate

# Install torch-spyre along with all dependencies (including torch_sendnn)
uv sync --all-extras --active
```

---

## Running Tests

```bash
cd mistral_small_3_2_24b_instruct_2506
pytest test_mistral_small_3_2_24b_instruct_2506.py -k "eager"
```

Note: Some operations are not yet implemented on Spyre, so test failures are expected when running on this device.

#### Run Tests on CPU

To avoid Spyre-related failures, switch the device to CPU in the following utility files:
`utils_mistral_small_3_2_24_instruct_2506.py`
`utils_mistral_small_3_2_24b_instruct_2506.py`

Update the device configuration:
```bash
DEVICE = torch.device("spyre")
```

Change it to:
```bash
DEVICE = torch.device("cpu")
```

### Run by Operator (Mark)

Each test class is tagged with a `pytestmark`.  Pass `-m <mark>` to select
one or more operators:

```bash
# Scaled dot-product attention
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_sdpa

# torch.reshape
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_reshape

# torch.nn.functional.silu
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_nn_functional_silu

# torch.cat
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_cat

# torch.zeros
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_zeros

# torch.add
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_add

# torch.mul
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_mul

# torch.neg
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_neg

# torch.pow
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_pow

# torch.rsqrt
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_rsqrt

# torch.unsqueeze
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_unsqueeze

# aten::_local_scalar_dense  (.item())
pytest test_mistral_small_3_2_24_instruct_2506.py -m local_scalar_dense

# torch.floor
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_floor

# torch.log
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_log

# torch.matmul
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_matmul

# torch.mean
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_mean

# torch.transpose
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_transpose

# torch.cos
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_cos

# torch.sin
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_sin

# Tensor.contiguous
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_contiguous

# torch.nn.functional.embedding
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_embedding

# Tensor.__getitem__  (indexing / slicing)
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_getitem

# torch.true_divide
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_truediv

# Tensor.index_copy_  (in-place KV-cache scatter)
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_index_copy

# torch.nn.functional.linear
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_linear

# Tensor.float / .half / .bfloat16  (dtype casts)
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_float

# Tensor.to  (dtype / device transfer)
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_to

# Tensor.view
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_view

# Tensor.expand
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_expand
```

### Run by Execution Mode

```bash
# Eager (uncompiled) variants only — all operators
pytest test_mistral_small_3_2_24_instruct_2506.py -k "eager"

# Compiled (torch.compile) variants only — all operators
pytest test_mistral_small_3_2_24_instruct_2506.py -k "compiled"
```

### Useful Flags

```bash
# Stop at the first failure with a full traceback and local variable values
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_mul -x --tb=long

# Compact summary — suppress per-test verbose output
pytest test_mistral_small_3_2_24_instruct_2506.py -q

# Disable output capture so print() appears immediately
pytest test_mistral_small_3_2_24_instruct_2506.py -s

# Collect (list) tests without executing them
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_view --collect-only

# Run tests in parallel (requires pytest-xdist)
pytest test_mistral_small_3_2_24_instruct_2506.py -n auto

# Show the slowest 10 tests after the run
pytest test_mistral_small_3_2_24_instruct_2506.py --durations=10

# Re-run only the tests that failed in the previous session
pytest test_mistral_small_3_2_24_instruct_2506.py --lf

# Re-run failures first, then the rest of the suite
pytest test_mistral_small_3_2_24_instruct_2506.py --ff
```

---

## Operators Covered

### `scaled_dot_product_attention` — `TestSDPA`

**Mark:** `torch_sdpa`

Tests run in both **eager** and **compiled** mode (controlled by the
`compiled` boolean in each parameter set).

```bash
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_sdpa
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_sdpa -k "eager"
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_sdpa -k "compiled"
```

SDPA has fine-grained sub-marks for each test scenario:

| Sub-mark | Base method | What it tests |
|---|---|---|
| `torch_sdpa_prefill_causal` | `test_sdpa_prefill_causal` | `is_causal=True`, varying batch and sequence lengths |
| `torch_sdpa_decode` | `test_sdpa_decode` | `seq_q=1`, no mask, varying batch and KV-cache lengths |
| `torch_sdpa_sliding_window` | `test_sdpa_sliding_window` | Sliding-window attention mask |
| `torch_sdpa_causal_flag_vs_mask` | `test_sdpa_causal_flag_vs_mask` | `is_causal=True` must equal explicit causal mask |
| `torch_sdpa_weights_sum_to_one` | `test_sdpa_weights_sum_to_one` | Softmax rows must sum to 1 |
| `torch_sdpa_gqa_shape` | `test_sdpa_gqa_shape` | K/V shape after GQA expansion equals Q shape |
| `torch_sdpa_batch_consistency` | `test_sdpa_batch_consistency` | Identical batch items produce identical outputs |
| `torch_sdpa_gradient_flow` | `test_sdpa_gradient_flow` | Q gradient back-propagates without NaNs (eager only) |
| `torch_sdpa_determinism` | `test_sdpa_determinism` | Two identical calls return identical outputs |
| `torch_sdpa_padding_mask` | `test_sdpa_padding_mask` | `-inf` masked positions get near-zero attention weight |

```bash
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_sdpa_decode
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_sdpa_prefill_causal
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_sdpa_gradient_flow
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_sdpa_decode -k "bs1_kv512"
```

---

### `torch.reshape` — `TestReshape`

**Mark:** `torch_reshape`

All `reshape` call-sites in the model, covering both prefill and decode:

| Group | Pattern | Example |
|---|---|---|
| A | Attention output `[B,S,32,128] → [B,S,4096]` | Prefill and decode |
| X | Grouped KV layout `[B,8,4,seq,128] → [B,32,seq,128]` | GQA reshape |
| L | Non-contiguous: `transpose → reshape` | Exact model path (line 173) |
| M | Full chain: `.reshape().contiguous()` | In-place materialisation |
| P | Multi-dtype (`bfloat16`, `float32`, `int32`) | Dtype coverage |
| H | Hidden-state / FFN shapes `[B,S,4096] → (-1, 4096)` | MLP flatten |
| S | CPU-only contiguity assertion | Structural sanity check |

```bash
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_reshape
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_reshape -k "eager"
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_reshape -k "compiled"
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_reshape -k "grouped_kv"
```

---

### `torch.nn.functional.silu` — `TestFunctionalSilu`

**Mark:** `torch_nn_functional_silu`

`F.silu` appears in every FFN block as the SwiGLU gate activation:
`F.silu(gate_proj(x)) * up_proj(x)`.  `intermediate_size = 16384`.

| Group | Pattern |
|---|---|
| A | FFN gate decode `[B, 1, 16384]` |
| B | FFN gate prefill `[B, S, 16384]` |
| C | Hidden-state gate `[B, S, 4096]` |
| D | Full SwiGLU product: `F.silu(gate) * up` |
| E | Multi-dtype: `float16`, `float32` |
| F | Non-contiguous input: `transpose → silu` |
| G | CPU-only IEEE 754 special values (`±inf`, `NaN`, `±0`) |
| H | CPU-only identity: `F.silu(x) == x * sigmoid(x)` |

```bash
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_nn_functional_silu
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_nn_functional_silu -k "eager"
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_nn_functional_silu -k "compiled"
```

---

### `torch._C._log_api_usage_once` — `TestLogApiUsageOnce`

**Mark:** `torch_log_api_usage_once`

Void C-extension telemetry hook: always returns `None`, idempotent,
thread-safe, and process-global.

| Group | Pattern |
|---|---|
| A | Ministral model keys (layer forward, MLP, RMSNorm, RoPE, LM head) |
| B | HuggingFace + PyTorch internal keys |
| C | Runtime / inference-stack keys (Spyre, vLLM) |
| D | Dynamic layer-index keys (layers 0, 19, 39) |
| E | Return-value contract: must return `None` (CPU-only) |
| F | Idempotency: N repeated calls must not raise (CPU-only) |
| G | Thread safety: concurrent calls from multiple threads (CPU-only) |
| H | Type guard: non-string argument must raise `TypeError` (CPU-only) |

```bash
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_log_api_usage_once
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_log_api_usage_once -k "eager"
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_log_api_usage_once -k "compiled"
```

---

### `torch.cat` — `TestCat`

**Mark:** `torch_cat`

Covers all `torch.cat` call-sites — RoPE rotary split/concat and KV-cache
concatenation (patterns 000–007).

```bash
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_cat
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_cat -k "eager"
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_cat -k "compiled"
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_cat -k "pattern_003"
```

---

### `torch.zeros` — `TestZeros`

**Mark:** `torch_zeros`

Factory operation.  Both CPU and Spyre receive `device=` automatically
(`needs_device=True`).

| Pattern | Variant |
|---|---|
| 000 | Default dtype (`float32`) |
| 001 | `float16` |
| 002 | `float32` explicit |
| 003 | `int32` |
| 004 | `int64` |
| 005 | `bool` |
| 006 | List-style shape input |
| 007 | Tuple-style shape input |
| 008 | `requires_grad=True` |
| 009 | `out=` parameter (manual CPU vs Spyre comparison) |

```bash
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_zeros
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_zeros -k "eager"
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_zeros -k "compiled"
```

---

### `torch.add` — `TestAdd`

**Mark:** `torch_add`

All `torch.add` call signatures observed in the model:

| Patterns | Signature | Example |
|---|---|---|
| 000–005 | Binary `tensor + tensor` | Rotary embed, residual add |
| 006–007 | `tensor + scalar` (variance epsilon) | `[1, 14, 1] + 1e-5` |
| 008–009 | `scalar + tensor` (attention scale) | `1 + tensor[14]` |
| 010 | `torch.add(a, b, alpha=value)` | Scaled second operand |
| 011 | In-place `tensor.add_(tensor)` | Residual update |

```bash
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_add
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_add -k "eager"
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_add -k "compiled"
```

---

### `torch.mul` — `TestMul`

**Mark:** `torch_mul`

All 16 `mul.*` entries from `Ministral-3-14B-Instruct-2512_spyre.yaml`:

| yaml entry | Shapes | Description |
|---|---|---|
| `mul.1` / `mul.9` | `[1,14,128]` / `[1,1,128]` × scalar `1.0` | Attention scaling |
| `mul.2` / `mul.10` | `[1,14,5120]` × `[1,14,1]` | rsqrt normalisation |
| `mul.3` / `mul.11` | `[5120]` × `[1,14,5120]` | Weight × hidden (broadcast) |
| `mul.4` / `mul.12` | `[1,32,14,128]` × `[1,1,14,128]` | Q × cos (RoPE broadcast) |
| `mul.5` / `mul.13` | `[1,8,14,128]` × `[1,1,14,128]` | K × cos (RoPE broadcast) |
| `mul.6` / `mul.14` | scalar `0.1` × `[14]` / `[1]` | Beta scaling |
| `mul.7` / `mul.15` | `[1,32,14,128]` × `[14,1]` | Query × attention scale |
| `mul.8` / `mul.16` | `[1,14,16384]` × `[1,14,16384]` | Gate × up (SwiGLU elementwise) |

```bash
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_mul
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_mul -k "eager"
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_mul -k "compiled"
```

---

### `torch.neg` — `TestNeg`

**Mark:** `torch_neg`

Unary elementwise negation from the `rotate_half` function in the RoPE path:

| Pattern | Shape | Source |
|---|---|---|
| 000 | `[1, 32, 14, 64]` | Q prefill |
| 001 | `[1, 8, 14, 64]` | K prefill |
| 002 | `[1, 32, 1, 64]` | Q decode |
| 003 | `[1, 8, 1, 64]` | K decode |

```bash
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_neg
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_neg -k "eager"
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_neg -k "compiled"
```

---

### `torch.pow` — `TestPow`

**Mark:** `torch_pow`

Integer exponent `2` for variance computation in RMSNorm:

| Pattern | Shape | Usage |
|---|---|---|
| 000 | `[1, 14, 5120]` | Variance prefill |
| 001 | `[1, 1, 5120]` | Variance decode |

```bash
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_pow
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_pow -k "eager"
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_pow -k "compiled"
```

---

### `torch.rsqrt` — `TestRsqrt`

**Mark:** `torch_rsqrt`

Inputs are strictly positive (`abs(randn) + epsilon`) to match the model's
actual `variance + 1e-5` argument:

| Pattern | Shape | Usage |
|---|---|---|
| 000 | `[1, 14, 1]` | Variance prefill |
| 001 | `[1, 1, 1]` | Variance decode |

```bash
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_rsqrt
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_rsqrt -k "eager"
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_rsqrt -k "compiled"
```

---

### `torch.unsqueeze` — `TestUnsqueeze`

**Mark:** `torch_unsqueeze`

| Pattern | Shape | `dim` | Usage |
|---|---|---|---|
| 000 | `[1, 14, 128]` | `1` | cos embedding broadcast (prefill) |
| 001 | `[14]` | `-1` | scaling vector (prefill) |
| 002 | `[1, 1, 128]` | `1` | cos embedding broadcast (decode) |
| 003 | `[1]` | `-1` | scaling vector (decode) |

```bash
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_unsqueeze
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_unsqueeze -k "eager"
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_unsqueeze -k "compiled"
```

---

### `aten::_local_scalar_dense` — `TestLocalScalarDense`

**Mark:** `local_scalar_dense`

The kernel backing `.item()` — exercised at every call-site in the model:

| Patterns | Mode | Source in model |
|---|---|---|
| 000–002 | `"sum"` (seq 128 / 512 / 2048) | `row.sum().item()` — sequence-length check |
| 003–004 | `"sum"` (batch=4, seq 128 / 512) | Per-sequence energy check |
| 005–007 | `"argmax"` | `row.argmax().item()` — next-token / EOS detection |
| 008–010 | `"any_zero"` | `(ids==0).any().item()` — attention-mask padding guard |
| 011 | `"item"` on `tensor(128)` | KV-cache length bookkeeping |
| 012 | `"item"` on `tensor(ROPE_THETA)` | RoPE theta scalar extraction |
| 013 | `"sum"` on unfinished-sequence flag | Active-beam counter |
| 014 | `"item"` on `tensor(5000)` | Sliding-window boundary check |
| 015 | `"item"` on `tensor(HEAD_DIM)` | Head-dim scalar for attention math |

```bash
pytest test_mistral_small_3_2_24_instruct_2506.py -m local_scalar_dense
pytest test_mistral_small_3_2_24_instruct_2506.py -m local_scalar_dense -k "eager"
pytest test_mistral_small_3_2_24_instruct_2506.py -m local_scalar_dense -k "compiled"
```

---

### `torch.floor` — `TestFloor`

**Mark:** `torch_floor`

Shapes `(14,)` and `(1,)` from the RoPE scaling formula.  Variants include
`t.floor()`, `torch.floor(t)`, in-place `t.clone().floor_()`,
`floor → int64` cast, `floor → clamp`, and CPU-only IEEE 754 special-value
checks (±inf preserved unchanged, NaN propagated).

```bash
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_floor
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_floor -k "eager"
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_floor -k "compiled"
```

---

### `torch.log` — `TestLog`

**Mark:** `torch_log`

Inputs are strictly positive (`abs(randn) + 1e-3`).  Variants cover
`torch.log(t)`, `t.log()`, all-ones input (`log(1.0) == 0`), large/small
`float16` values, `log → clamp`, `log → float32` cast, and CPU-only boundary
checks (`log(0) == -inf`, `log(+inf) == +inf`, NaN propagation).

```bash
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_log
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_log -k "eager"
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_log -k "compiled"
```

---

### `torch.matmul` — `TestMatmul`

**Mark:** `torch_matmul`

| Shape A | Shape B | Output | Coverage |
|---|---|---|---|
| `(1, 64, 1)` | `(1, 1, 1)` | `(1, 64, 1)` | Basic |
| `(1, 64, 1)` | `(1, 1, 14)` | `(1, 64, 14)` | Broadcast |
| `(1, 1, 1)` | `(1, 1, 1)` | `(1, 1, 1)` | Scalar-like |
| `(1, 1, 1)` | `(1, 1, 14)` | `(1, 1, 14)` | Broadcast |

Variants cover `torch.matmul`, `a.matmul(b)`, `a @ b`, zero/ones inputs,
matmul+bias, and CPU-only `±inf` / `NaN` propagation checks.

```bash
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_matmul
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_matmul -k "eager"
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_matmul -k "compiled"
```

---

### `torch.mean` — `TestMean`

**Mark:** `torch_mean`

Input shapes `(1, 1, 5120)` and `(1, 14, 5120)`.  Variants cover global
mean, `dim=-1` with and without `keepdim`, `dim=1` (token pooling), `dim=0`
(batch mean), method alias `t.mean()`, zero/one inputs, `mean → float32`
cast, and CPU-only NaN / inf propagation checks.

```bash
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_mean
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_mean -k "eager"
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_mean -k "compiled"
```

---

### `torch.transpose` — `TestTranspose`

**Mark:** `torch_transpose`

Input shapes cover all `transpose` call-sites in the attention and projection
layers:

| 3-D shapes | 4-D shapes |
|---|---|
| `[1, 64, 1]`, `[1, 64, 14]` | `[1, 1, 32, 128]`, `[1, 1, 8, 128]` |
| | `[1, 32, 1, 128]`, `[1, 14, 32, 128]` |
| | `[1, 14, 8, 128]`, `[1, 32, 14, 128]` |

Sub-groups test shape correctness, value correctness, negative dimension
indexing (e.g. `(-2, -1) == (2, 3)`), and dtype preservation.

```bash
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_transpose
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_transpose -k "eager"
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_transpose -k "compiled"
```

---

### `torch.cos` — `TestCos`

**Mark:** `torch_cos`

Pointwise cosine from the RoPE rotary embedding path.  Shapes `[1, 14, 128]`
(prefill) and `[1, 1, 128]` (decode).  Sub-groups test shape correctness,
value correctness (random and zero inputs), and dtype preservation.

```bash
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_cos
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_cos -k "eager"
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_cos -k "compiled"
```

---

### `torch.sin` — `TestSin`

**Mark:** `torch_sin`

Pointwise sine from the RoPE rotary embedding path.  Shapes `[1, 14, 128]`
(prefill) and `[1, 1, 128]` (decode).  Includes an `out=` parameter variant.

```bash
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_sin
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_sin -k "eager"
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_sin -k "compiled"
```

---

### `Tensor.contiguous` — `TestContiguous`

**Mark:** `torch_contiguous`

Input shapes: `[1, 14, 32, 128]`, `[1, 1, 32, 128]`, `[1, 14, 4096]`,
`[1, 1, 4096]`.  Sub-groups test shape preservation, value correctness on
already-contiguous inputs, correctness after a non-contiguous `transpose`
view (only dim pairs where both swapped sizes > 1 guarantee
non-contiguity), and dtype preservation.

```bash
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_contiguous
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_contiguous -k "eager"
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_contiguous -k "compiled"
```

---

### `torch.nn.functional.embedding` — `TestEmbedding`

**Mark:** `torch_embedding`

| Parameter | Value |
|---|---|
| Weight shape | `[131072, 5120]` (`VOCAB_SIZE × EMBED_DIM`) |
| Index dtype | `torch.int64` |
| Weight dtype | `torch.float16` |
| Index shapes | `[1, 14]` (prefill), `[1, 1]` (decode) |

Sub-groups test output shape (`[*index.shape, EMBED_DIM]`), value
correctness (random, all-zero, last vocab index), and dtype preservation.

```bash
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_embedding
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_embedding -k "eager"
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_embedding -k "compiled"
```

---

### `Tensor.__getitem__` — `TestGetitem`

**Mark:** `torch_getitem`

Index expressions are plain Python (integer, `slice`, tuple of slices) and
are passed through unchanged.

| Tensor shape | Index | Output shape | Model usage |
|---|---|---|---|
| `[64]` int64 | `[:32]` | `[32]` | Position ID slicing |
| `[1, 14]` int64 | `[0]` | `[14]` | Batch squeeze |
| `[1, 32, 14, 128]` fp16 | `[:, :, :1, :]` | `[1, 32, 1, 128]` | Decode slice |
| `[1, 8, 2048, 128]` fp16 | `[:, :, :14, :]` | `[1, 8, 14, 128]` | KV-cache prefill read |
| `[1, 14, 5120]` fp16 | `[:, 0, :]` | `[1, 5120]` | First-token hidden state |

```bash
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_getitem
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_getitem -k "eager"
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_getitem -k "compiled"
```

---

### `torch.true_divide` — `TestTruediv`

**Mark:** `torch_truediv`

Implements `torch.true_divide(x, divisor)` (equivalent to
`torch.div(..., rounding_mode=None)`).  Appears in the RoPE scaling formula:
`scaling = 1 + beta * log(1 + floor(position_ids / max_position_embeddings))`.

| Pattern | Input shape | Divisor | Check |
|---|---|---|---|
| 000–001 | `[14]` / `[1]` int64 | `16384` | Basic division |
| 002 | `tensor([16384])` | `16384` | Exact result = `1.0` |
| 003 | `tensor([0])` | `16384` | Zero input = `0.0` |
| 004 | `tensor([7])` | `2` | No flooring: `7 / 2 == 3.5` |
| 005 | `[14]` int64 | `16384` | Output dtype must be floating-point |
| 006–007 | `[14]` / `[1]` int64 | `16384` | `out=` parameter |

```bash
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_truediv
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_truediv -k "eager"
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_truediv -k "compiled"
```

---

### `Tensor.index_copy_` — `TestTensorIndexCopy`

**Mark:** `torch_index_copy`

In-place scatter into the KV cache:
`self.keys.index_copy_(dim=2, index=cache_position, tensor=key_states)`.
The cache tensor is cloned before the in-place operation to prevent mutation
of the original parameter set data.

| Pattern | Cache shape | `dim` | Index shape | Source shape | Usage |
|---|---|---|---|---|---|
| 000 | `[1, 8, 2048, 128]` | `2` | `[14]` int64 | `[1, 8, 14, 128]` | Prefill write |
| 001 | `[1, 8, 2048, 128]` | `2` | `[1]` int64 | `[1, 8, 1, 128]` | Decode write |

```bash
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_index_copy
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_index_copy -k "eager"
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_index_copy -k "compiled"
```

---

### `torch.nn.functional.linear` — `TestLinear`

**Mark:** `torch_linear`

`F.linear(input, weight, bias=None)` implements `y = x A^T + b`.
Patterns 000–010 have no bias; patterns 011–021 include a bias vector.

| Input shape | Weight shape | Layer |
|---|---|---|
| `[1, 14, 5120]` | `[4096, 5120]` | Q projection prefill |
| `[1, 14, 5120]` | `[1024, 5120]` | KV projection prefill |
| `[1, 14, 4096]` | `[5120, 4096]` | Output projection prefill |
| `[1, 14, 5120]` | `[16384, 5120]` | Gate / up FFN projection prefill |
| `[1, 14, 16384]` | `[5120, 16384]` | Down FFN projection prefill |
| `[1, 1, 5120]` | `[131072, 5120]` | LM head decode |
| `[1, 1, 5120]` | `[4096, 5120]` | Q projection decode |
| *(+ decode variants for all projection layers above)* | | |

> **Note:** Tolerances are `atol=2.0, rtol=0.05` because large matrix
> multiplications accumulate `float16` rounding error beyond the default
> thresholds.

```bash
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_linear
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_linear -k "eager"
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_linear -k "compiled"
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_linear -k "bias"
```

---

### `Tensor.float / .half / .bfloat16` — `TestFloat`

**Mark:** `torch_float`

Dtype cast shorthand methods (`tensor.float()`, `tensor.half()`,
`tensor.bfloat16()`):

| Group | Source → Target | Model usage |
|---|---|---|
| 000–003 | `fp16` / `bf16` → `fp32` | `hidden_states.float()` in RMSNorm upcast |
| 004–006 | `fp32` → `fp16` / `bf16` | Attention weights, KV cache storage |
| 007–009 | `fp32` → `fp16` / `bf16` | Model weight loading |
| 010–012 | `fp16` / `fp32` → various | 1-D RMSNorm weight cast |
| 013–015 | `fp16` → `fp32` | `logits.float()` before cross-entropy loss |

```bash
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_float
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_float -k "eager"
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_float -k "compiled"
```

---

### `Tensor.to` — `TestTo`

**Mark:** `torch_to`

`Tensor.to(dtype=..., device=...)` — covers dtype transfer, device transfer,
and combined dtype+device transfer:

| Group | Transfer type | Typical usage |
|---|---|---|
| 000–004 | dtype only | `hidden_states.to(torch.float32)`, mask cast |
| 005–007 | dtype only | Weight loading, KV cache storage |
| 008–010 | dtype only | `input_ids.to(int64)`, causal mask cast |
| 011–015 | device only | Activations, KV cache, token IDs to `"cpu"` |
| 016–018 | dtype + device | Combined activation / KV cache transfer |

```bash
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_to
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_to -k "eager"
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_to -k "compiled"
```

---

### `Tensor.view` — `TestView`

**Mark:** `torch_view`

Zero-copy reshape via `.view()`.  The base method asserts that storage is
shared (`data_ptr()` unchanged — no allocation occurs).

| Group | Pattern | Example |
|---|---|---|
| 000–003 | Q/K/V split | `[1, 128, 5120] → [1, 128, 32, 128]` |
| 004–006 | Attention output merge | `[1, 128, 32, 128] → [1, 128, 5120]` |
| 007–009 | MLP flatten | `[1, 128, 5120] → [128, 5120]` |
| 010–011 | MLP unflatten | `[128, 5120] → [1, 128, 5120]` |
| 012 | LM-head flatten | `[1, 128, 5120] → (-1, 5120)` |
| 013–015 | Padded / medium-seq / large-batch | Various |

```bash
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_view
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_view -k "eager"
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_view -k "compiled"
```

---

### `Tensor.expand` — `TestExpand`

**Mark:** `torch_expand`

Zero-copy broadcast view via `.expand()`.  Expanded dimensions receive
`stride=0`; no new memory is allocated.  The base method asserts that
storage is shared (`data_ptr()` unchanged).

| Group | Pattern | Example |
|---|---|---|
| 000–003 | GQA KV-head expansion | `[1, 8, 1, 128, 128] → [1, 8, 4, 128, 128]` |
| 004–006 | Attention mask broadcast | `[1, 1, 128, 128] → [4, 32, 128, 128]` |
| 007–009 | RoPE cos/sin broadcast | `[1, 128, 128] → [4, 128, 128]` |
| 010–012 | Bias / scalar broadcast | `[1, 1, 4096] → [1, 128, 4096]` |
| 013–015 | Padded / large-batch variants | Various |

```bash
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_expand
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_expand -k "eager"
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_expand -k "compiled"
```

---

## Combining Filters

`-m` selects by operator mark; `-k` filters by any substring of the
generated test method name.  They compose freely:

```bash
# Eager path only for torch.mul
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_mul -k "eager"

# Compiled path only for torch.add
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_add -k "compiled"

# A single cat pattern
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_cat -k "pattern_003"

# SDPA decode tests for a specific batch/kv size
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_sdpa_decode -k "bs1_kv512"

# Multiple operators in one run
pytest test_mistral_small_3_2_24_instruct_2506.py -m "torch_neg or torch_rsqrt or torch_pow"

# Everything except SDPA (faster CI pass)
pytest test_mistral_small_3_2_24_instruct_2506.py -m "not torch_sdpa"

# Everything except SDPA and linear (skip the heaviest matmul patterns)
pytest test_mistral_small_3_2_24_instruct_2506.py -m "not torch_sdpa and not torch_linear"

# All reshape tests for the grouped KV layout (Group X)
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_reshape -k "grouped_kv"

# All linear tests that include a bias term
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_linear -k "bias"

# All eager silu tests
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_nn_functional_silu -k "eager"

# Stop at the first failure with a full traceback
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_mul -x --tb=long

# Collect (list) tests without executing them
pytest test_mistral_small_3_2_24_instruct_2506.py -m torch_view --collect-only
```

---

## Tolerances

`compare_with_cpu` selects tolerances automatically based on execution mode:

| Mode | `atol` | `rtol` | Constant |
|---|---|---|---|
| Eager (uncompiled) | `5e-3` | `5e-3` | `EAGER_ATOL` / `EAGER_RTOL` |
| Compiled (`torch.compile`) | `1e-1` | `1e-2` | `COMPILED_ATOL` / `COMPILED_RTOL` |

SDPA tests use per-dtype tolerances from `TOLERANCES`:

| dtype | `atol` | `rtol` |
|---|---|---|
| `torch.float32` | `1e-4` | `1e-3` |
| `torch.bfloat16` | `1e-2` | `1e-2` |
| `torch.float16` | `1e-2` | `1e-2` |

`TestLinear` uses relaxed tolerances (`atol=2.0, rtol=0.05`) because
accumulation of `float16` rounding error in large matrix multiplications
can exceed the default thresholds.

Compiled mode uses looser bounds because `torch.compile` may fuse or reorder
operations, introducing small additional rounding differences compared to
plain eager execution.  **Failures exceeding `EAGER_ATOL` should be
investigated as backend bugs rather than addressed by widening the
tolerance.**

---

## Debugging Failures

When a test fails on Spyre, enable Dynamo's internal logging to get a full
backend trace:

```bash
TORCHDYNAMO_VERBOSE=1 TORCH_LOGS="+dynamo" \
    pytest test_mistral_small_3_2_24_instruct_2506.py \
    -m "<mark>" -k "<filter>" \
    -x --tb=long -s \
    2>&1 | tee debug_output.txt
```

| Flag / variable | Purpose |
|---|---|
| `TORCHDYNAMO_VERBOSE=1` | Prints the internal Dynamo stack trace (normally suppressed) |
| `TORCH_LOGS="+dynamo"` | Enables all Dynamo-level log output |
| `-m "<mark>"` | Restrict to one operator — e.g. `-m torch_cat` |
| `-k "<filter>"` | Narrow further — e.g. `-k "eager"` or `-k "pattern_003"` |
| `-x` | Stop at the first failure |
| `--tb=long` | Show the full Python traceback including local variable values |
| `-s` | Disable output capture so `print()` appears immediately |
| `2>&1` | Merge stderr (Dynamo logs) into stdout |
| `tee debug_output.txt` | Write everything to a file while still printing to the terminal |

**Example — debug an eager `torch.cat` failure:**

```bash
TORCHDYNAMO_VERBOSE=1 TORCH_LOGS="+dynamo" \
    pytest test_mistral_small_3_2_24_instruct_2506.py \
    -m "torch_cat" -k "eager" \
    -x --tb=long -s \
    2>&1 | tee debug_output.txt
```

**Example — debug a compiled `torch.linear` failure:**

```bash
TORCHDYNAMO_VERBOSE=1 TORCH_LOGS="+dynamo" \
    pytest test_mistral_small_3_2_24_instruct_2506.py \
    -m "torch_linear" -k "compiled" \
    -x --tb=long -s \
    2>&1 | tee debug_output.txt
```

**Search the captured log for the root cause:**

```bash
grep -A 10 "Error\|Exception\|FAILED" debug_output.txt
```

---

## Known Constraints

| Constraint | Detail |
|---|---|
| **SDPA gradient flow — eager only** | `test_sdpa_gradient_flow` has no `compiled=True` variant because the compiled backward pass is not supported on Spyre. |
| **GQA `expand_kv` runs on CPU** | Head expansion via `repeat_interleave` is performed on CPU before moving tensors to Spyre, to avoid a crash inside the Spyre `maybe_get_squeezed_layout` allocator that affects `view` / `reshape` / `unsafe_view`. |
| **`torch.zeros` `out=` pattern** | `compare_with_cpu` cannot be used directly because the `out=` parameter changes the call convention.  These tests run CPU and Spyre manually and compare results. |

---

## How Test Expansion Works — `ParameterizedTestMeta`

The `ParameterizedTestMeta` metaclass expands the `PARAMS` dict defined on
each `unittest.TestCase` subclass into individual `test_*` methods at class
creation time.  `functools.wraps` is deliberately **not** used: it sets
`__wrapped__`, which causes pytest to misidentify each test's source location
and silently deselect it.  `__name__` and `__qualname__` are set explicitly
instead.

**Expected `PARAMS` structure:**

```python
PARAMS = {
    # Key: (generated_test_prefix, base_method_name_in_class)
    ("test_my_op_pattern_000", "_run_my_op_test"): {
        # Optional: when present, tests are expanded as ops × cases.
        "ops_dict": {
            "op_name": op_callable,
            ...
        },
        "param_sets": {
            "case_name_eager":    (arg0, arg1, ..., False),
            "case_name_compiled": (arg0, arg1, ..., True),
            ...
        },
    },
    ...
}
```

**Pytest mark derivation** (applied when the class does not already define
`pytestmark`):

```
_run_cat_test     → @pytest.mark.torch_cat
test_sdpa_decode  → @pytest.mark.torch_sdpa_decode
```

When `pytestmark` is set on the class (e.g. `pytest.mark.torch_sdpa`),
that mark stamps every method instead and no additional mark is derived.

---

## License

```
Copyright 2025 The Torch-Spyre Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```