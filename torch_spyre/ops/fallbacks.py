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


# How to add a CPU fallback operator:
#
# Step 1. Check if the target operator has a default decomposition.
#    - If yes, verify whether the decomposition expands into sub-ops that:
#        * Can be compiled, OR
#        * Correctly fall back to CPU.
#      If both conditions hold, no further action is needed.
#
#    - If some sub-ops cannot compile or fall back to CPU:
#        * Option A: Proceed to Step 2.
#        * Option B: Repeat Step 1 for each unsupported sub-op.
#
#    Example:
#      aten.arange decomposes into prims.iota, which only supports integer
#      dtypes. This requires int-to-float conversion, which Spyre does not
#      fully support yet. In this case, disable the default decomposition.
#      See: https://github.com/pytorch/pytorch/blob/v2.9.1/torch/_refs/__init__.py#L5124-L5222
#
# Step 2. Define an eager CPU fallback in: torch_spyre/fallbacks.py
#
#    Example:
#    @register_fallback([aten.sin.default, aten.sin.out])
#    def spyre__sin(input, **kwargs):
#        return torch.sin(input, **kwargs)
#
#    Note: You can identify the ATen operator name (e.g., aten.sin.default) by:
#      * torch.ops.aten.sin.overloads() # lists overloads like ['default', 'out']
#      * torch.ops.aten.sin.default     # OpOverload object for 'default'
#      * torch.ops.aten.sin._schema     # shows the dispatcher schema


import functools
import os
import warnings
import logging

import torch

aten = torch._ops.ops.aten

fallback_ops = list()

logger = logging.getLogger(__name__)
_SPYRE_FALLBACK_WARN = int(os.environ.get("TORCH_SPYRE_FALLBACK_WARN", "1"))


class FallbackWarning(UserWarning):
    """
    Warning issued when an operator runs on a fallback device (e.g., CPU)
    instead of Spyre
    """


warnings.simplefilter("once", FallbackWarning)
_warn_skips = (
    os.path.dirname(__file__),
    os.path.dirname(torch.__file__),
    torch._inductor.runtime.cache_dir_utils.cache_dir(),
)


def warn_fallback(op, fallback_device="cpu"):
    warnings.warn(
        f"{op} is falling back to {fallback_device}",
        category=FallbackWarning,
        skip_file_prefixes=_warn_skips,
    )


def register_fallback(ops, device="cpu"):
    """
    Decorator to register a CPU-fallback kernel for each op.

    - Moves all tensor inputs to the fallback device (default: CPU) before calling
      the wrapped function.
    - Executes the function on the fallback device, then returns the result on the
      target Spyre device.
    - If `out=` is provided, allocates a buffer tensor on the fallback device and
      copies the result into the `out` tensor.

    Target Spyre device resolution:
      - If `device=` is specified: treat it as the target device
      - Otherwise infer from tensor inputs; if none exist, use `torch.get_default_device()`

    Example:
        @register_fallback(["aten::op1", "aten::op1.out"]):
        def spyre_op1(input1, input2, **kwargs):
            return torch.op1(input1, input2, **kwargs)
    """

    fallback_device = torch.device(device)

    def _is_tensor(x):
        return isinstance(x, torch.Tensor)

    def _ensure_device(args, kwargs):
        # If `device=` was explicitly specified, use it as the target Spyre device
        spyre_device = kwargs.get("device")
        if spyre_device is not None:
            kwargs["device"] = fallback_device
            return torch.device(spyre_device)

        # Infer the target Spyre device from tensor inputs
        devices = {a.device for a in (*args, *kwargs.values()) if _is_tensor(a)}

        if not devices:
            # No tensor inputs and no 'device=' provided
            kwargs["device"] = fallback_device
            return torch.get_default_device()

        if len(devices) > 1:
            raise RuntimeError(
                f"Expected all tensors to be on the same device, but found: {devices}"
            )

        return devices.pop()

    def _move_tensors(args, kwargs):
        # Cache moved tensors to preserve aliasing and avoid redundant moves
        memo = {}

        def _move(t):
            key = id(t)
            moved = memo.get(key)
            if moved is None:
                # Preserve dtype when moving to fallback device
                moved = t.to(device=fallback_device, dtype=t.dtype)
                memo[key] = moved
            return moved

        for i, v in enumerate(args):
            if _is_tensor(v):
                args[i] = _move(v)

        for k, v in kwargs.items():
            if k != "out" and _is_tensor(v):
                kwargs[k] = _move(v)

        # Prepare `out` buffer on the fallback device; reuse alias if already moved
        out = kwargs.get("out")
        if out is not None:
            moved = memo.get(id(out))
            if moved is None:
                moved = torch.empty_like(out, device=fallback_device)
            kwargs["out"] = moved

    def _fallback(fn, *args, **kwargs):
        # Make args mutable
        args = list(args)

        # Validate 'out='
        out = kwargs.get("out")
        if out is not None and not _is_tensor(out):
            raise TypeError(f"argument 'out' must be Tensor, not {type(out)}")

        # Resolve the target Spyre device, and update 'device=' if necessary
        spyre_device = _ensure_device(args, kwargs)

        # Move input tensors to the fallback device
        _move_tensors(args, kwargs)

        # Compute on the fallback device
        fallback_result = fn(*args, **kwargs)

        # If 'out=' was specified, copy result into it
        if out is not None:
            out.copy_(fallback_result)
            return out

        # Otherwise, return result moved to the target Spyre device
        return fallback_result.to(spyre_device)

    def _decorator(fn):
        for op in ops:

            @functools.wraps(fn)
            def _wrapped(*args, **kwargs):
                warn_fallback(op, fallback_device)
                return _fallback(fn, *args, **kwargs)

            fallback_ops.append(op)

            torch.library.register_kernel(op, ["spyre"])(_wrapped)
        return fn

    return _decorator


#  CPU-fallback eager operators


@register_fallback([aten.arange.default, aten.arange.start, aten.arange.start_step])
def spyre__arange(*args, **kwargs):
    return torch.arange(*args, **kwargs)


@register_fallback([aten.arange.out, aten.arange.start_out])
def spyre__arange_out(*args, out, **kwargs):
    kwargs.update({"device": "cpu", "dtype": out.dtype, "layout": out.layout})
    return torch.arange(*args, **kwargs)


@register_fallback([aten.sin.default, aten.sin.out])
def spyre__sin(input, **kwargs):
    return torch.sin(input, **kwargs)


@register_fallback([aten.cos.default, aten.cos.out])
def spyre__cos(input, **kwargs):
    return torch.cos(input, **kwargs)


# Manually append to fallback_ops: register_fallback cannot be used here because
# normal_ is an in-place op — register_fallback is designed for out-of-place ops
# and would leave the original Spyre tensor unfilled.
# The kernel itself is registered in ops.py (and therefore codegen_ops.py).
fallback_ops.append(aten.normal_.default)


@register_fallback([aten.embedding.default])
def spyre__embedding(
    weight, indices, padding_idx=-1, scale_grad_by_freq=False, sparse=False
):
    """
    Fallback for torch.nn.functional.embedding.

    Embedding requires indirect indexing (weight[indices]), which is not
    supported by Spyre's current pointwise operation framework.
    """
    # TODO: Remove this fallback once we enable gather/scatter ops on spyre
    return aten.embedding(weight, indices, padding_idx, scale_grad_by_freq, sparse)


@register_fallback(
    [
        aten.isin.Tensor_Tensor,
        aten.isin.Tensor_Tensor_out,
        aten.isin.Tensor_Scalar,
        aten.isin.Tensor_Scalar_out,
        aten.isin.Scalar_Tensor,
        aten.isin.Scalar_Tensor_out,
    ]
)
def spyre__isin(
    elements, test_elements, *, assume_unique=False, invert=False, **kwargs
):
    """
    Fallback for torch.isin on Spyre.

    """
    return torch.isin(
        elements, test_elements, assume_unique=assume_unique, invert=invert, **kwargs
    )


@register_fallback([aten.tril.default, aten.tril.out])
def spyre__tril(input, diagonal=0, **kwargs):
    return torch.tril(input, diagonal, **kwargs)


@register_fallback([aten.triu.default, aten.triu.out])
def spyre__triu(input, diagonal=0, **kwargs):
    return torch.triu(input, diagonal, **kwargs)


@register_fallback([aten.slice.Tensor])
def spyre__slice(self, dim=0, start=None, end=None, step=1):
    return torch.ops.aten.slice(self, dim, start, end, step)


#-------------------####################--------------------------------#
# -------------------------------------------------------------------
# Shared CPU fallback — preserves source device on output
# -------------------------------------------------------------------

def cpu_fallback(op, *args, **kwargs):
    """
    Move tensors to CPU, execute op, return outputs to originating device.
    Logs a warning when TORCH_SPYRE_FALLBACK_WARN=1 (default).
    """
    # STEP 1: Remember where tensors came from
    source_device = next(
        (a.device for a in args if isinstance(a, torch.Tensor)), 
        torch.device("cpu")
    )
    # Example: source_device = "spyre:0"
    
    # STEP 2: Move all tensors to CPU
    def to_cpu(x):
        return x.to("cpu") if isinstance(x, torch.Tensor) else x
    
    cpu_args = [to_cpu(a) for a in args]
    cpu_kwargs = {k: to_cpu(v) for k, v in kwargs.items()}
    # Example: tensor on spyre:0 → tensor on cpu
    
    # STEP 3: Log warning (if enabled)
    if _SPYRE_FALLBACK_WARN and source_device.type == "spyre":
        logger.warning("CPU fallback triggered for op=%s", op)
    
    # STEP 4: Execute operation on CPU
    out = op(*cpu_args, **cpu_kwargs)
    # Example: torch.ops.aten.reshape(cpu_tensor, shape)
    
    # STEP 5: Move result back to original device
    def restore(x):
        return x.to(source_device) if isinstance(x, torch.Tensor) else x
    
    if isinstance(out, (tuple, list)):
        return type(out)(restore(o) for o in out)
    return restore(out)
    # Example: cpu_tensor → spyre:0 tensor


# -------------------------------------------------------------------
# aten.reshape
# -------------------------------------------------------------------

def is_supported_reshape(x, shape):
    """Returns True only if Spyre can safely handle this reshape."""
      # Check 1: Handle -1 in shape (inferred dimension)
    shape = list(shape)
    inferred = [s for s in shape if s == -1]
    
    if len(inferred) > 1:
        return False  # Multiple -1 is invalid
    
    # Check 2: Verify numel matches
    known_prod = 1
    for s in shape:
        if s != -1:
            known_prod *= int(s)
    
    if len(inferred) == 1:
        # With -1: numel must be divisible by known dimensions
        if known_prod == 0 or x.numel() % known_prod != 0:
            return False
    else:
        # Without -1: numel must match exactly
        if x.numel() != known_prod:
            return False
    
    # Check 3: Must be contiguous
    if not x.is_contiguous():
        return False
    
    return True  # All checks passed → Spyre can handle it

@register_fallback([aten.reshape.default])
def spyre__reshape(self, shape):
# DECISION POINT: Can Spyre handle this?
    if is_supported_reshape(self, shape):
        # YES → Try Spyre
        try:
            import pdb; pdb.set_trace("hiting  spyre")
            return torch.ops.aten.reshape(self, shape) 
        except Exception as e:
            # Spyre failed at runtime → fall through to CPU
            logger.debug("spyre__reshape native failed, using CPU fallback")
    
    import pdb; pdb.set_trace("hiting  cpu")
    
    # NO (or Spyre failed) → Use CPU
    return cpu_fallback(torch.ops.aten.reshape, self, shape) 


# -------------------------------------------------------------------
# aten.linear
# -------------------------------------------------------------------

def is_supported_linear(input, weight, bias):
    """Returns True only if all conditions for Spyre linear are met."""

    if not isinstance(input, torch.Tensor) or not isinstance(weight, torch.Tensor):
        return False

    if weight.dim() < 2:
        return False

    # int64 not supported on Spyre
    if input.dtype == torch.int64 or weight.dtype == torch.int64:
        return False

    # dtype must match between input and weight
    if input.dtype != weight.dtype:
        return False

    if bias is not None:
        if not isinstance(bias, torch.Tensor):
            return False
        if bias.dtype != input.dtype:
            return False

    # linear: (..., in_features) @ (out_features, in_features).T
    if input.size(-1) != weight.size(-1):
        return False

    # empty tensors not supported
    if input.numel() == 0 or weight.numel() == 0:
        return False

    # non-contiguous layouts not supported (includes channels_last implicitly)
    if not input.is_contiguous() or not weight.is_contiguous():
        return False

    return True


@register_fallback([aten.linear.default])
def spyre__linear(input, weight, bias=None):
    if is_supported_linear(input, weight, bias):
        try:
            return torch.ops.aten.linear.default(input, weight, bias)
        except Exception as e:
            logger.debug("spyre__linear native failed (%s), using CPU fallback", e)

    return cpu_fallback(torch.nn.functional.linear, input, weight, bias)


# -------------------------------------------------------------------
# aten.clone
# -------------------------------------------------------------------

def is_supported_clone(x, memory_format=torch.preserve_format):
    """Returns True only if Spyre can safely handle this clone."""

    if not isinstance(x, torch.Tensor):
        return False

    # Expanded (stride-0) tensors not supported
    if any(s == 0 for s in x.stride()):
        return False

    if not x.is_contiguous():
        return False

    if x.numel() == 0:
        return False

    # Non-default memory formats may not be supported
    if memory_format not in (torch.preserve_format, torch.contiguous_format):
        return False

    return True


@register_fallback([aten.clone.default])
def spyre__clone(x, memory_format=torch.preserve_format):
    if is_supported_clone(x, memory_format):
        try:
            return torch.ops.aten.clone.default(x, memory_format=memory_format)
        except Exception as e:
            logger.debug("spyre__clone native failed (%s), using CPU fallback", e)

    return cpu_fallback(torch.ops.aten.clone.default, x, memory_format=memory_format)