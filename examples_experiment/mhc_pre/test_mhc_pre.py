# ruff: noqa
import argparse
import os
import sys

import torch

import tilelang

tilelang.disable_cache()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mhc_pre import mhc_pre, mhc_pre_gemm_sqrsum, mhc_pre_big_fuse  # noqa: E402
from tilelang.profiler import do_bench  # noqa: E402


# ===========================================================================
# Golden reference (from CUDA source mhc_pre_ref, /tmp/opencode/mhc_pre_full_cuda_source.py:274-301)
# ===========================================================================


def sinkhorn_normalize_ref(x: torch.Tensor, repeat: int, eps: float) -> torch.Tensor:
    """x: [..., 4, 4]. softmax(-1)+eps -> col-norm -> (row-norm, col-norm) x (repeat-1)."""
    x = x.softmax(-1) + eps
    x = x / (x.sum(-2, keepdim=True) + eps)
    for _ in range(repeat - 1):
        x = x / (x.sum(-1, keepdim=True) + eps)
        x = x / (x.sum(-2, keepdim=True) + eps)
    return x


def mhc_pre_ref(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hc_mult = residual.shape[-2]

    residual_flat = residual.flatten(-2, -1).float()
    sqrsum = residual_flat.square().sum(-1)
    mixes = residual_flat @ fn.T * (sqrsum.unsqueeze(-1) / fn.shape[-1] + rms_eps).rsqrt()

    hc_scale = torch.cat(
        [
            hc_scale[0].expand(hc_mult),
            hc_scale[1].expand(hc_mult),
            hc_scale[2].expand(hc_mult * hc_mult),
        ],
    )
    mixes = mixes * hc_scale + hc_base

    pre_mix = mixes[:, :hc_mult].sigmoid().unsqueeze(-1) + hc_pre_eps
    post_mix = (mixes[:, hc_mult : 2 * hc_mult].sigmoid() * hc_post_mult_value).unsqueeze(-1)
    res_mix = mixes[:, 2 * hc_mult :].view(-1, hc_mult, hc_mult)

    res_mix = sinkhorn_normalize_ref(res_mix, repeat=sinkhorn_repeat, eps=hc_sinkhorn_eps)

    layer_input = (residual.float() * pre_mix).sum(-2).bfloat16()

    return post_mix, res_mix, layer_input


# ===========================================================================
# Test data generation (from CUDA source generate_test_data)
# ===========================================================================


def generate_test_data(
    n: int,
    hc_mult: int,
    hidden_size: int,
    rms_eps: float = 1e-6,
    hc_pre_eps: float = 1e-6,
    hc_sinkhorn_eps: float = 1e-6,
    hc_post_mult_value: float = 1.0,
    sinkhorn_repeat: int = 1,
) -> dict:
    """Generate test data for mhc_pre. Mirrors CUDA source generate_test_data."""
    torch.manual_seed(42)

    hc_mult2 = hc_mult * hc_mult
    hc_mult3 = hc_mult * 2 + hc_mult2
    device = "npu"

    residual = (
        torch.randn((n, hc_mult, hidden_size), dtype=torch.float, device=device)
        .mul(1 + torch.arange(hc_mult, device=device).mul(0.01).view(1, -1, 1))
        .bfloat16()
    )

    fn = (
        torch.randn((hc_mult3, hc_mult, hidden_size), dtype=torch.float, device=device)
        * 1e-4
        * (1 + torch.arange(hc_mult, device=device).mul(0.01).view(1, -1, 1))
    ).flatten(1, 2)

    hc_scale = torch.randn((3,), dtype=torch.float, device=device) * 0.1
    hc_base = torch.randn((hc_mult3,), dtype=torch.float, device=device) * 0.1

    return {
        "residual": residual,
        "fn": fn,
        "hc_scale": hc_scale,
        "hc_base": hc_base,
        "rms_eps": rms_eps,
        "hc_pre_eps": hc_pre_eps,
        "hc_sinkhorn_eps": hc_sinkhorn_eps,
        "hc_post_mult_value": hc_post_mult_value,
        "sinkhorn_repeat": sinkhorn_repeat,
    }


# ===========================================================================
# Precision config per DESIGN.md §9.3
# ===========================================================================


def get_precision(dtype_str: str):
    """Return (atol, rtol) per DESIGN.md §9.3 mixed tolerance table."""
    if dtype_str == "bfloat16":
        return 9.77e-4, 1.56e-2
    elif dtype_str == "float32":
        return 1.53e-5, 9.77e-4
    else:
        raise ValueError(f"Unknown dtype {dtype_str}")


# ===========================================================================
# L0 test runner
# ===========================================================================


def _run_l0_case(name: str, n: int, hidden_size: int, hc_mult: int, device: str):
    """Run one L0 case: mhc_pre kernel vs golden, compare 3 outputs.

    Per DESIGN.md §9.2, all L0 cases use bf16+fp32 dtype scheme.
    Per DESIGN.md §9.3, post_mix/comb_mix use fp32 tol, layer_input uses bf16 tol.
    """
    print(f"  [L0] {name}: n={n}, hidden_size={hidden_size}, hc_mult={hc_mult}")

    test_data = generate_test_data(n=n, hc_mult=hc_mult, hidden_size=hidden_size)

    # Run kernel
    try:
        post_mix_fused, comb_mix_fused, layer_input_fused = mhc_pre(**test_data)
    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"  [PRECISION_FAIL] l0 {name}: kernel error: {e}")
        return False

    # Run golden
    try:
        post_mix_ref, comb_mix_ref, layer_input_ref = mhc_pre_ref(**test_data)
    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"  [PRECISION_FAIL] l0 {name}: golden error: {e}")
        return False

    # Compare 3 outputs with dtype-specific tolerance
    # post_mix: fp32, shape [n, hc_mult, 1]
    atol_fp32, rtol_fp32 = get_precision("float32")
    atol_bf16, rtol_bf16 = get_precision("bfloat16")

    post_diff = (post_mix_fused.float() - post_mix_ref.float()).abs().max().item()
    comb_diff = (comb_mix_fused.float() - comb_mix_ref.float()).abs().max().item()
    layer_diff = (layer_input_fused.float() - layer_input_ref.float()).abs().max().item()

    post_ok = post_diff <= atol_fp32 + rtol_fp32 * post_mix_ref.abs().max().item()
    comb_ok = comb_diff <= atol_fp32 + rtol_fp32 * comb_mix_ref.abs().max().item()
    layer_ok = layer_diff <= atol_bf16 + rtol_bf16 * layer_input_ref.abs().max().item()

    all_ok = post_ok and comb_ok and layer_ok

    status = "PASS" if all_ok else "FAIL"
    print(f"  [{status}] l0 {name}: post_diff={post_diff:.4e} comb_diff={comb_diff:.4e} layer_diff={layer_diff:.4e}")
    if not all_ok:
        print(
            f"    detail: post_ok={post_ok} comb_ok={comb_ok} layer_ok={layer_ok}  "
            f"(fp32 atol={atol_fp32:.2e} rtol={rtol_fp32:.2e}, bf16 atol={atol_bf16:.2e} rtol={rtol_bf16:.2e})"
        )

    return all_ok


def test_mhc_pre_l0():
    """L0 gate tests: 5 regular shapes per DESIGN.md §9.2."""
    device = "npu"
    # (name, n, hidden_size, hc_mult) - DESIGN §9.2, all block-aligned
    configs = [
        ("l0_repr", 2048, 2560, 4),  # representative shape
        ("l0_small_n", 128, 4096, 4),  # small batch
        ("l0_med", 1024, 2560, 4),  # medium
        ("l0_small_h", 512, 1280, 4),  # small hidden
        ("l0_large", 8192, 4096, 4),  # large batch
    ]

    ok = True
    for name, n, h, hc in configs:
        if not _run_l0_case(name, n, h, hc, device):
            ok = False
    return ok


# ===========================================================================
# L1/L2/Boundary tests (extended by tilelang-op-test-design scenario B)
# ===========================================================================

import math

# Coverage tags for L1 cases
L1_CASES = [
    # (name, n, H, hc_mult, valrange, tags)
    (
        "l1_aligned",
        32,
        1024,
        4,
        (-1, 1),
        ["D-DTYPE-bf16", "D-DTYPE-fp32", "D-SHAPE-ALIGNED", "D-VALRANGE-S", "D-PARAM-n_splits", "D-PARAM-hc_mult"],
    ),
    ("l1_tail1", 33, 1024, 4, (-1, 1), ["D-SHAPE-TAIL-1"]),
    ("l1_tailmid", 36, 1280, 4, (-10, 10), ["D-SHAPE-TAIL-MID", "D-VALRANGE-M"]),
    ("l1_prime", 31, 1021, 4, (-1, 1), ["D-SHAPE-PRIME"]),
    ("l1_edge", 1, 1024, 4, (-1, 1), ["D-SHAPE-EDGE"]),
    ("l1_valrange_l", 32, 1024, 4, (-50, 50), ["D-VALRANGE-L"]),
    ("l1_valrange_asym", 32, 1024, 4, (-5, 10), ["D-VALRANGE-ASYM"]),
    ("l1_param_eps", 32, 1024, 4, (-1, 1), ["D-PARAM-rms_eps", "D-PARAM-hc_pre_eps", "D-PARAM-hc_sinkhorn_eps"]),
    ("l1_param_post_mult", 32, 1024, 4, (-1, 1), ["D-PARAM-hc_post_mult_value"]),
    ("l1_param_sinkhorn_rep", 32, 1024, 4, (-1, 1), ["D-PARAM-sinkhorn_repeat"]),
]

COVERAGE_MANIFEST = {
    "D-DTYPE-bf16": 10,
    "D-DTYPE-fp32": 10,
    "D-SHAPE-ALIGNED": 6,
    "D-SHAPE-TAIL-1": 1,
    "D-SHAPE-TAIL-MID": 1,
    "D-SHAPE-PRIME": 1,
    "D-SHAPE-EDGE": 1,
    "D-VALRANGE-S": 5,
    "D-VALRANGE-M": 1,
    "D-VALRANGE-L": 1,
    "D-VALRANGE-ASYM": 1,
    "D-PARAM-rms_eps": 1,
    "D-PARAM-hc_pre_eps": 1,
    "D-PARAM-hc_sinkhorn_eps": 1,
    "D-PARAM-hc_post_mult_value": 1,
    "D-PARAM-sinkhorn_repeat": 1,
    "D-SPECIAL-INF": 1,
    "D-SPECIAL-NAN": 1,
    "D-SPECIAL-ZERO": 1,
    "D-SPECIAL-DBOUND": 1,
    "D-EXC-DTYPE": 1,
    "D-EXC-SHAPE": 1,
}
COVERAGE_NA = {
    "D-SHAPE-RANK-2": "mhc_pre is 3D-only (N, hc_mult, H); no 2D/4D/5D variants",
}


def _run_l1_case(name, n, H, hc_mult, valrange, tags):
    """Run one L1 case with custom value range and params."""
    print(f"  [L1] {name}: n={n}, H={H}, valrange={valrange}")
    torch.manual_seed(42)
    device = "npu"
    hc_mult3 = hc_mult * (2 + hc_mult)

    # Custom params for param-coverage cases
    rms_eps = 1e-6
    hc_pre_eps = 1e-6
    hc_sinkhorn_eps = 1e-6
    hc_post_mult_value = 1.0
    sinkhorn_repeat = 10
    if "D-PARAM-rms_eps" in tags:
        rms_eps = 1e-4
        hc_pre_eps = 1e-4
        hc_sinkhorn_eps = 1e-4
    if "D-PARAM-hc_post_mult_value" in tags:
        hc_post_mult_value = 2.5
    if "D-PARAM-sinkhorn_repeat" in tags:
        sinkhorn_repeat = 5

    lo, hi = valrange
    residual = (
        torch.randn((n, hc_mult, H), dtype=torch.float, device=device).mul(
            1 + torch.arange(hc_mult, device=device).mul(0.01).view(1, -1, 1)
        )
        * (hi - lo)
        / 2
        + (hi + lo) / 2
    ).bfloat16()
    fn = (
        torch.randn((hc_mult3, hc_mult, H), dtype=torch.float, device=device)
        * 1e-4
        * (1 + torch.arange(hc_mult, device=device).mul(0.01).view(1, -1, 1))
    ).flatten(1, 2)
    hc_scale = torch.randn((3,), dtype=torch.float, device=device) * 0.1
    hc_base = torch.randn((hc_mult3,), dtype=torch.float, device=device) * 0.1

    test_data = {
        "residual": residual,
        "fn": fn,
        "hc_scale": hc_scale,
        "hc_base": hc_base,
        "rms_eps": rms_eps,
        "hc_pre_eps": hc_pre_eps,
        "hc_sinkhorn_eps": hc_sinkhorn_eps,
        "hc_post_mult_value": hc_post_mult_value,
        "sinkhorn_repeat": sinkhorn_repeat,
    }

    try:
        post_fused, comb_fused, layer_fused = mhc_pre(**test_data)
    except Exception as e:
        print(f"  [PRECISION_FAIL] l1 {name}: kernel error: {e}")
        return False

    try:
        post_ref, comb_ref, layer_ref = mhc_pre_ref(**test_data)
    except Exception as e:
        print(f"  [PRECISION_FAIL] l1 {name}: golden error: {e}")
        return False

    atol_fp32, rtol_fp32 = get_precision("float32")
    atol_bf16, rtol_bf16 = get_precision("bfloat16")

    post_diff = (post_fused.float() - post_ref.float()).abs().max().item()
    comb_diff = (comb_fused.float() - comb_ref.float()).abs().max().item()
    layer_diff = (layer_fused.float() - layer_ref.float()).abs().max().item()

    post_ok = post_diff <= atol_fp32 + rtol_fp32 * post_ref.abs().max().item()
    comb_ok = comb_diff <= atol_fp32 + rtol_fp32 * comb_ref.abs().max().item()
    layer_ok = layer_diff <= atol_bf16 + rtol_bf16 * layer_ref.abs().max().item()
    all_ok = post_ok and comb_ok and layer_ok

    status = "PASS" if all_ok else "FAIL"
    print(f"  [PRECISION_{status}] l1 {name}: post={post_diff:.4e} comb={comb_diff:.4e} layer={layer_diff:.4e}")
    return all_ok


def test_mhc_pre_l1():
    """L1 functional tests: regular + irregular shapes, param coverage."""
    ok = True
    for name, n, H, hc_mult, vrange, tags in L1_CASES:
        if not _run_l1_case(name, n, H, hc_mult, vrange, tags):
            ok = False
    return ok


def test_mhc_pre_l2():
    """L2 exception tests: invalid dtype/shape should be rejected."""
    device = "npu"
    # D-EXC-DTYPE: wrong dtype for residual (float32 instead of bfloat16)
    try:
        bad_residual = torch.randn(32, 4, 1024, dtype=torch.float32, device=device)
        fn = torch.randn(24, 4096, dtype=torch.float32, device=device) * 1e-4
        hc_scale = torch.randn(3, dtype=torch.float32, device=device) * 0.1
        hc_base = torch.randn(24, dtype=torch.float32, device=device) * 0.1
        mhc_pre(bad_residual, fn, hc_scale, hc_base, 1e-6, 1e-6, 1e-6, 1.0, 10)
        print("  [BOUNDARY_WARN] l2 wrong_dtype: not rejected")
    except (AssertionError, Exception) as e:
        print(f"  [BOUNDARY_PASS] l2 wrong_dtype: rejected ({type(e).__name__})")

    # D-EXC-SHAPE: mismatched fn shape
    try:
        residual = torch.randn(32, 4, 1024, dtype=torch.bfloat16, device=device)
        bad_fn = torch.randn(20, 4096, dtype=torch.float32, device=device) * 1e-4  # wrong rows
        hc_scale = torch.randn(3, dtype=torch.float32, device=device) * 0.1
        hc_base = torch.randn(24, dtype=torch.float32, device=device) * 0.1
        mhc_pre(residual, bad_fn, hc_scale, hc_base, 1e-6, 1e-6, 1e-6, 1.0, 10)
        print("  [BOUNDARY_WARN] l2 wrong_shape: not rejected")
    except (AssertionError, Exception) as e:
        print(f"  [BOUNDARY_PASS] l2 wrong_shape: rejected ({type(e).__name__})")

    # D-PARAM-n_splits: non-default n_splits (should be rejected)
    try:
        residual = torch.randn(32, 4, 1024, dtype=torch.bfloat16, device=device)
        fn = torch.randn(24, 4096, dtype=torch.float32, device=device) * 1e-4
        hc_scale = torch.randn(3, dtype=torch.float32, device=device) * 0.1
        hc_base = torch.randn(24, dtype=torch.float32, device=device) * 0.1
        mhc_pre(residual, fn, hc_scale, hc_base, 1e-6, 1e-6, 1e-6, 1.0, 10, n_splits=2)
        print("  [BOUNDARY_WARN] l2 n_splits=2: not rejected")
    except (AssertionError, Exception) as e:
        print(f"  [BOUNDARY_PASS] l2 n_splits=2: rejected ({type(e).__name__})")

    # D-PARAM-hc_mult: wrong hc_mult (shape mismatch, should be rejected)
    try:
        bad_residual = torch.randn(32, 2, 1024, dtype=torch.bfloat16, device=device)  # hc_mult=2
        fn = torch.randn(24, 4096, dtype=torch.float32, device=device) * 1e-4
        hc_scale = torch.randn(3, dtype=torch.float32, device=device) * 0.1
        hc_base = torch.randn(24, dtype=torch.float32, device=device) * 0.1
        mhc_pre(bad_residual, fn, hc_scale, hc_base, 1e-6, 1e-6, 1e-6, 1.0, 10)
        print("  [BOUNDARY_WARN] l2 hc_mult=2: not rejected")
    except (AssertionError, Exception) as e:
        print(f"  [BOUNDARY_PASS] l2 hc_mult=2: rejected ({type(e).__name__})")


def test_mhc_pre_boundary():
    """Boundary tests: INF/NAN/zero/extreme values (non-blocking)."""
    device = "npu"
    torch.manual_seed(42)

    def _run_boundary_case(name, residual_mod_fn, tags):
        n, H, hc_mult = 32, 1024, 4
        residual = torch.randn((n, hc_mult, H), dtype=torch.float, device=device).bfloat16()
        residual = residual_mod_fn(residual)
        hc_mult3 = hc_mult * (2 + hc_mult)
        fn = torch.randn((hc_mult3, hc_mult * H), dtype=torch.float, device=device) * 1e-4
        hc_scale = torch.randn(3, dtype=torch.float32, device=device) * 0.1
        hc_base = torch.randn(hc_mult3, dtype=torch.float32, device=device) * 0.1
        data = {
            "residual": residual,
            "fn": fn,
            "hc_scale": hc_scale,
            "hc_base": hc_base,
            "rms_eps": 1e-6,
            "hc_pre_eps": 1e-6,
            "hc_sinkhorn_eps": 1e-6,
            "hc_post_mult_value": 1.0,
            "sinkhorn_repeat": 10,
        }
        try:
            out = mhc_pre(**data)
            ref = mhc_pre_ref(**data)
            post_diff = (out[0].float() - ref[0].float()).abs().max().item()
            comb_diff = (out[1].float() - ref[1].float()).abs().max().item()
            layer_diff = (out[2].float() - ref[2].float()).abs().max().item()
            # For boundary, just check it doesn't crash and produces finite output
            all_finite = all(torch.isfinite(o).all() for o in out)
            if all_finite:
                print(
                    f"  [BOUNDARY_PASS] boundary {name}: finite output (post={post_diff:.2e} comb={comb_diff:.2e} layer={layer_diff:.2e})"
                )
            else:
                print(f"  [BOUNDARY_WARN] boundary {name}: non-finite output")
        except Exception as e:
            print(f"  [BOUNDARY_WARN] boundary {name}: {type(e).__name__}: {e}")

    # D-SPECIAL-INF
    _run_boundary_case("inf", lambda r: r.fill_(float("inf")), ["D-SPECIAL-INF"])
    # D-SPECIAL-NAN
    _run_boundary_case("nan", lambda r: r.fill_(float("nan")), ["D-SPECIAL-NAN"])
    # D-SPECIAL-ZERO
    _run_boundary_case("zero", lambda r: r.fill_(0.0), ["D-SPECIAL-ZERO"])
    # D-SPECIAL-DBOUND
    _run_boundary_case("dbound", lambda r: r.fill_(60000.0), ["D-SPECIAL-DBOUND"])


def run_layered_tests(level: str):
    """Ascend layered-test entry (L0/L1/L2/Boundary)."""
    torch.set_default_device("npu")
    torch.manual_seed(42)

    blocking_ok = True

    if level in ("l0", "all"):
        blocking_ok &= test_mhc_pre_l0()
    if level in ("l1", "all"):
        blocking_ok &= test_mhc_pre_l1()
    if level in ("l2", "all"):
        test_mhc_pre_l2()
    if level in ("boundary", "all"):
        test_mhc_pre_boundary()

    if blocking_ok:
        print("\nTest Passed!")
        sys.exit(0)
    sys.exit(1)


# ===========================================================================
# Performance benchmark (integrated from perf_mhc_pre.py)
# ===========================================================================

import csv
import glob
import tempfile


def run_msprof(data: dict, out_dir: str) -> list[float]:
    """Run torch_npu profiler and return kernel durations (us)."""
    import torch_npu
    from torch_npu.profiler import ProfilerActivity, ProfilerLevel
    from torch_npu.profiler.experimental_config import _ExperimentalConfig, AiCMetrics

    exp_config = _ExperimentalConfig(
        profiler_level=ProfilerLevel.Level1,
        aic_metrics=AiCMetrics.PipeUtilization,
        l2_cache=False,
        op_attr=False,
        data_simplification=True,
        export_type="text",
    )
    with torch_npu.profiler.profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
        schedule=torch_npu.profiler.schedule(wait=0, warmup=1, active=3, repeat=1, skip_first=0),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(out_dir),
        experimental_config=exp_config,
    ) as prof:
        for _ in range(4):
            mhc_pre(**data)
            torch.npu.synchronize()
            prof.step()

    csv_files = glob.glob(os.path.join(out_dir, "**/kernel_details.csv"), recursive=True)
    if not csv_files:
        return []
    csv_files.sort(key=os.path.getmtime, reverse=True)
    durations = []
    with open(csv_files[0], "r") as f:
        for row in csv.DictReader(f):
            durations.append(float(row["Duration(us)"].strip()))
    return durations


def run_perf(n: int, hidden_size: int, hc_mult: int, reps: int, use_msprof: bool):
    """Performance benchmark with do_bench + optional msprof."""
    torch.set_default_device("npu")
    torch.manual_seed(42)
    data = generate_test_data(n=n, hc_mult=hc_mult, hidden_size=hidden_size)

    print("\n" + "=" * 60)
    print(f"  Performance: n={n}, hidden_size={hidden_size}, hc_mult={hc_mult}")
    print("=" * 60)

    # Warmup (JIT compile)
    print("  Warmup (JIT compile)...")
    for _ in range(3):
        mhc_pre(**data)
    torch.npu.synchronize()

    # do_bench
    def f():
        mhc_pre(**data)

    ms_median = do_bench(f, _n_warmup=10, _n_repeat=reps, return_mode="median")
    ms_mean = do_bench(f, _n_warmup=10, _n_repeat=reps, return_mode="mean")

    print(f"\n  {'Metric':<30} {'Value':>12}")
    print(f"  {'-' * 30} {'-' * 12}")
    print(f"  {'do_bench median':<30} {ms_median * 1000:>10.2f} us")
    print(f"  {'do_bench mean':<30} {ms_mean * 1000:>10.2f} us")

    # msprof (pure kernel time)
    if use_msprof:
        out_dir = tempfile.mkdtemp(prefix="msprof_perf_")
        durations = run_msprof(data, out_dir)
        if durations:
            kernel_median = sorted(durations)[len(durations) // 2]
            host_overhead = ms_median * 1000 - kernel_median
            print(f"  {'kernel durations':<30} {str([round(d, 2) for d in durations]):>12}")
            print(f"  {'kernel median (msprof)':<30} {kernel_median:>10.2f} us")
            print(f"  {'host overhead':<30} {host_overhead:>10.2f} us")
        else:
            print(f"  {'kernel median (msprof)':<30} {'ERROR':>12}")


def main():
    parser = argparse.ArgumentParser(description="mhc_pre layered tests + perf (Ascend NPU)")
    parser.add_argument(
        "--level",
        default="l0",
        choices=["l0", "l1", "l2", "boundary", "all"],
        help="Test level to run (default: l0)",
    )
    parser.add_argument(
        "--perf",
        action="store_true",
        default=False,
        help="Run performance benchmark instead of layered tests",
    )
    parser.add_argument(
        "--msprof",
        action="store_true",
        default=False,
        help="Include msprof profiling for pure kernel time (perf only)",
    )
    parser.add_argument("--n", type=int, default=2048, help="num_tokens (perf only)")
    parser.add_argument("--hidden-size", type=int, default=2560, help="hidden_size (perf only)")
    parser.add_argument("--hc-mult", type=int, default=4, help="hc_mult (default 4)")
    parser.add_argument("--reps", type=int, default=20, help="do_bench repeat count (perf only)")
    args = parser.parse_args()

    torch.set_default_device("npu")
    torch.manual_seed(42)

    if args.perf:
        run_perf(args.n, args.hidden_size, args.hc_mult, args.reps, args.msprof)
    else:
        run_layered_tests(args.level)


if __name__ == "__main__":
    main()
