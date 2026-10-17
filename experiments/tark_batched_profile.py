"""Instrument eager batched TARK operators; this is not a latency benchmark."""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import subprocess

import torch

from torchtark import tark_batched


WATCHED = ("aten::index", "aten::select", "aten::sum", "aten::mul", "aten::add_",
           "aten::isfinite", "aten::all", "aten::any", "aten::item", "aten::_local_scalar_dense")
TIME_FIELDS = ("self_cpu_time_total", "cpu_time_total", "self_device_time_total", "device_time_total")


def summarize(profiler):
    operators, missing = [], set()
    for event in profiler.key_averages():
        row = dict(key=event.key, count=int(event.count))
        for field in TIME_FIELDS:
            value = getattr(event, field, None)
            if value is None:
                missing.add(field)
                row[field] = None
            else:
                value = float(value)
                if not math.isfinite(value) or value < 0:
                    raise RuntimeError(f"Invalid profiler time: {field}")
                row[field] = value
        operators.append(row)
    if not operators or any(row["self_cpu_time_total"] is None for row in operators):
        raise RuntimeError("Profiler CPU operator summaries unsupported or empty")
    operators.sort(key=lambda row: row["self_cpu_time_total"], reverse=True)
    totals = {field: (None if field in missing else sum(row[field] for row in operators))
              for field in TIME_FIELDS}
    return dict(top_operators=operators[:30], sorted_by="self_cpu_time_total descending",
                watched_operators=[row for row in operators if row["key"] in WATCHED],
                absent_watched_operators=[key for key in WATCHED if not any(row["key"] == key for row in operators)],
                totals=totals, missing_api_fields=sorted(missing),
                operator_group_count=len(operators), operator_call_count=sum(row["count"] for row in operators),
                gpu_kernel_events=dict(status="unavailable", count=None, total_time_us=None,
                    reason="No portable kernel-only event classification used; device operator totals are not kernel counts"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chain-counts", default="1,128,512")
    for flag, default in (("n-rows", 1024), ("n-cols", 64), ("num-steps", 100),
                          ("warmup", 2), ("seed", 7)):
        parser.add_argument(f"--{flag}", default=str(default))
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1], capture_output=True,
            text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    m = dict(status="failure", exit_status=1, timestamp_utc=datetime.now(timezone.utc).isoformat(),
             source_commit=commit, pytorch_version=torch.__version__, python_version=platform.python_version(),
             selected_device=args.device, actual_tensor_device=None, gpu_name=None,
             cuda_available=torch.cuda.is_available(), dtype="torch.float32", operator_time_units="microseconds",
             parameters={key: getattr(args, key) for key in
                         ("n_rows", "n_cols", "num_steps", "warmup", "seed", "chain_counts")},
             profiler_settings=dict(record_shapes=False, profile_memory=False, with_stack=False,
                                    calls_per_K=1, trace_export=False), profiles=[],
             random_policy="One seeded CPU generator: float32 randn A then b, norm-squared replacement multinomial max(K)*T rows reshaped [max(K),T]; per-K prefix; transfer before profiling",
             synchronization="CUDA synchronize before entering and after exiting each profiler context, outside profiled solver; warmups outside context",
             interpretation="Instrumented operator costs are not benchmark speedups. Inclusive CPU/device totals overlap and are not additive wall time. Python-loop overhead cannot be isolated by subtraction. Indexing, reductions, dispatch and host synchronization are hypotheses until real device evidence. T=100 is not historical T=500 latency.",
             traces=[], kernel_metrics_policy="unavailable unless reliably classified; never infer kernels from operator calls")
    try:
        p = {key: int(value) for key, value in m["parameters"].items() if key != "chain_counts"}
        counts = [int(value) for value in args.chain_counts.split(",")]
        p["chain_counts"] = counts
        m["parameters"] = p
        if (not counts or any(k < 1 for k in counts) or len(set(counts)) != len(counts)
                or any(p[key] < 1 for key in ("n_rows", "n_cols", "num_steps"))
                or not 0 <= p["warmup"] <= 10 or not 0 <= p["seed"] < 2**63):
            raise ValueError("Distinct positive K/sizes/steps, warmup in [0,10], seed in [0,2**63) required")
        T, d, Kmax = p["num_steps"], p["n_cols"], max(counts)
        if (len(counts) > 4 or T > 100 or Kmax > 512 or p["n_rows"] > 100_000
                or p["n_rows"] * d > 1_000_000 or Kmax * T > 51_200
                or Kmax * d > 131_072 or sum(counts) * T * d * (p["warmup"]+1) > 200_000_000):
            raise ValueError("Requested profiler allocation/event/work budget exceeds limits")
        if args.device == "cuda" and not m["cuda_available"]:
            raise RuntimeError("CUDA requested but unavailable; no CPU fallback")
        activities = [torch.profiler.ProfilerActivity.CPU]
        if args.device == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        supported = torch.profiler.supported_activities()
        if not set(activities).issubset(supported):
            raise RuntimeError("Requested profiler activities unsupported")
        m["profiler_settings"]["activities"] = [str(a) for a in activities]
        device = torch.device(args.device)
        if device.type == "cuda":
            m["gpu_name"] = torch.cuda.get_device_name(device)
        rng = torch.Generator(device="cpu").manual_seed(p["seed"])
        A = torch.randn(p["n_rows"], d, generator=rng, dtype=torch.float32, device="cpu")
        b = torch.randn(p["n_rows"], generator=rng, dtype=torch.float32, device="cpu")
        w = A.square().sum(1)
        rows = torch.multinomial(w/w.sum(), Kmax*T, replacement=True, generator=rng).reshape(Kmax, T)
        A, b, rows = A.to(device), b.to(device), rows.to(device)
        for K in counts:
            reference = None
            selected_rows = rows[:K]

            def check(result):
                nonlocal reference
                if (result.shape != (K, d) or result.dtype != A.dtype or result.device != A.device
                        or not torch.isfinite(result).all()):
                    raise RuntimeError("Invalid solver output shape/dtype/device/finiteness")
                if reference is not None and not torch.allclose(result, reference, rtol=1e-5, atol=1e-6):
                    raise RuntimeError("Warmup/profile output disagreement")
                reference = result.clone()

            for _ in range(p["warmup"]):
                check(tark_batched(A, b, selected_rows))
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            with torch.profiler.profile(activities=activities, record_shapes=False,
                                        profile_memory=False, with_stack=False) as prof:
                result = tark_batched(A, b, selected_rows)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            check(result)
            checksum = result.double().sum().item()
            if not math.isfinite(checksum):
                raise RuntimeError("Nonfinite output checksum")
            report = summarize(prof)
            report.update(chains=K, output_shape=list(result.shape), output_dtype=str(result.dtype),
                          output_finite=True, output_checksum=checksum, actual_tensor_device=str(result.device))
            m["profiles"].append(report)
            m["actual_tensor_device"] = str(result.device)
        m.update(status="success", exit_status=0)
    except (ValueError, RuntimeError, TypeError, OverflowError, AttributeError, NotImplementedError) as error:
        m["error"] = str(error)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(m, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"{m['status']}: device={args.device}, profiled_K={[p['chains'] for p in m['profiles']]}")
    if "error" in m:
        print(m["error"])
    return m["exit_status"]


if __name__ == "__main__":
    raise SystemExit(main())
