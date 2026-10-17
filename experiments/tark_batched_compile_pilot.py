"""Bounded opt-in compiler feasibility probe; unchanged public eager solver."""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import subprocess
import time

import torch

from torchtark import tark_batched
from experiments.tark_benchmark import timed_call


def counters():
    try:
        from torch._dynamo.utils import counters as state
        return {group: {str(k): int(v) for k, v in values.items()}
                for group, values in state.items()}
    except (ImportError, AttributeError, TypeError, ValueError):
        return None


def counter_delta(before, after):
    if before is None or after is None:
        return None
    return {group: {key: value - before.get(group, {}).get(key, 0)
                    for key, value in values.items()
                    if value != before.get(group, {}).get(key, 0)}
            for group, values in after.items()}


def compare(result, expected, inputs, snapshots):
    if (result.shape != expected.shape or result.device != expected.device
            or result.dtype != expected.dtype or not torch.isfinite(result).all()):
        raise RuntimeError("Compiled output shape/device/dtype/finiteness mismatch")
    if any(not torch.equal(a, b) for a, b in zip(inputs, snapshots)):
        raise RuntimeError("Input mutation detected")
    error = (result - expected).abs()
    rel = error / expected.abs().clamp_min(torch.finfo(expected.dtype).eps)
    atol, rtol = (1e-5, 1e-4) if expected.dtype == torch.float32 else (1e-10, 1e-8)
    if not torch.allclose(result, expected, atol=atol, rtol=rtol):
        raise RuntimeError("Compiled/eager numerical disagreement")
    return dict(max_abs_error=error.max().item(), max_relative_error=rel.max().item(),
                atol=atol, rtol=rtol, output_finite=True, inputs_unchanged=True)


def probe(A, b, rows, x0, burn_in, backend_name, fullgraph, warmup, repeats):
    from torch._dynamo.backends.registry import lookup_backend
    if torch._dynamo.config.suppress_errors:
        raise RuntimeError("suppress_errors enabled; refusing silent compiler fallback")
    backend = lookup_backend(backend_name)
    graphs = []

    def counted_backend(gm, examples):
        entry = dict(nodes=len(list(gm.graph.nodes)),
                     targets=[str(n.target) for n in gm.graph.nodes if n.op.startswith("call_")],
                     executions=0)
        compiled_graph = backend(gm, examples)
        graphs.append(entry)
        def execute(*args):
            entry["executions"] += 1
            return compiled_graph(*args)
        return execute

    inputs = [A, b, rows] + ([] if x0 is None else [x0])
    snapshots = [v.clone() for v in inputs]
    def eager():
        return tark_batched(A, b, rows, burn_in=burn_in, x0=x0)
    expected = eager()
    before = counters()
    record = dict(fullgraph=fullgraph, graphs=graphs, status="failure",
                  recompiles="unavailable: backend submissions/counter deltas reported, not unique recompile events")
    if A.device.type == "cuda":
        torch.cuda.synchronize(A.device)
    start = time.perf_counter()
    try:
        compiled = torch.compile(tark_batched, backend=counted_backend,
                                 fullgraph=fullgraph, dynamic=False)
        def call():
            return compiled(A, b, rows, burn_in=burn_in, x0=x0)
        result = call()
        if A.device.type == "cuda":
            torch.cuda.synchronize(A.device)
        record["cold_start_ms"] = (time.perf_counter() - start) * 1000
        record["comparison"] = compare(result, expected, inputs, snapshots)
        if not graphs or not any(g["executions"] for g in graphs):
            raise RuntimeError("No captured backend graph execution; refusing eager-only success")
        record["cold_counter_delta"] = counter_delta(before, counters())
        cold_graphs = len(graphs)
        for _ in range(warmup):
            compare(call(), expected, inputs, snapshots)
        steady_before = counters()
        steady_graphs = len(graphs)
        eager_ms, compiled_ms = [], []
        # Alternate matched eager/compiled calls; backend execution counting adds host overhead.
        for _ in range(repeats):
            reference, elapsed = timed_call(eager, A.device)
            compare(reference, expected, inputs, snapshots)
            eager_ms.append(elapsed)
            result, elapsed = timed_call(call, A.device)
            compare(result, expected, inputs, snapshots)
            compiled_ms.append(elapsed)
        if any(not math.isfinite(t) or t <= 0 for t in eager_ms + compiled_ms):
            raise RuntimeError("Nonpositive/nonfinite elapsed time")
        if len(graphs) != steady_graphs:
            raise RuntimeError("Backend compiled new graphs during steady-state measurement")
        record.update(status="captured_graph_execution" if fullgraph else "partial_graph_execution",
                      cold_graphs=cold_graphs, warmup_new_graphs=steady_graphs-cold_graphs,
                      steady_new_graphs=len(graphs)-steady_graphs,
                      steady_counter_delta=counter_delta(steady_before, counters()),
                      eager_elapsed_ms=eager_ms, compiled_elapsed_ms=compiled_ms,
                      eager_median_ms=statistics.median(eager_ms),
                      compiled_median_ms=statistics.median(compiled_ms),
                      actual_tensor_device=str(result.device), output_shape=list(result.shape))
    except Exception as error:
        # A feasibility probe must preserve compiler failures, never substitute eager execution.
        if A.device.type == "cuda":
            torch.cuda.synchronize(A.device)
        record.update(error_type=type(error).__name__, error=str(error)[:6000],
                      failure_elapsed_ms=(time.perf_counter()-start)*1000,
                      counter_delta=counter_delta(before, counters()))
        if fullgraph and isinstance(error, torch._dynamo.exc.Unsupported):
            record["status"] = "unsupported_fullgraph"
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=("eager", "aot_eager", "inductor"), default="aot_eager")
    parser.add_argument("--route", choices=("both", "regular", "fullgraph"), default="both")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--chain-counts", default="1,128,512")
    parser.add_argument("--nonzero-x0", action="store_true")
    for flag, default in (("n-rows", "16"), ("n-cols", "4"), ("num-steps", "4"),
                          ("burn-in", "default"), ("warmup", "1"), ("repeats", "2"), ("seed", "7")):
        parser.add_argument(f"--{flag}", default=default)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
                             capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        sha = None
    m = dict(status="failure", exit_status=1, source_commit=sha,
             timestamp_utc=datetime.now(timezone.utc).isoformat(), pytorch_version=torch.__version__,
             selected_device=args.device, cuda_available=torch.cuda.is_available(), gpu_name=None,
             backend=args.backend, mode="default", dynamic=False, route=args.route, dtype=args.dtype,
             parameters=vars(args).copy(), cases=[],
             policy="CPU seeded normal A,b, max-K weighted replacement row histories and optional nonzero x0 generated once, transferred before probes; K prefixes",
             limitations="Regular route permits graph breaks/eager regions explicitly. Captured graphs may only cover validation; inspect targets. eager/aot_eager are diagnostic backends, not optimized GPU codegen. No speedup claim. Backend counters add overhead; cold-start includes tracing/first execution, not isolated compiler-only time. Private Dynamo APIs/counters may vary; unknown metrics unavailable. No timeout guarantee for native compiler; T<=10 and small workload only.")
    m["parameters"].pop("output_dir")
    try:
        p = {key: int(getattr(args, key)) for key in ("n_rows", "n_cols", "num_steps", "warmup", "repeats", "seed")}
        Ks = [int(k) for k in args.chain_counts.split(",")]
        B = None if args.burn_in == "default" else int(args.burn_in)
        if (not Ks or len(Ks)>3 or len(set(Ks))!=len(Ks) or min(Ks)<1 or max(Ks)>512
                or not 1<=p["num_steps"]<=10 or not 1<=p["n_rows"]<=1024
                or not 1<=p["n_cols"]<=64 or not 0<=p["warmup"]<=3
                or not 1<=p["repeats"]<=5 or not 0<=p["seed"]<2**63
                or B is not None and not 0<=B<p["num_steps"]):
            raise ValueError("Pilot limits: <=3 distinct K in [1,512], n<=1024,d<=64,T<=10,warmup<=3,repeats<=5, valid seed/burn-in")
        if args.device == "cuda" and not m["cuda_available"]:
            raise RuntimeError("CUDA requested but unavailable; no CPU fallback")
        if not callable(getattr(torch, "compile", None)):
            raise RuntimeError("torch.compile unavailable")
        device, dtype = torch.device(args.device), getattr(torch, args.dtype)
        if device.type == "cuda":
            m["gpu_name"] = torch.cuda.get_device_name(device)
        rng = torch.Generator(device="cpu").manual_seed(p["seed"])
        A = torch.randn(p["n_rows"], p["n_cols"], generator=rng, dtype=dtype, device="cpu")
        b = torch.randn(p["n_rows"], generator=rng, dtype=dtype, device="cpu")
        w = A.square().sum(1)
        rows = torch.multinomial(w/w.sum(), max(Ks)*p["num_steps"], replacement=True,
                                 generator=rng).reshape(max(Ks), p["num_steps"])
        x0 = torch.randn(max(Ks), p["n_cols"], generator=rng, dtype=dtype, device="cpu") if args.nonzero_x0 else None
        A,b,rows = A.to(device),b.to(device),rows.to(device)
        if x0 is not None:
            x0 = x0.to(device)
        m["parameters"].update(p, chain_counts=Ks, burn_in=B)
        for K in Ks:
            case = dict(chains=K, probes=[])
            m["cases"].append(case)
            for strict in ([True, False] if args.route=="both" else [args.route=="fullgraph"]):
                record = probe(A,b,rows[:K],None if x0 is None else x0[:K],B,
                               args.backend,strict,p["warmup"],p["repeats"])
                case["probes"].append(record)
                if record["status"] == "failure" or (args.route=="fullgraph" and record["status"]=="unsupported_fullgraph"):
                    raise RuntimeError(f"K={K}: {record.get('error', record['status'])}")
        m.update(status="feasibility_report_complete", exit_status=0)
    except Exception as error:
        m["error"] = f"{type(error).__name__}: {error}"[:6000]
    (args.output_dir/"manifest.json").write_text(json.dumps(m,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(f"{m['status']}: device={args.device}, backend={args.backend}")
    if "error" in m:
        print(m["error"])
    return m["exit_status"]


if __name__ == "__main__":
    raise SystemExit(main())
