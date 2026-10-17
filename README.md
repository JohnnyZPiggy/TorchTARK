# TorchTARK

TorchTARK is a project to implement the Tail-Averaged Randomized Kaczmarz algorithm in PyTorch, with CPU and GPU support.

The v0.1 solver implements dense, real, single-row TARK on CPU and CUDA using
native PyTorch operations. It is a non-autograd correctness baseline.

Install the package with `python -m pip install -e ".[test]"`.

```python
from torchtark import tark

# A: (n,d), b: (n,), matching float32/float64 dtype and device
x = tark(A, b, num_steps=1000)  # default burn_in=500
```

`burn_in=B` averages the post-update iterates `x_(B+1), ..., x_T`.
`x0` defaults to zeros. An optional matching-device `torch.Generator` controls
squared-row-norm sampling with replacement. Alternatively, `row_indices` is a
1-D integer tensor of length `num_steps` on `A.device` for fixed-row replay;
it cannot be combined with `generator` or select zero-norm rows. Inputs are
never mutated. CPU and CUDA random streams need not match.

For independent chains sharing A/b, `from torchtark import tark_batched` and
call `Y = tark_batched(A, b, row_indices, burn_in=None, x0=None)`.
Required integer `row_indices` has shape `[K,T]` on A.device; optional x0 has
shape `[K,d]` with matching dtype/device. Output `[K,d]` averages each chain's
post-update tail, default burn-in T//2. The non-autograd solver vectorizes only
the chain axis, preserves inputs, and rejects zero-norm selected rows. Supply
your own row sequences; no automatic sampling is provided in this API.
This is independent-chain replay, not block RK or single-chain acceleration.

```bash
python -m experiments.tark_batched_correctness --device cpu --output-dir outputs/batched_cpu
python -m experiments.tark_batched_correctness --device cuda --output-dir outputs/batched_cuda
```

The CLI compares K=1/K=3 worked examples against stacked scalar calls and
writes device/commit/comparison evidence. Real CUDA validation is coordinator-run;
no batched throughput or speedup claim is made.

Measure independent-chain batch latency and aggregate throughput:

```bash
python -m experiments.tark_batched_benchmark --device cpu --output-dir outputs/batched_bench_cpu --n-rows 1024 --n-cols 64 --num-steps 500 --chain-counts 1,8,32,128,512 --warmup 2 --repeats 5 --seed 7
```

Use `--device cuda` for coordinator-run GPU measurements. CPU float32 A/b and
max-K weighted row histories are generated once with one seeded CPU generator;
each requested K uses a prefix, transferred before timing. Only public
`tark_batched` (including validation, zero x0 and half burn-in) is timed;
CUDA synchronizes before/after every measured call. Warmups, generation,
transfers, checks and reporting are excluded. JSON includes per-K raw/summary
latency in ms, chains/s, chain-steps/s, output diagnostics and source metadata.
Duplicate/nonpositive counts fail. Limits: 32 K values, 10M matrix elements,
1M rows, 2M max-K×T and max-K×d, 100k steps, 100 total calls/K, 5M aggregate
chain-steps including warmups, and 500M chain-step×column work units.
Partial failures remain failures; external termination may prevent a manifest.
K=1 still uses the batched solver and is not comparable with scalar `tark`
latency. Compare matched CPU/GPU batched workloads in the same runtime;
throughput is not single-chain speedup. GPU scaling evidence remains pending
coordinator execution; no acceleration is assumed.

Inspect eager batched operators (instrumentation, not a benchmark):

```bash
python -m experiments.tark_batched_profile --device cpu --output-dir outputs/profile_cpu --n-rows 1024 --n-cols 64 --num-steps 100 --chain-counts 1,128,512 --warmup 2 --seed 7
```

Use `--device cuda` for actual coordinator GPU profiling. One separate
`torch.profiler` context per K records one unchanged public solver call, CPU
activities plus CUDA when supported; shapes/memory/stacks disabled. Seeded CPU
float32 inputs and max-K weighted row histories are transferred beforehand;
warmups and CUDA synchronization boundaries are outside the profiled call.
Manifests contain top 30 operators sorted by self CPU time, requested diagnostic
operators when present, counts and time totals in microseconds. Missing device
API fields are null; kernel-only counts are explicitly unavailable rather than
inferred from operator calls. Source SHA is null when Git is unavailable.
Caps: four K values, K<=512, T<=100, warmup<=10, n<=100k, n*d<=1M,
Kmax*T<=51200, Kmax*d<=131072, and sum(K)*T*d*(warmup+1)<=200M.
No Chrome traces, cProfile or unprofiled timing are exported in this minimal CLI.
Thus no trace upload/storage budget is required; profiler events are bounded by
the workload limits. Output is local only; no credentials or token access.
Review any exported metadata before sharing. Unsupported profiler APIs or
missing CUDA fail with a failure manifest when possible; external termination
cannot guarantee one. Existing untracked files are never managed by this CLI.

Inclusive operator costs overlap and cannot be added into wall time or
subtracted to isolate Python-loop overhead. Indexing, dispatch and implicit
host synchronization are hypotheses until measured on actual CUDA. T=100
profiling is not comparable to historical T=500 latency. Use the reviewed
batched benchmark CLI for comparative performance; GPU profiling remains
pending coordinator execution. No compile/custom kernels or optimization here.

```bash
python -m experiments.tark_correctness --device cpu --output-dir outputs/tark_cpu
python -m experiments.tark_correctness --device cuda --output-dir outputs/tark_cuda
```

The correctness CLI records deterministic consistent/inconsistent worked
examples, actual device, dtype and Git commit when available in `manifest.json`.
Real CUDA validation requires CUDA hardware; no performance claim is made.

Run the eager fixed-row baseline benchmark (use `--device cuda` for CUDA):

```bash
python -m experiments.tark_benchmark --device cpu --output-dir outputs/bench_cpu --n-rows 1024 --n-cols 64 --num-steps 500 --warmup 2 --repeats 5 --seed 7
```

CPU-seeded float32 normal inputs and norm-squared sampled row indices are
generated once, then transferred before timing. Only the public solver call
(including its validation, zero initialization and default half burn-in) is
timed with `perf_counter`; CUDA synchronizes before/after every measured call.
Generation, transfers, stochastic index generation, output checks and reporting
are excluded: this is fixed-row replay, not end-to-end stochastic sampling.
Each repeat uses identical data/indices and checks output agreement. Manifests
include all durations, summary statistics, parameters, commit and environment.
Limits: 10M matrix elements, 1M rows, 100k steps, 100 total calls and 100M
step×call×column work units. Failures return nonzero and write a manifest when
possible; external process termination cannot guarantee a manifest. GPU results
remain pending actual coordinator runs; no comparative acceleration is claimed.

Install dependencies with `python -m pip install -r requirements.txt`.
From the repository root, run the device smoke check:

```bash
python -m experiments.smoke --device cpu --output-dir outputs/cpu
python -m experiments.smoke --device cuda --output-dir outputs/cuda
python -m pytest -q
```

Each run writes `manifest.json` with device, environment, seed, and numerical
check results. CUDA requests fail with a nonzero exit code when CUDA cannot run;
they never fall back to CPU. This is a workflow smoke check, not a benchmark.
