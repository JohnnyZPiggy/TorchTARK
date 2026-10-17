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
