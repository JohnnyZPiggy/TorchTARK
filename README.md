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

```bash
python -m experiments.tark_correctness --device cpu --output-dir outputs/tark_cpu
python -m experiments.tark_correctness --device cuda --output-dir outputs/tark_cuda
```

The correctness CLI records deterministic consistent/inconsistent worked
examples, actual device, dtype and Git commit when available in `manifest.json`.
Real CUDA validation requires CUDA hardware; no performance claim is made.

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
