# TorchTARK

TorchTARK is a project to implement the Tail-Averaged Randomized Kaczmarz algorithm in PyTorch, with CPU and GPU support.

The project is in its initial setup stage; the algorithm is not implemented yet.

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
