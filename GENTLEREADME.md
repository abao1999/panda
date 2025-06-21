# Quickstart Guide for Panda: Patched Attention for Non-linear Dynamics

A minimal “Gentle” README to get started with the Panda foundation model.

## Installation

Panda supports CPU-only inference and testing. Clone the repository and synchronise dependencies using the `uv` tool:

```bash
git clone https://github.com/abao1999/panda.git
uv lock
uv sync
```

## Running the Lorenz Example

Run the Lorenz forecasting example to see Panda in action:

```bash
uv run examples/lorenz.py
```

## Overview

Panda is a pretrained Transformer “physics head” for chaotic systems. Below is an opinionated take on where Panda can accelerate real-world workflows.

---

### Key Idea

Trained purely on synthetic trajectories from ~20 000 chaotic ODEs discovered via an evolutionary algorithm, Panda generalises zero-shot to:

- Unseen low-dimensional chaotic systems  
- Noisy experimental data (e.g., double-pendulum, analog oscillator circuits)  
- High-dimensional PDEs (e.g., Kuramoto–Sivashinsky, von Kármán vortex streets)  

All in a compact (~21 M parameter) model ready for downstream reuse.

### Model Architecture & Training

- **PatchTST-style Transformer encoder** with channel attention for cross-variable couplings  
- **Dynamical embeddings** (Takens delay, random Fourier, polynomial features)  
- Two heads: fixed-horizon forecaster and masked-patch completer (MLM-style)  
- Trained on 4096-step windows from chaotic attractors (e.g., Lorenz, Rössler, …)

### Emergent Capabilities

| Capability                         | Evidence                                                             | Impact                                           |
|------------------------------------|----------------------------------------------------------------------|--------------------------------------------------|
| Zero-shot forecasting of unseen ODEs | MAE/sMAPE beats Chronos and other foundation models                 | Eliminates per-system retraining                 |
| Experimental data forecasting      | Double pendulum, C. elegans “eigen-worm”, analog oscillator circuits | Bridges simulation to lab/field                  |
| PDE generalisation                 | Kuramoto–Sivashinsky & von Kármán vortex street predictions          | Enables CFD-class workflows at minimal cost      |
| Neural scaling law                 | Performance scales with number of systems (not sample count)         | Guides data-generation strategy                  |
| Interpretable attention motifs     | Toeplitz, selector, resonance patterns align with dynamics structure | Enhances trust & diagnostics                     |

## Concrete Use Cases

1. **Surrogate CFD & Aeroelastic Loads**  
   Fast previews of vortex-shedding forces; integrate with solvers for 10× speed-up over URANS/LES.

2. **Real-time Digital Twins & Predictive Maintenance**  
   Zero-shot ODE skill enables rapid anomaly forecasting on turbines, pumps, and other assets.

3. **Control of Under-actuated Robots & Test Rigs**  
   Use Panda in Model-Predictive Control or differentiable RL for double-pendulum and cart-pole systems.

4. **Design-space Exploration & Optimisation**  
   Cluster trajectory embeddings to search for bifurcations or interesting dynamics without full simulation.

5. **Multi-physics Sensor Fusion**  
   Channel-attention for electrical, thermal, and mechanical state couplings in real-world plants.

6. **Education & Rapid Prototyping**  
   Interactive “chaos sandbox” for visual demos using evolutionary attractor scripts.

## Quickstart Steps

| Step | Action                                                                                               |
|------|------------------------------------------------------------------------------------------------------|
| 1    | Clone & test (run Lorenz example)                                                                    |
| 2    | Export as TorchScript or ONNX for GPU/edge deployment                                                |
| 3    | Fine-tune (optional) with LoRA (~5 k params) on 5–10 k trajectories                                   |
| 4    | Embed in digital-twin loop for real-time monitoring                                                   |
| 5    | Monitor failure modes and revert to solver when forecast error exceeds threshold                     |

## Limitations & Open Questions

- **State dimensionality**: Trained on ≤10-D ODEs; high-DoF turbulent flows may need sparse attention or hierarchical tiling.  
- **Noise & non-stationarity**: Field data sensor drift may require light fine-tuning.  
- **Certification & explainability**: Attention maps help, but physics-based verification is still needed for regulated domains.

---

## Final Thoughts

Think of Panda as the “ResNet-50 of chaos”: small, fast, surprisingly general, and primed for downstream integration. Early adopters can compress weeks of simulation into minutes of inference.

---

## Performance Summary

- **Memory (CPU inference)**: ~85 MB weights + ~30 MB activations (seq=512)  
- **Compute**: ~1.5×10^9 FLOPs per 512-step window (8 layers, hidden=384, 8 heads)  
- **Optimisations**: PyTorch ≥1.12 Fast-Transformer, ONNX/TorchScript fusion for 2–4× speed-ups

---

## Stay Tuned: Two 30‑Minute Extensions

1 – **Real hardware “shake test”**  
Record 10 s of accelerometer data from your phone taped to a desk fan (or any vibratory gadget), resample to 200 Hz, and run the same script. Panda treats each accelerometer axis as a channel and will anticipate the next few hundred milliseconds of motion—ideal for a digital‑twin hunch. (No citation; try it!)

2 – **Double‑pendulum video dataset**  
IBM’s open **Double Pendulum Chaotic** CSV gives x,y angles at 250 fps. Load two channels, normalise to [–1,1], then run Panda’s forecast head. You’ll get a short‑horizon prediction that matches the swing for several seconds—mirroring the authors’ lab result.

---

## Aha! Demo: Lorenz Zero-shot Forecast

Below is a hands-on, laptop-friendly “aha!” demo that lets you feel Panda’s zero-shot forecasting power in under 10 minutes:

```python
import torch
import matplotlib.pyplot as plt
from dysts.flows import Lorenz
from panda.patchtst.pipeline import PatchTSTPipeline

# 1. Simulate Lorenz for 1024 steps
traj = Lorenz().make_trajectory(1024)  # shape: (1024, 3)
ctx, true_future = traj[:512], traj[512:]

# 2. Load model on CPU
device = "cpu"
model = PatchTSTPipeline.from_pretrained(
    "GilpinLab/panda",
    mode="predict",
    device_map=device
)
ctx_t = torch.tensor(ctx, dtype=torch.float32).unsqueeze(0).to(device)

# 3. Forecast next 512 steps
with torch.no_grad():
    pred = (
        model.predict(
            ctx_t,
            prediction_length=512,
            limit_prediction_length=False,
            sliding_context=True
        )
        .squeeze()
        .cpu()
        .numpy()
    )

# 4. Plot results
fig, axes = plt.subplots(3, 1, figsize=(6, 4))
for i, label in enumerate("xyz"):
    axes[i].plot(range(512), ctx[:, i], label=f"context {label}")
    axes[i].plot(range(512, 1024), true_future[:, i], label=f"true {label}")
    axes[i].plot(range(512, 1024), pred[:, i], "--", label=f"Panda {label}")
    axes[i].legend(fontsize=6)
fig.tight_layout()
plt.savefig("forecast.png", bbox_inches="tight")
# plt.show()
```
