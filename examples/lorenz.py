import torch
import matplotlib.pyplot as plt
from dysts.flows import Lorenz
from panda.patchtst.pipeline import PatchTSTPipeline

# 1. generate 1 024‑step Lorenz trajectory
traj = Lorenz().make_trajectory(1024)  # (1024, 3)

# 2. split into context (512) and future (512)
ctx, future = traj[:512], traj[512:]

# 3. load model and prepare input
device = "cpu"
model = PatchTSTPipeline.from_pretrained(
    mode="predict",
    pretrain_path="GilpinLab/panda",
    device_map=device,             # or omit; defaults to CPU
)
ctx_t = torch.tensor(ctx, dtype=torch.float32).unsqueeze(0).to(device)

# 4. forecast 512 steps ahead
with torch.no_grad():
    pred = (
        model.predict(
            ctx_t,
            prediction_length=512,
            limit_prediction_length=False,  # allow longer forecasts
            sliding_context=True            # match notebook example
        )
        .squeeze()
        .cpu()
        .numpy()
    )

# 5. quick visual check
for k, label in enumerate("xyz"):
    plt.subplot(3, 1, k + 1)
    plt.plot(range(512), ctx[:, k], label=f"context {label}")
    plt.plot(range(512, 1024), future[:, k], label=f"true {label}")
    plt.plot(range(512, 1024), pred[:, k], "--", label=f"panda {label}")
    plt.legend(loc="upper right", fontsize=6)

plt.tight_layout()
plt.savefig("forecast.png", bbox_inches="tight")  # save the image
# plt.show()  # optional; comment out if you only want to save
