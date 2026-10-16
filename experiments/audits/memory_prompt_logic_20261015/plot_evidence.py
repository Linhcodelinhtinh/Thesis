"""Plot recorded telemetry only; no policy or simulator execution."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).parent
KEY = "libero_10_task0/init_0"


def read(run):
    return [json.loads(x) for x in (ROOT / "experiments/results" / run / KEY / "model_output.jsonl").read_text().splitlines()]


compact = read("v2_memory/prompt_shoren_obs_text")
off = read("libero-10-wsl")
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), constrained_layout=True)
for data, label, color in [(off, "Original instruction", "#2374ab"),
                           (compact, "Instruction + static memory", "#d26336")]:
    a = data[:201]
    axes[0].plot([x["step"] for x in a], [100*x["contact_diagnostics"]["dist_eef_to_object"] for x in a],
                 label=label, color=color, linewidth=2)
axes[0].set(title="A. Paths diverge before the first writer event", xlabel="Completed action step",
            ylabel="EEF to alphabet-soup body center (cm)")
axes[0].legend(frameon=False, loc="upper right")
axes[0].text(.02, .04, "Same recorded seed and settling snapshot.\nFirst inferred grasp in memory run: step 295.",
             transform=axes[0].transAxes, fontsize=9,
             bbox={"facecolor": "white", "alpha": .9, "edgecolor": "none"})
axes[0].grid(alpha=.2)
a = compact[293:302]
axes[1].axhspan(15, 65, color="#f6bd60", alpha=.25, label="Writer's grasp interval")
axes[1].plot([x["step"] for x in a], [1000*sum(abs(v) for v in x["gripper_qpos_actual"]) for x in a],
             marker="o", color="#7048a5", linewidth=2, label="Recorded finger gap")
axes[1].axvline(295, linestyle="--", color="#666666", linewidth=1)
axes[1].axvline(300, linestyle="--", color="#666666", linewidth=1)
axes[1].text(295.1, 72, '"grasp"', fontsize=9)
axes[1].text(300.1, 72, '"slip"', fontsize=9)
axes[1].text(.03, .06, "Named target remains 11.5–12.1 cm away.\nNo logged bilateral contact; target height unchanged.",
             transform=axes[1].transAxes, fontsize=9)
axes[1].set(title="B. Closure alone generates grasp → slip", xlabel="Completed action step",
            ylabel="Finger gap (mm)", ylim=(0, 100), xlim=(292.5, 301.6))
axes[1].legend(frameon=False, loc="upper right", fontsize=8)
axes[1].grid(alpha=.2)
fig.suptitle("Existing-log audit: LIBERO-10 task 0, initial state 0", fontsize=14)
fig.savefig(OUT / "telemetry_evidence.png", dpi=160)
fig.savefig(OUT / "telemetry_evidence.pdf")
print(OUT / "telemetry_evidence.png")
