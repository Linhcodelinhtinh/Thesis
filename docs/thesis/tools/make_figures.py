"""Figures from checked local artifacts; never synthesize benchmark outcomes."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "figures"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "svg.fonttype": "none"})


def rows(name):
    with (BASE / "evidence" / name).open(encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def save(fig, name):
    for ext in ["png", "svg"]:
        fig.savefig(OUT / f"{name}.{ext}", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


fig, ax = plt.subplots(figsize=(8, 3.9))
values = rows("validation_loss.csv")
for mode, color, label in [("memory", "#1b6e75", "Mem validation: memory on"),
                           ("no_memory", "#b46a28", "Mem validation: memory off")]:
    selected = [r for r in values if r["domain"] == "mem" and r["inference_memory"] == mode]
    ax.plot([int(r["step"]) for r in selected], [float(r["loss"]) for r in selected], "o-", color=color, label=label)
selected = [r for r in values if r["domain"] == "base" and r["inference_memory"] == "memory"]
ax.plot([int(r["step"]) for r in selected], [float(r["loss"]) for r in selected], "s--", color="#626d7a", label="Base validation (both modes identical)")
ax.set(xlabel="Bước huấn luyện", ylabel="Validation flow loss", title="Cùng adapter F1; phép đo offline")
ax.grid(axis="y", alpha=.2)
ax.legend(fontsize=8, loc="best")
fig.tight_layout()
save(fig, "validation_loss")

fig, ax = plt.subplots(figsize=(8, 3.6))
values = rows("mem_test_counts.csv")
for offset, condition, color in [(-.17, "B0", "#626d7a"), (.17, "F1", "#1b6e75")]:
    selected = [next(r for r in values if r["condition"] == condition and r["task"] == task) for task in ["T1", "T3"]]
    heights = [100 * float(r["sr"]) for r in selected]
    bars = ax.bar([x + offset for x in [0, 1]], heights, width=.30, label=condition, color=color)
    for bar, r in zip(bars, selected):
        ax.annotate(f"{r['successes']}/{r['n_task_episodes']}", (bar.get_x()+bar.get_width()/2, bar.get_height()),
                    xytext=(0, 5), textcoords="offset points", ha="center")
ax.set(xticks=[0, 1], xticklabels=["T1", "T3"], ylim=(0, 100), ylabel="Success rate (%)",
       title="Kết quả local trên 10 state/task, một seed")
ax.legend(loc="upper left")
ax.grid(axis="y", alpha=.2)
fig.tight_layout()
save(fig, "mem_success")

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.set(xlim=(0, 10), ylim=(0, 6.1))
ax.axis("off")


def box(x, y, w, h, label, color="#edf3f4"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=.08", linewidth=1, edgecolor="#546773", facecolor=color))
    ax.text(x+w/2, y+h/2, label, ha="center", va="center", fontsize=9)


def arrow(start, end, label="", position=None, color="#546773", style="-"):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12, color=color, linestyle=style))
    if label and position:
        ax.text(*position, label, ha="center", va="center", fontsize=8, color=color)


box(.15, 3.1, 2.05, 1.1, "Simulator\nT1/T3 + controller")
box(3.15, 3.1, 2.0, 1.1, "Allowlist\nRGB + proprioception")
box(6.2, 3.1, 1.35, 1.1, "SmolVLA\n+ LoRA")
box(8.4, 3.1, 1.35, 1.1, "Action chunk\nH=50, s=10")
box(.15, .35, 2.05, 1.05, "Evaluator\nSR / SCR / overshot", "#f4eee4")
box(3.15, .35, 2.0, 1.05, "Detector + tracker\nBằng chứng RGB")
box(5.7, .35, 1.65, 1.05, "Event writer\nCausal snapshot")
box(8.05, .35, 1.7, 1.05, "Renderer\n≤48 token")
arrow((2.3, 3.65), (3.05, 3.65))
arrow((5.25, 3.65), (6.1, 3.65))
arrow((7.65, 3.65), (8.3, 3.65))
arrow((4.15, 3.0), (4.15, 1.5), "Ảnh + gripper", (4.85, 2.25))
arrow((5.25, .875), (5.6, .875))
arrow((7.45, .875), (7.95, .875))
arrow((8.9, 1.5), (6.9, 3.0), "Instruction + memory", (8.65, 2.45))
arrow((1.175, 3.0), (1.175, 1.5), "Trạng thái\nđặc quyền", (2.0, 2.2), color="#a66c33", style="--")
ax.plot([9.08, 9.08, 1.175, 1.175], [4.3, 5.1, 5.1, 4.8], color="#546773")
arrow((1.175, 4.8), (1.175, 4.3))
ax.text(5.15, 5.35, "Thực thi action rồi nhận quan sát kế tiếp", ha="center", fontsize=10)
ax.text(.2, 5.9, "Đường điều khiển và memory trong một episode", fontsize=12, weight="bold")
ax.text(.18, -.15, "Evaluator chỉ chấm điểm; không đưa progress vào prompt.", fontsize=9, color="#8b5a2b")
fig.tight_layout()
save(fig, "architecture")
print("Generated 3 figures in PNG and SVG from local evidence/source architecture.")
