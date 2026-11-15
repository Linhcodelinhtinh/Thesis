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

fig, ax = plt.subplots(figsize=(10.8, 6.3))
ax.set(xlim=(0, 12), ylim=(0, 8))
ax.axis("off")


def box(x, y, w, h, label, color="#edf3f4"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=.08", linewidth=1, edgecolor="#546773", facecolor=color))
    ax.text(x+w/2, y+h/2, label, ha="center", va="center", fontsize=9)


def arrow(start, end, label="", position=None, color="#546773", style="-"):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12, color=color, linestyle=style))
    if label and position:
        ax.text(*position, label, ha="center", va="center", fontsize=8, color=color)


box(.3, 6.0, 1.65, .95, "Mô phỏng\nrobot + camera")
box(2.35, 6.0, 1.9, .95, "Quan sát RGB\n+ proprioception")
box(4.65, 6.0, 1.9, .95, "Evidence và\nidentity tracking")
box(6.95, 6.0, 1.8, .95, "Memory state\nvalidity + history")
box(9.15, 6.0, 1.7, .95, "Projection\n≤48 tokens")
box(9.15, 3.75, 1.7, .95, "VLA policy\nbackbone + adapter")
box(6.95, 3.75, 1.7, .95, "Action proposal\nchunk")
box(4.65, 3.75, 1.9, .95, "Supervisor\ncontinue / requery / abstain", "#eaf1f2")
box(2.35, 3.75, 1.9, .95, "Controller\nOSC + gripper")
box(9.3, .55, 2.0, .95, "Evaluator\nSR / SCR / endpoint", "#f4eee4")
arrow((2.0, 6.47), (2.3, 6.47))
arrow((4.3, 6.47), (4.6, 6.47))
arrow((6.6, 6.47), (6.9, 6.47))
arrow((8.8, 6.47), (9.1, 6.47))
arrow((10.0, 5.9), (10.0, 4.75), "Instruction + selected facts", (10.8, 5.35))
arrow((9.1, 4.22), (8.7, 4.22))
arrow((6.9, 4.22), (6.6, 4.22))
arrow((4.6, 4.22), (4.3, 4.22))
arrow((7.85, 5.9), (5.8, 4.8), "Fact/version changes", (6.35, 5.25), color="#287f83")
# Route executed control around the diagram so the feedback path does not cross nodes.
arrow((3.3, 3.7), (3.3, 2.05))
ax.plot([3.3, .08, .08], [2.05, 2.05, 6.48], color="#546773", lw=1.1)
arrow((.08, 6.48), (.28, 6.48))
ax.text(5.7, 1.78, "Hành động đã thi hành → quan sát tiếp theo", ha="center", fontsize=8, color="#546773")
ax.plot([1.1, 1.1, 9.2], [5.95, .95, .95], color="#a66c33", lw=1.1, linestyle="--")
arrow((9.2, .95), (9.3, .95), color="#a66c33", style="--")
ax.text(5.1, 1.08, "Trạng thái đặc quyền, chỉ cho evaluator", ha="center", fontsize=8, color="#a66c33")
ax.text(.25, 7.45, "Bộ nhớ sự kiện có bằng chứng và điều phối truy vấn", fontsize=12, weight="bold")
ax.text(9.2, .15, "Evaluator chỉ chấm; không cấp nhãn cho policy.", fontsize=8, color="#8b5a2b")
fig.tight_layout()
save(fig, "architecture")

fig, axes = plt.subplots(1, 3, figsize=(10, 3.8), sharey=True)
panels = [
    ("OM / OS · Chu kỳ", [(1.0, "rời đĩa", "#477a99", "top"), (3.0, "quay về", "#287f83", "bottom"),
                           (4.1, "evidence sẵn có", "#287f83", "top")], 5.4,
     "Query sau khi evidence\nđã bền qua quan sát"),
    ("OR · Danh tính", [(1.0, "quan hệ A→P", "#477a99", "top"), (2.5, "mất dấu", "#a66c33", "bottom"),
                         (4.0, "ID chưa rõ", "#a66c33", "top")], 5.4,
     "Không gán lại ID khi\nbằng chứng chưa phân giải"),
    ("OO · Vật chứa", [(1.0, "vào giỏ", "#477a99", "top"), (2.4, "bị che", "#a66c33", "bottom"),
                        (4.2, "giỏ dịch chuyển", "#287f83", "top")], 5.4,
     "Quan hệ hiện tại có thể\nunknown khi vật bị che"),
]
for ax, (title, events, query_x, note) in zip(axes, panels):
    ax.set(xlim=(0, 6), ylim=(0, 3.2), title=title, xlabel="Chỉ số quan sát (sơ đồ)")
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.axhline(1.55, color="#788995", lw=1.2)
    for x, label, color, side in events:
        ax.plot([x, x], [1.35, 1.75], color=color, lw=2)
        ax.scatter([x], [1.55], color=color, s=30, zorder=3)
        ax.text(x, 2.0 if side == "top" else .95, label, ha="center", va="center", fontsize=8, color=color)
    ax.axvline(query_x, color="#1b6e75", ls="--", lw=1.2)
    ax.text(query_x, 2.7, "policy query", ha="center", fontsize=8, color="#1b6e75")
    ax.text(3, .25, note, ha="center", va="center", fontsize=8)
fig.suptitle("Thời điểm xảy ra, evidence khả dụng và query (minh họa khái niệm)", fontsize=11, weight="bold")
fig.tight_layout(rect=(0, 0, 1, .91))
save(fig, "memory_timeline")
print("Generated architecture and timeline schematics plus two figures from local pilot evidence.")
