"""Build the thesis diagrams and extract verified rollout frames with FFmpeg."""
import hashlib
import json
from math import cos, pi, sin
from pathlib import Path
import runpy
import subprocess
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path as DrawingPath

THESIS = Path(__file__).resolve().parents[1]
REPO = THESIS.parents[1]
FIGURES = THESIS / "figures"
EVIDENCE = THESIS / "evidence"
STYLE = runpy.run_path(str(THESIS / "chapter3_outline/draw_figures.py"))
canvas, box, arrow, label = (STYLE[key] for key in ("canvas", "box", "arrow", "label"))
INK, BLUE, GREEN, TEAL, GRAY = (STYLE[key] for key in ("INK", "BLUE", "GREEN", "TEAL", "GRAY"))
RED, ROSE = "#a5323c", "#fce9e7"
ICE, ICE_INK = "#dcefff", "#21658d"
RAW_STEP, MEMORY_STEP = 0, 312
plt.rcParams.update({"font.family": "Constantia", "pdf.fonttype": 42})


def save(fig, name):
    fig.savefig(FIGURES / name, facecolor="white",
                metadata={"Creator": "build_thesis_figures.py", "CreationDate": None})
    latex_figures = THESIS / "latex/figures"
    if latex_figures.is_dir():
        fig.savefig(latex_figures / name, facecolor="white",
                    metadata={"Creator": "build_thesis_figures.py", "CreationDate": None})
    plt.close(fig)


@plt.rc_context({
    "font.family": "Constantia",
    "mathtext.fontset": "custom",
    "mathtext.rm": "Constantia",
    "mathtext.it": "Constantia:style=italic",
    "mathtext.fallback": None,
})
def architecture():
    fig, ax = canvas(4.10)
    modules = [
        (.15, 2.57, 2.60, "Bộ nhớ sự kiện", "Ghi và chiếu dữ kiện", GREEN),
        (3.77, 2.57, 2.46, "Chính sách VLA", "Sinh đoạn hành động", BLUE),
        (7.22, 2.57, 2.63, "Bộ điều phối", "Thực thi hoặc gọi lại VLA", ROSE),
        (7.22, .40, 2.63, "Mô phỏng robot", "Cập nhật môi trường", BLUE),
    ]
    for x, y, width, title, detail, fill in modules:
        box(ax, x, y, width, .97, "", fill)
        label(ax, x + width / 2, y + .62, title, 19.5)
        label(ax, x + width / 2, y + .28, detail, 13.5)

    for x, text in [(1.45, r"Chỉ dẫn gốc $\ell$"), (5, r"Quan sát tại truy vấn $o_q$")]:
        label(ax, x, 3.94, text, 14)
        arrow(ax, (x, 3.78), (x, 3.54))
    for start, end, text, symbol in [
        (2.75, 3.77, "Lời nhắc", r"$C_q$"),
        (6.23, 7.22, "Đoạn lệnh", r"$A_q$"),
    ]:
        arrow(ax, (start, 3.02), (end, 3.02))
        label(ax, (start + end) / 2, 3.25, text, 13)
        label(ax, (start + end) / 2, 2.81, symbol, 13)

    evidence_line = "#8b9aa4"
    ax.plot([1.86, 1.86, 8.96], [2.57, 1.60, 1.60],
            color=evidence_line, lw=1.0, ls="--")
    ax.annotate("", xy=(8.96, 2.57), xytext=(8.96, 1.60),
                arrowprops={"arrowstyle": "-|>", "color": evidence_line, "lw": 1.0,
                            "ls": "--", "shrinkA": 0, "shrinkB": 1, "mutation_scale": 11})

    ax.plot([8.65, 8.65, 5], [2.57, 2.13, 2.13], color=RED, lw=1.6, ls="--")
    ax.annotate("", xy=(5, 2.57), xytext=(5, 2.13),
                arrowprops={"arrowstyle": "-|>", "color": RED, "lw": 1.6, "ls": "--",
                            "shrinkA": 0, "shrinkB": 1, "mutation_scale": 13})
    label(ax, 6.78, 2.31, "Yêu cầu truy vấn sớm", 13.5, RED)

    arrow(ax, (9.33, 2.57), (9.33, 1.37))
    label(ax, 9.48, 1.97, r"$a_t$", 13, ha="left")
    ax.plot([7.22, .70], [.88, .88], color=TEAL, lw=1.4)
    arrow(ax, (.70, .88), (.70, 2.57), TEAL)
    label(ax, 3.84, .64, r"Quan sát sau hành động $o_{t+1}$", 14, TEAL)
    save(fig, "vong-dieu-khien.pdf")


def snowflake(ax, x, y, radius=.17):
    for arm in range(6):
        angle = arm * pi / 3
        dx, dy = cos(angle), sin(angle)
        tx, ty = x + radius * dx, y + radius * dy
        ax.plot([x, tx], [y, ty], color=ICE_INK, lw=1.25, solid_capstyle="round")
        bx, by = x + radius * .62 * dx, y + radius * .62 * dy
        for fork in (-pi / 3, pi / 3):
            ax.plot([bx, bx + radius * .36 * cos(angle + fork)],
                    [by, by + radius * .36 * sin(angle + fork)],
                    color=ICE_INK, lw=1.25, solid_capstyle="round")


def flame(ax, x, y):
    points = [(0, .23), (.12, .15), (.03, .08), (.12, .10),
              (.22, .01), (.20, -.17), (.02, -.18),
              (-.16, -.20), (-.22, -.05), (-.13, .07),
              (-.10, .02), (-.03, .08), (0, .23)]
    path = DrawingPath([(x + px, y + py) for px, py in points],
                       [DrawingPath.MOVETO] + [DrawingPath.CURVE4] * 12)
    ax.add_patch(PathPatch(path, fc="#d54d39", ec=RED, lw=.8))


def smolvla():
    fig, ax = canvas(4.35)
    ax.add_patch(Rectangle((.24, 2.44), 9.52, 1.05, fc="#f3f5f6", ec="#a9b5bd", lw=1))
    for center, text in [(1.70, "Ảnh RGB"), (5.0, "Chỉ dẫn / lời nhắc"), (8.30, "Trạng thái robot")]:
        label(ax, center, 4.04, text, 15)
        arrow(ax, (center, 3.85), (center, 3.25))
    for center, text in [(1.70, "SigLIP"), (8.30, "Chiếu trạng thái")]:
        box(ax, center - 1.17, 2.61, 2.34, .62, "", ICE)
        snowflake(ax, center - .84, 2.92)
        label(ax, center + .12, 2.92, text, 14, ICE_INK)
    box(ax, 3.83, 2.61, 2.34, .62, "Token hóa", "#ffffff", 14)
    for center in (1.70, 5.0, 8.30):
        arrow(ax, (center, 2.61), (center, 2.06))
    box(ax, .65, 1.42, 8.70, .62, "", ICE)
    snowflake(ax, 1.05, 1.73)
    label(ax, 5, 1.73, "SmolLM2   ·   Xử lý token ngôn ngữ - thị giác", 16, ICE_INK)
    arrow(ax, (5, 1.42), (5, 1.11))
    box(ax, .65, .38, 8.70, .71, "", "#ffd9d6")
    label(ax, 5, 0.73, "Action Expert   ·   Flow Matching   ·   Action chunking", 15, RED)
    # ax.plot([6.63, 6.63], [.51, .97], color="#bd676b", lw=.8)
    flame(ax, 1.05, .73)
    save(fig, "smolvla-lora.pdf")


def availability():
    fig, ax = canvas(2.70)
    ax.plot([.55, 9.30], [1.05, 1.05], color=INK, lw=2.2)
    arrow(ax, (9.30, 1.05), (9.70, 1.05))
    ax.plot([4.25, 7.15], [1.05, 1.05], color=TEAL, lw=5.0, solid_capstyle="butt", zorder=3)
    for x, symbol, title in [(1.35, r"$\tau$", "Sự kiện xảy ra"),
                              (4.25, r"$\nu$", "Xác nhận bằng chứng"),
                              (7.15, "$q$", "Truy vấn VLA")]:
        ax.scatter([x], [1.05], s=120, color="white", edgecolor=TEAL, linewidth=2.5, zorder=4)
        ax.scatter([x], [1.05], s=40, color=TEAL, zorder=5)
        label(ax, x, .48, symbol, 22)
        label(ax, x, 1.50, title, 17.5, weight="bold")

    bbox_gray = {"boxstyle": "round,pad=.32", "facecolor": "#f2f5f7", "edgecolor": "#c5cfd6", "linewidth": 1.1}
    bbox_teal = {"boxstyle": "round,pad=.32", "facecolor": "#e7f3ef", "edgecolor": "#a3d2c4", "linewidth": 1.1}
    bbox_red  = {"boxstyle": "round,pad=.32", "facecolor": ROSE,      "edgecolor": "#f5b7b1", "linewidth": 1.1}

    label(ax, 2.80, 2.18, "Chưa đủ bằng chứng", 15.5, "#4a5568", weight="bold", bbox=bbox_gray)
    label(ax, 5.70, 2.18, r"Đủ điều kiện: $\nu \leq q$", 15.5, TEAL, weight="bold", bbox=bbox_teal)
    label(ax, 8.55, 2.18, r"Chưa khả dụng tại $q$", 15.5, RED, weight="bold", bbox=bbox_red)
    save(fig, "truc-bang-chung.pdf")


def memory_prompt(query_path):
    with query_path.open(encoding="utf-8") as stream:
        prompts = {row["step"]: row["prompt"] for row in map(json.loads, stream)}
    raw = prompts[RAW_STEP]
    instruction, suffix = prompts[MEMORY_STEP].split(" Memory: ", 1)
    assert raw == instruction and "1 observed returns." in suffix
    fig, ax = canvas(3.66)
    label(ax, .48, 3.38, "Mẫu khi có dữ kiện đủ điều kiện", 15, ha="left", weight="bold")
    box(ax, .46, 2.64, 9.08, .51, "{chỉ dẫn gốc} Memory: {các câu dữ kiện}", BLUE, 15)
    label(ax, .48, 2.32, f"Chỉ dẫn gốc · bước {RAW_STEP}", 15, ha="left", weight="bold")
    box(ax, .46, 1.58, 9.08, .51, "", "#f3f5f6")
    label(ax, .69, 1.83, raw, 14, ha="left")
    label(ax, .48, 1.25, f"Lời nhắc có bộ nhớ · bước {MEMORY_STEP}", 15, ha="left", weight="bold")
    box(ax, .46, .14, 9.08, .90, "", GREEN)
    label(ax, .69, .77, instruction, 14, ha="left")
    label(ax, .69, .41, "Memory: " + suffix, 14, TEAL, ha="left")
    save(fig, "loi-nhac-bo-nho.pdf")


def retention_chart(summary):
    fig, ax = plt.subplots(figsize=(10, 2.95))
    task_ids = [row["task_id"] for row in summary["per_task"]]
    for key, offset, color, hatch, title in (
        ("b0_successes", -.18, "#4b6681", "", "B0"),
        ("f1_off_successes", .18, TEAL, "///", "F1-off"),
    ):
        ax.bar([task + offset for task in task_ids], [row[key] for row in summary["per_task"]],
               width=.34, color=color, edgecolor=INK, linewidth=.55, hatch=hatch, label=title)
    ax.set(xticks=task_ids, yticks=range(6), ylim=(0, 5.5),
           xlabel="ID tác vụ LIBERO-10", ylabel="Lượt thành công / 5")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#dbe1e5", lw=.6)
    ax.legend(frameon=False, ncol=2, loc="lower right", bbox_to_anchor=(1, 1), fontsize=13)
    ax.tick_params(labelsize=12)
    ax.xaxis.label.set_size(14)
    ax.yaxis.label.set_size(14)
    fig.tight_layout(pad=.6)
    save(fig, "duy-tri-e6.pdf")


def reference(path):
    return {"path": path.relative_to(REPO).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def filmstrip(video, indices, crop, rotation, output):
    selection = "+".join(f"eq(n,{index})" for index in indices)
    filters = [f"select='{selection}'", "format=rgb24"]
    if rotation:
        filters.extend(("hflip", "vflip"))
    filters.extend(("crop=" + ":".join(map(str, crop)),
                    f"tile={len(indices)}x1:padding=8:color=white"))
    subprocess.run(
        ["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", str(video), "-vf",
         ",".join(filters), "-frames:v", "1", "-update", "1", str(output)], check=True,
    )
    return {"video": reference(video), "frame_indices": indices,
            "crop_whxy_pixels": crop, "display_rotation_degrees": rotation,
            "figure": reference(output)}


def main():
    # Reuse the existing diagram primitives and artwork with the same serif font.
    with tempfile.TemporaryDirectory() as preview_dir:
        names = {"02-su-kien-va-quan-he.pdf": "su-kien-che-khuat.pdf",
                 "03-truy-van-som.pdf": "truy-van-som.pdf"}

        def save_outline(fig, name):
            save(fig, names[name])

        for function in ("evidence", "timeline"):
            draw = STYLE[function]
            draw.__globals__["save"] = save_outline
            draw.__globals__["OUT"] = Path(preview_dir)
            draw()
    architecture()
    smolvla()
    availability()
    receipt = json.loads((EVIDENCE / "d2_t1_event_success_video_20261121.json").read_bytes())
    episode = receipt["episode"]
    t1 = REPO / episode["run"] / "episodes" / episode["episode_id"]
    queries = t1 / "queries.jsonl"
    memory_prompt(queries)
    grasp_camera = t1 / "agentview_image.mp4"
    assert reference(grasp_camera)["sha256"] == episode["artifact_sha256"][grasp_camera.name]
    grasp = filmstrip(grasp_camera, [142, 240, 409], [148, 124, 73, 70], 180,
                     FIGURES / "t1-gap-bat.png")
    e6 = REPO / "experiments/results/e6_retention_locked_50"
    book = "libero_10_task5/init_17"
    comparison = [e6 / f"{arm}_task_5" / book / "episode.json" for arm in ("B0", "F1_actual")]
    b0, f1 = (json.loads(path.read_bytes()) for path in comparison)
    assert b0["seed"] == f1["seed"] == 3559 and b0["initial_state_id"] == f1["initial_state_id"] == 17
    assert b0["success"] and not f1["success"] and f1["num_steps"] == 1000
    assert f1["diagnostics"]["evidence"]["ever_lifted"] and not f1["diagnostics"]["evidence"]["ever_placed"]
    # This video came from the display renderer, which already applies a vertical flip.
    degradation = filmstrip(comparison[1].with_name("video.mp4"), [0, 77, 103, 1000],
                           [240, 200, 8, 35], 0, FIGURES / "e6-dat-sach.png")
    summary = EVIDENCE / "e6_retention_paired_bootstrap_20261120.json"
    retention_chart(json.loads(summary.read_bytes()))
    manifest = {
        "smolvla_lora": {
            "paper": "https://arxiv.org/abs/2506.01844v1",
            "checkpoint_config": reference(REPO / "resources/checkpoints/smolvla_libero/config.json"),
            "training_config": reference(
                REPO / "experiments/training/results_f1_author_split_v2/output_f1/run_config.json"
            ),
        },
        "t1_memory_prompt": {"queries": reference(queries), "raw_query_step": RAW_STEP,
                             "memory_query_step": MEMORY_STEP,
                             "figure": reference(FIGURES / "loi-nhac-bo-nho.pdf")},
        "t1_grasp": grasp,
        "e6_degradation": {**degradation, "episode_records": list(map(reference, comparison)),
                           "comparability": "NON-COMPARABLE_OFFICIAL_PAPER"},
        "e6_retention_chart": {"summary": reference(summary),
                               "figure": reference(FIGURES / "duy-tri-e6.pdf")},
    }
    (EVIDENCE / "thesis_figure_sources.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
