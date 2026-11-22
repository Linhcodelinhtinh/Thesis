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
    plt.close(fig)


def architecture():
    fig, ax = canvas(2.48)
    modules = [
        (.22, "Bộ nhớ sự kiện", "Ghi và chiếu dữ kiện", GREEN),
        (2.78, "Chính sách VLA", "Sinh đoạn hành động", BLUE),
        (5.34, "Bộ điều phối", "Chọn lịch truy vấn", ROSE),
        (7.90, "Mô phỏng robot", "Thực thi hành động", BLUE),
    ]
    for x, title, detail, fill in modules:
        box(ax, x, 1.26, 1.88, .77, "", fill)
        label(ax, x + .94, 1.77, title, 14, weight="bold")
        label(ax, x + .94, 1.52, detail, 12.5)
    for index, text in enumerate(("Lời nhắc", "Đoạn lệnh", "Hành động")):
        x = modules[index][0] + 1.88
        next_x = modules[index + 1][0]
        arrow(ax, (x, 1.65), (next_x, 1.65))
        label(ax, (x + next_x) / 2, 2.22, text, 12.5)
    ax.plot([8.84, 8.84, 1.16], [1.26, .60, .60], color=TEAL, lw=1.3)
    for x in (1.16, 3.72):
        arrow(ax, (x, .60), (x, 1.26), TEAL)
    label(ax, 6.10, .35, "Quan sát sau hành động", 13, TEAL)
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
        label(ax, center + .12, 2.92, text + "\nĐóng băng", 14, ICE_INK)
    box(ax, 3.83, 2.61, 2.34, .62, "Token hóa", "#ffffff", 14)
    for center in (1.70, 5.0, 8.30):
        arrow(ax, (center, 2.61), (center, 2.06))
    box(ax, .65, 1.42, 8.70, .62, "", ICE)
    snowflake(ax, 1.05, 1.73)
    label(ax, 5, 1.73, "SmolLM2   ·   ghép và xử lý token   ·   đóng băng", 16, ICE_INK)
    arrow(ax, (5, 1.42), (5, 1.11))
    box(ax, .65, .38, 8.70, .71, "", "#ffd9d6")
    label(ax, 3.58, .73, "Action Expert   ·   flow matching   ·   50 hành động", 15, RED)
    ax.plot([6.63, 6.63], [.51, .97], color="#bd676b", lw=.8)
    flame(ax, 7.02, .73)
    label(ax, 8.15, .73, r"LoRA trên $W_q, W_v$", 14, RED)
    label(ax, 5, .15, "Ngọn lửa chỉ các tham số LoRA được cập nhật", 12, RED)
    save(fig, "smolvla-lora.pdf")


def availability():
    fig, ax = canvas(2.04)
    ax.plot([.55, 9.32], [.87, .87], color=INK, lw=1.5)
    arrow(ax, (9.32, .87), (9.65, .87))
    for x, symbol, title in [(1.45, r"$\tau$", "Sự kiện xảy ra"),
                              (4.10, r"$\nu$", "Đủ bằng chứng"),
                              (6.90, "$q$", "Truy vấn VLA")]:
        ax.scatter([x], [.87], s=46, zorder=3, color=TEAL)
        label(ax, x, .52, symbol, 15)
        label(ax, x, 1.22, title, 14)
    label(ax, 2.72, 1.69, "Thu thập bằng chứng", 12, GRAY)
    label(ax, 5.50, 1.69, r"Đủ điều kiện thời gian: $\nu \leq q$", 12, TEAL)
    label(ax, 8.33, 1.69, "Chưa khả dụng tại $q$", 12, RED,
          bbox={"boxstyle": "round,pad=.25", "facecolor": ROSE, "edgecolor": "none"})
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
