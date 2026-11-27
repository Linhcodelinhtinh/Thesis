"""Hình tổng quan Chương 1 (bộ nhớ sự kiện + cơ chế dừng), cùng phong cách Hình 3.1.

Dùng lại canvas/box/arrow/label và bảng màu của chapter3_outline/draw_figures.py,
giống build_thesis_figures.architecture(). Chạy trên Windows để có font Constantia:
    python docs/thesis/tools/build_overview_figure.py   (ghi figures/vong-dieu-khien.pdf, Hình 3.1)
Biến môi trường THESIS_FIG_FONT cho phép đổi font khi xem thử trên máy không có Constantia.
"""
import os
import runpy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Ellipse, FancyBboxPatch, Rectangle, Wedge

THESIS = Path(os.environ.get("THESIS_DIR", Path(__file__).resolve().parents[1]))
FIGURES = THESIS / "figures"
STYLE = runpy.run_path(str(THESIS / "chapter3_outline/draw_figures.py"))
canvas, box, arrow, label = (STYLE[key] for key in ("canvas", "box", "arrow", "label"))
INK, BLUE, GREEN, TEAL = (STYLE[key] for key in ("INK", "BLUE", "GREEN", "TEAL"))
RED, ROSE = "#a5323c", "#fce9e7"
EVIDENCE_LINE = "#8b9aa4"
FONT = os.environ.get("THESIS_FIG_FONT", "Constantia")
OUT_NAME = os.environ.get("THESIS_FIG_OUT", "vong-dieu-khien.pdf")


def save(fig, name):
    meta = {"Creator": "build_overview_figure.py", "CreationDate": None}
    targets = [FIGURES / name]
    if (THESIS / "latex/figures").is_dir():
        targets.append(THESIS / "latex/figures" / name)
    for path in targets:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, facecolor="white", metadata=meta)
    plt.close(fig)


ASSETS = Path(os.environ.get("OVERVIEW_ASSETS", THESIS / "figures" / "overview_assets"))
ARM_FILL, ARM_EDGE, JOINT = "#f4f6f7", "#8b9aa4", "#3b4854"


def dashed_panel(ax, x, y, w, h, title):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.16",
                                linewidth=1.2, edgecolor=INK, facecolor="none", ls=(0, (4, 3))))
    ax.add_patch(FancyBboxPatch((x + .12, y + h - .50), w - .24, .38,
                                boxstyle="round,pad=0.005,rounding_size=0.05",
                                linewidth=1.0, edgecolor=EVIDENCE_LINE, facecolor="#d9dfe3"))
    label(ax, x + w / 2, y + h - .31, title, 15, weight="bold")


def franka_schematic(ax, x, y, w, h):
    """Hình vẽ cánh tay Franka Panda (nét đơn giản): thân trắng, khớp xám đậm, bàn kẹp hai ngón."""
    import numpy as np
    sx, sy = w / 2.45, h / 1.65
    P = lambda px, py: np.array([x + px * sx, y + py * sy])
    base = P(.55, .10)
    pts = [base, P(.55, .52), P(.55, .70), P(1.15, 1.30), P(1.80, 1.00), P(1.95, .62)]
    ax.add_patch(FancyBboxPatch(P(.12, 0.0), .86 * sx, .10 * sy,
                                boxstyle="round,pad=0.0,rounding_size=0.03",
                                linewidth=1.0, edgecolor=ARM_EDGE, facecolor="#cfd6db"))

    def capsule(a, b, width, color):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=width, solid_capstyle="round")

    def cap(point, direction, length=.075, width=13):
        d = np.array([direction[0] / sx, direction[1] / sy])
        d = d / np.hypot(*d)
        u = np.array([d[0] * sx, d[1] * sy]) * length
        capsule(point - u, point + u, width, JOINT)

    for a, b in zip(pts[:-1], pts[1:]):
        capsule(a, b, 19, ARM_EDGE)
    for a, b in zip(pts[:-1], pts[1:]):
        capsule(a, b, 16, ARM_FILL)
    dirs = [(0, 1), (0, 1), (.6, .6), (.65, -.3), (.15, -.4)]
    for pt, d in zip(pts[1:], dirs[1:]):
        cap(pt, d)
    # bàn kẹp: mặt bích, thân tay và hai ngón
    fx, fy = pts[-1]
    ax.add_patch(Rectangle((fx - .12 * sx, fy - .22 * sy), .24 * sx, .17 * sy, facecolor=ARM_FILL,
                           edgecolor=ARM_EDGE, lw=1.0))
    for dx in (-.09, .09):
        ax.plot([fx + dx * sx] * 2, [fy - .22 * sy, fy - .42 * sy], color=INK, lw=2.4,
                solid_capstyle="round")
    for text, point, off in [(r"$q_1$", pts[1], (-.28, 0)), (r"$q_2$", pts[2], (-.28, .02)),
                             (r"$q_4$", pts[3], (-.02, .24)), (r"$q_6$", pts[4], (.08, .24))]:
        label(ax, point[0] + off[0] * sx, point[1] + off[1] * sy, text, 11, TEAL)
    label(ax, fx + .30 * sx, fy - .32 * sy, r"$g$", 11, RED)


def node(ax, x, y, w, h, text):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.12",
                                linewidth=1.0, edgecolor=TEAL, facecolor="white"))
    label(ax, x + w / 2, y + h / 2, text, 12.5, TEAL)


def overview():
    with plt.rc_context({"font.family": FONT, "pdf.fonttype": 42,
                         "mathtext.fontset": "custom", "mathtext.rm": FONT,
                         "mathtext.it": f"{FONT}:style=italic", "mathtext.fallback": None}):
        fig, ax = canvas(6.75)

        # 1. Khung đầu vào: ảnh RGB thật từ rollout LIBERO, trạng thái robot, chỉ dẫn
        dashed_panel(ax, .15, 4.02, 9.30, 2.60, r"Đầu vào $o_t$")
        img = plt.imread(ASSETS / "libero_frame.png")
        ax.imshow(img, extent=(2.30, 3.80, 4.52, 6.02), zorder=2, aspect="auto")
        ax.add_patch(Rectangle((2.30, 4.52), 1.50, 1.50, fill=False, edgecolor=EVIDENCE_LINE,
                               lw=1.0, zorder=3))
        label(ax, 3.05, 4.30, "Ảnh quan sát (RGB)", 13.5, weight="bold")
        robot = plt.imread(ASSETS / "franka_panda.png")
        ax.imshow(robot, extent=(6.675 - 0.632, 6.675 + 0.632, 4.52, 6.02), zorder=2, aspect="auto")
        label(ax, 6.675, 4.30, "Trạng thái robot", 13.5, weight="bold")

        # 2. Bộ ghi sự kiện có bằng chứng (máy trạng thái hữu hạn)
        arrow(ax, (5.0, 4.02), (5.0, 3.57))
        label(ax, 5.12, 3.80, "dữ kiện kèm bằng chứng", 12.5, ha="left")
        box(ax, 2.95, 2.05, 4.10, 1.50, "", GREEN)
        label(ax, 5.0, 3.26, "Bộ ghi sự kiện có bằng chứng", 17.5)
        node(ax, 3.22, 2.33, 1.12, .48, "Tay rỗng")
        node(ax, 5.66, 2.33, 1.12, .48, "Đang giữ vật")
        arrow(ax, (4.36, 2.70), (5.64, 2.70), TEAL)
        label(ax, 5.0, 2.86, "gắp", 12, TEAL)
        arrow(ax, (5.64, 2.44), (4.36, 2.44), TEAL)
        label(ax, 5.0, 2.25, r"đặt: $k \leftarrow k+1$", 12, TEAL)

        # 3a. Nhánh trái: tóm tắt ngôn ngữ cho VLA
        box(ax, .15, 2.20, 2.22, 1.20, "", GREEN)
        label(ax, 1.26, 3.03, "Tóm tắt ngôn ngữ", 16.5)
        label(ax, 1.26, 2.58, "“đã xong 2/3 chu kỳ”", 13, style="italic")
        arrow(ax, (2.95, 2.80), (2.37, 2.80))
        label(ax, 2.66, 3.00, "chiếu", 12)

        # 3b. Nhánh phải: bộ điều phối
        box(ax, 7.63, 2.20, 1.82, 1.20, "", ROSE)
        label(ax, 8.54, 3.17, "Bộ điều phối", 16.5)
        label(ax, 8.54, 2.80, r"dừng khi $k \geq K$", 12)
        label(ax, 8.54, 2.59, "ngắt đoạn lệnh", 12)
        label(ax, 8.54, 2.38, "truy vấn lại", 12)
        arrow(ax, (7.05, 2.80), (7.63, 2.80))
        label(ax, 7.34, 3.00, r"$k$", 12)

        # 4. Chính sách VLA và robot
        box(ax, .15, .30, 4.25, 1.07, "", BLUE)
        label(ax, 2.27, 1.00, "Chính sách VLA (SmolVLA / X-VLA)", 17)
        label(ax, 2.27, .58, r"Lời nhắc = chỉ dẫn $\ell$ + tóm tắt $C_q$", 13.5)
        arrow(ax, (1.26, 2.20), (1.26, 1.37))
        label(ax, 1.38, 1.78, r"$C_q$", 13, ha="left")

        box(ax, 5.95, .30, 3.50, 1.07, "", BLUE)
        label(ax, 7.70, 1.00, "Mô phỏng robot", 17)
        label(ax, 7.70, .58, "Thực thi đoạn hành động", 13.5)
        arrow(ax, (4.40, .84), (5.95, .84))
        label(ax, 5.17, 1.04, "Đoạn lệnh", 12.5)
        label(ax, 5.17, .63, r"$A_q$", 12.5)

        for xs, ys, tip in [([7.95, 7.95, 3.40], [2.20, 1.62, 1.62], (3.40, 1.37)),
                            ([8.90, 8.90], [2.20, 1.62], (8.90, 1.37))]:
            ax.plot(xs, ys, color=RED, lw=1.6, ls="--")
            ax.annotate("", xy=tip, xytext=(xs[-1], ys[-1]),
                        arrowprops={"arrowstyle": "-|>", "color": RED, "lw": 1.6,
                                    "shrinkA": 0, "shrinkB": 1, "mutation_scale": 13})
        label(ax, 5.60, 1.82, "Yêu cầu truy vấn lại", 13, RED)
        label(ax, 8.80, 1.82, "Ngắt, dừng", 13, RED, ha="right")

        # Vòng phản hồi quan sát
        ax.plot([9.45, 9.74, 9.74], [.84, .84, 5.32], color=TEAL, lw=1.4)
        arrow(ax, (9.74, 5.32), (9.45, 5.32), TEAL)
        label(ax, 9.86, 3.00, r"$o_{t+1}$", 13, TEAL, rotation=90)
        save(fig, OUT_NAME)


if __name__ == "__main__":
    overview()
