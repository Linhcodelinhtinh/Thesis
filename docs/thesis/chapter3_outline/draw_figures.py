"""Generate the three vector diagrams for the Chapter 3 writing outline."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

OUT = Path(__file__).resolve().parent
INK = "#172f40"
TEAL = "#12685f"
BLUE = "#eaf1fa"
GREEN = "#e7f3ef"
SAND = "#fff2df"
GRAY = "#596976"
RED = "#a5323c"
SLATE = "#37474f"

plt.rcParams.update({"font.family": "Constantia", "pdf.fonttype": 42})


def canvas(height):
    fig = plt.figure(figsize=(10, height))
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, 10), ylim=(0, height), aspect="equal")
    ax.axis("off")
    return fig, ax


def label(ax, x, y, value, size=12, color=INK, **kwargs):
    ax.text(x, y, value, ha=kwargs.pop("ha", "center"), va="center",
            fontsize=size, color=color, **kwargs)


def box(ax, x, y, w, h, value, fill, size=13):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015,rounding_size=0.16",
                                linewidth=1.1, edgecolor=INK, facecolor=fill))
    label(ax, x+w/2, y+h/2, value, size)


def arrow(ax, start, end, color=INK):
    ax.annotate("", xy=end, xytext=start,
                arrowprops={"arrowstyle": "-|>", "lw": 1.5, "color": color,
                            "shrinkA": 1, "shrinkB": 1, "mutation_scale": 13})


def save(fig, name):
    fig.savefig(OUT / name, facecolor="white", metadata={"Creator": "Chapter 3 outline"})
    plt.close(fig)


def architecture():
    red = "#a5323c"
    with plt.rc_context({"font.family": "Constantia"}):
        fig, ax = canvas(3.4)
        box(ax,.25,1.48,2.4,.85,"Bộ nhớ\ncó bằng chứng",GREEN,16)
        box(ax,3.88,1.48,2.2,.85,"Chính sách VLA",BLUE,16)
        box(ax,7.26,1.48,2.48,.85,"Điều phối\nvà thực thi","#fce9e7",16)
        arrow(ax,(2.65,1.90),(3.88,1.90))
        label(ax,3.26,2.12,"ngữ cảnh",11)
        arrow(ax,(6.08,1.90),(7.26,1.90))
        label(ax,6.67,2.12,"đoạn lệnh",11)
        label(ax,1.46,2.96,"Quan sát sau hành động",12)
        arrow(ax,(1.46,2.72),(1.46,2.34))
        label(ax,4.98,2.96,"Quan sát và chỉ dẫn",12)
        arrow(ax,(4.98,2.72),(4.98,2.34))
        ax.plot([1.45,1.45,8.8],[1.48,.30,.30],color=INK,lw=1.2)
        arrow(ax,(8.8,.30),(8.8,1.48))
        label(ax,3.18,.51,"dữ kiện vừa cập nhật",12)
        ax.plot([8.12,8.12,5.00],[1.48,.91,.91],color=red,lw=1.7,ls="--")
        arrow(ax,(5.00,.91),(5.00,1.48),red)
        label(ax,6.55,.66,"yêu cầu truy vấn lại",12,red)
        save(fig,"01-kien-truc.pdf")


def cycle():
    fig, ax = canvas(2.05)
    for x, value in [(0.18,"Mốc tham chiếu"), (3.90,"Chu kỳ đang mở"), (7.60,"Đóng chu kỳ")]:
        box(ax,x,.82,2.15,.66,value,GREEN,12)
    arrow(ax,(2.33,1.15),(3.9,1.15))
    arrow(ax,(6.05,1.15),(7.60,1.15))
    label(ax,3.11,1.66,"xác nhận rời",11)
    label(ax,6.83,1.66,"xác nhận về",11)
    arrow(ax,(4.98,.82),(4.98,.40),"#a5323c")
    label(ax,4.98,.19,"Hủy chu kỳ nếu mất dấu kéo dài hoặc có bằng chứng trái chiều",11)
    save(fig,"02b-xac-nhan-chu-ky.pdf")


def evidence():
    red_bg = "#fde8e8"      # Mau do nhat (thay the mau vang SAND #fff2df)
    red_border = "#f5b7b1"  # Vien do mem
    red_tag = "#922b21"     # Nhan do dam
    blue_border = "#b8d0eb"
    green_border = "#b6ded2"
    
    with plt.rc_context({"font.family": "Constantia"}):
        fig = plt.figure(figsize=(10.0, 4.10))
        ax = fig.add_axes([0, 0, 1, 1], xlim=(0, 10.0), ylim=(0, 4.10), aspect="equal")
        ax.axis("off")
        
        w = 2.92
        margin = 0.24
        gap = (10.0 - 2 * margin - 3 * w) / 2.0
        xs = [margin, margin + w + gap, margin + 2 * (w + gap)]
        
        headings = ["Quan sát được lần cuối", "Sau khi che khuất", "Khi có bằng chứng mới"]
        rows = [
            ("Quan sát", "Vật A đi vào hộp X", "Hai hộp ra khỏi tầm nhìn", "A được thấy trong hộp Y"),
            ("Lịch sử", "Đã thấy A vào X", "Giữ sự kiện đã ghi", "Giữ sự kiện đã ghi"),
            ("Hiện tại", "A ở trong X", "Vị trí chưa xác định", "A ở trong Y"),
        ]
        
        row_colors = [BLUE, GREEN, red_bg]
        border_colors = [blue_border, green_border, red_border]
        tag_colors = ["#1d5c96", "#0e6251", red_tag]
        
        for i, (x, h) in enumerate(zip(xs, headings)):
            ax.text(x + w / 2, 3.65, h, ha="center", va="center",
                    fontsize=17.5, color=INK, weight="bold")
            for j, (k, *values) in enumerate(rows):
                y = 2.45 - 1.02 * j
                rect = FancyBboxPatch((x, y - 0.36), w, 0.90,
                                      boxstyle="round,pad=0.02,rounding_size=0.14",
                                      linewidth=1.2, edgecolor=border_colors[j], facecolor=row_colors[j])
                ax.add_patch(rect)
                # Nhan loai thong tin
                ax.text(x + 0.16, y + 0.24, k.upper(), ha="left", va="center",
                        fontsize=14.0, color=tag_colors[j], weight="bold")
                # Noi dung su kien / quan he
                ax.text(x + 0.16, y - 0.10, values[i], ha="left", va="center",
                        fontsize=16.5, color=INK)
        
        y_mid = 1.43
        for i in range(2):
            arrow_start_x = xs[i] + w + 0.06
            arrow_end_x = xs[i+1] - 0.06
            ax.annotate("", xy=(arrow_end_x, y_mid), xytext=(arrow_start_x, y_mid),
                        arrowprops={"arrowstyle": "-|>", "lw": 1.8, "color": INK,
                                    "shrinkA": 0, "shrinkB": 0, "mutation_scale": 15})
        fig.savefig(OUT / "02-su-kien-va-quan-he.png", dpi=200, facecolor="white")
        save(fig, "02-su-kien-va-quan-he.pdf")



def timeline():
    with plt.rc_context({"font.family": "Constantia"}):
        fig, ax = canvas(4.50)
        x = lambda step: 2.60 + step * 0.57
        for step in [0, 2, 3, 10, 12]:
            ax.plot([x(step), x(step)], [.65, 3.20], color="#b7c2c8", lw=1.1, ls=":")
            label(ax, x(step), .35, str(step), 17)
        label(ax, 2.30, .35, "Bước", 18, ha="right", weight="bold")
        label(ax, 2.30, 2.45, "Theo bằng chứng", 18, ha="right", weight="bold")
        label(ax, 2.30, 1.20, "Định kỳ", 18, ha="right", weight="bold")

        ax.plot([x(0), x(3)], [2.45, 2.45], c=INK, lw=5.0, solid_capstyle="butt")
        ax.plot([x(3), x(12)], [2.45, 2.45], c=TEAL, lw=5.0, solid_capstyle="butt")
        ax.plot([x(0), x(10)], [1.20, 1.20], c=INK, lw=5.0, solid_capstyle="butt")
        ax.plot([x(10), x(12)], [1.20, 1.20], c=TEAL, lw=5.0, solid_capstyle="butt")

        for s, y in [(0, 2.45), (3, 2.45), (0, 1.20), (10, 1.20)]:
            ax.scatter([x(s)], [y], s=140, color="white", edgecolor=INK, linewidth=2.2, zorder=5)

        # Mốc bước 2: Phát hiện biến cố mới (Slate)
        label(ax, x(2) - .15, 3.95, "Dữ kiện đủ căn cứ", 16.5, SLATE, ha="right", weight="bold")
        arrow(ax, (x(2), 3.72), (x(2), 2.60), SLATE)

        # Mốc bước 3: Quyết định truy vấn sớm / ngắt đoạn (RED)
        label(ax, x(3) + .15, 3.40, "Truy vấn sớm sau 3 hành động", 16.5, RED, ha="left", weight="bold")
        arrow(ax, (x(3), 3.17), (x(3), 2.60), RED)

        # Nhãn đoạn thực thi hành động mới (TEAL)
        label(ax, x(7.5), 2.82, "Thực thi đoạn mới", 16.5, TEAL, weight="bold")

        # Mốc bước 10: Truy vấn định kỳ theo lịch (INK) kèm mũi tên chỉ mốc
        label(ax, x(10), 1.82, "Truy vấn định kỳ", 16.5, INK, weight="bold", ha="center")
        arrow(ax, (x(10), 1.62), (x(10), 1.35), INK)

        save(fig, "03-truy-van-som.pdf")


if __name__ == "__main__":
    architecture()
    cycle()
    evidence()
    timeline()
