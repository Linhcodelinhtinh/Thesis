import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os

ep_sh = "experiments/evaluation/decision_shadow_dev12_t1_t3_20261120_wsl/episodes/T1_F1_seen_state7"
cap_a = cv2.VideoCapture(f"{ep_sh}/agentview_image.mp4")
cap_w = cv2.VideoCapture(f"{ep_sh}/robot0_eye_in_hand_image.mp4")

steps = [0, 110, 142, 240]
titles = [
    "Bước 0: Khởi tạo",
    "Bước 110: Tiếp cận",
    "Bước 142: Trượt kẹp",
    "Bước 240: Nâng rỗng"
]

agent_imgs = []
wrist_imgs = []

for s in steps:
    cap_a.set(cv2.CAP_PROP_POS_FRAMES, s)
    cap_w.set(cv2.CAP_PROP_POS_FRAMES, s)
    _, fa = cap_a.read()
    _, fw = cap_w.read()
    
    # Flip vertically to correct OpenGL upside-down rendering for human eyes
    fa_flipped = cv2.flip(fa, 0)
    fw_flipped = cv2.flip(fw, 0)
    
    agent_imgs.append(Image.fromarray(cv2.cvtColor(fa_flipped, cv2.COLOR_BGR2RGB)))
    wrist_imgs.append(Image.fromarray(cv2.cvtColor(fw_flipped, cv2.COLOR_BGR2RGB)))

cap_a.release()
cap_w.release()

cell_w = 256
cell_h = 256
header_h = 36
banner_w = 110
gap = 6

total_w = banner_w + 4 * cell_w + 3 * gap
total_h = header_h + 2 * cell_h + gap

canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
draw = ImageDraw.Draw(canvas)

try:
    font_title = ImageFont.truetype("arialbd.ttf", 15)
    font_row = ImageFont.truetype("arialbd.ttf", 14)
    font_sub = ImageFont.truetype("arial.ttf", 13)
except:
    font_title = ImageFont.load_default()
    font_row = font_title
    font_sub = font_title

# Draw column titles
for i, title in enumerate(titles):
    x = banner_w + i * (cell_w + gap)
    bbox = draw.textbbox((0, 0), title, font=font_title)
    tw = bbox[2] - bbox[0]
    tx = x + (cell_w - tw) // 2
    draw.text((tx, 10), title, fill=(20, 20, 20), font=font_title)

# Row 0: Agentview
y0 = header_h
draw.rectangle([(0, y0), (banner_w - gap, y0 + cell_h)], fill=(245, 246, 248))
text_r0_1 = "Góc ngoài"
text_r0_2 = "Agentview"
bbox1 = draw.textbbox((0, 0), text_r0_1, font=font_row)
bbox2 = draw.textbbox((0, 0), text_r0_2, font=font_sub)
draw.text(((banner_w - gap - (bbox1[2]-bbox1[0]))//2, y0 + cell_h//2 - 16), text_r0_1, fill=(30, 30, 30), font=font_row)
draw.text(((banner_w - gap - (bbox2[2]-bbox2[0]))//2, y0 + cell_h//2 + 8), text_r0_2, fill=(100, 100, 100), font=font_sub)

for i in range(4):
    x = banner_w + i * (cell_w + gap)
    canvas.paste(agent_imgs[i], (x, y0))

# Row 1: Wristview
y1 = header_h + cell_h + gap
draw.rectangle([(0, y1), (banner_w - gap, y1 + cell_h)], fill=(245, 246, 248))
text_r1_1 = "Góc cổ tay"
text_r1_2 = "Wrist view"
bbox1 = draw.textbbox((0, 0), text_r1_1, font=font_row)
bbox2 = draw.textbbox((0, 0), text_r1_2, font=font_sub)
draw.text(((banner_w - gap - (bbox1[2]-bbox1[0]))//2, y1 + cell_h//2 - 16), text_r1_1, fill=(30, 30, 30), font=font_row)
draw.text(((banner_w - gap - (bbox2[2]-bbox2[0]))//2, y1 + cell_h//2 + 8), text_r1_2, fill=(100, 100, 100), font=font_sub)

for i in range(4):
    x = banner_w + i * (cell_w + gap)
    canvas.paste(wrist_imgs[i], (x, y1))

out_path_latex = "docs/thesis/latex/figures/smolvla-chunk-failure.png"
out_path_docs = "docs/thesis/figures/smolvla-chunk-failure.png"

canvas.save(out_path_latex, dpi=(300, 300))
canvas.save(out_path_docs, dpi=(300, 300))
print("Saved vertically flipped failure figure successfully.")
