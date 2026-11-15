from pathlib import Path
from PIL import Image, ImageOps, ImageDraw
import re
root = Path(__file__).parent
files = sorted(root.glob('page-*.png'), key=lambda p: int(re.search(r'-(\d+)\.png$', p.name).group(1)))
cols, rows = 4, 4
thumb_w, thumb_h = 300, 430
for start in range(0, len(files), cols * rows):
    batch = files[start:start + cols * rows]
    sheet = Image.new('RGB', (cols * thumb_w, rows * (thumb_h + 24)), 'white')
    draw = ImageDraw.Draw(sheet)
    for j, path in enumerate(batch):
        im = Image.open(path).convert('RGB')
        im.thumbnail((thumb_w - 12, thumb_h - 12))
        x, y = (j % cols) * thumb_w, (j // cols) * (thumb_h + 24)
        sheet.paste(im, (x + (thumb_w-im.width)//2, y + 3))
        draw.text((x + 8, y + thumb_h + 2), f'Page {start+j+1}', fill='black')
    out = root / f'contact-{start//(cols*rows)+1:02d}.png'
    sheet.save(out)
    print(out.name, len(batch), sheet.size)
