from pypdf import PdfReader
from pathlib import Path
pdf = Path('output/pdf/khoa-luan-luu-quang-linh-ban-thao.pdf')
reader = PdfReader(str(pdf))
terms = ['author-action replay', 'three sequence milestones', 'terminal success', 'T8 source', 'provenance', 'custom author-action']
for i, page in enumerate(reader.pages, 1):
    text = page.extract_text() or ''
    found = [term for term in terms if term.lower() in text.lower()]
    if found:
        print(i, found)
        for line in text.splitlines():
            if any(term.split()[0].lower() in line.lower() for term in found):
                print(' ', line[:220])
