"""Convert this manuscript's deliberately small Markdown subset to LaTeX.

This is not a general Markdown renderer. Unknown heading/image constructs fail
loudly so that generated source cannot silently omit manuscript content.
"""
from pathlib import Path
import re
import shutil

BASE = Path(__file__).resolve().parents[1]
TEX = BASE / "latex"
(TEX / "content").mkdir(exist_ok=True)
(TEX / "assets").mkdir(exist_ok=True)
source_logo = BASE.parent / "format_reference/format latex/image/UET.png"
shutil.copyfile(source_logo, TEX / "assets/UET.png")


def escape(text):
    mapping = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
               "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(mapping.get(c, c) for c in text)


def inline(text):
    saved = []
    def hold(value):
        saved.append(value)
        return f"ZZPLACEHOLDER{len(saved)-1}ZZ"
    text = re.sub(r"\[@([^\]]+)\]", lambda m: hold(r"\cite{" + ",".join(x.strip().lstrip("@") for x in m[1].split(";")) + "}"), text)
    text = re.sub(r"\$([^$\n]+)\$", lambda m: hold("$" + m[1] + "$"), text)
    text = re.sub(r"`([^`]+)`", lambda m: hold(r"\path{" + m[1] + "}"), text)
    text = escape(text)
    text = re.sub(r"\*\*([^*]+)\*\*", lambda m: r"\textbf{" + m[1] + "}", text)
    for i, value in enumerate(saved):
        text = text.replace(f"ZZPLACEHOLDER{i}ZZ", value)
    return text


CAPTIONS = [
    "So sánh cơ chế memory và phạm vi liên hệ với khóa luận",
    "Môi trường và protocol được ghi trong các artifact",
    "Số demonstration theo domain và split trong dataset manifest",
    "Kết quả T1/T3 local: một policy seed, mười nhóm initial state",
]
table_index = 0


def table(lines):
    global table_index
    cells = [[v.strip() for v in line.strip().strip("|").split("|")] for line in lines]
    header, rows = cells[0], cells[2:]
    n = len(header)
    if any(len(row) != n for row in rows):
        raise ValueError("Non-rectangular table")
    fractions = {2: [.31, .69], 4: [.18, .23, .26, .33],
                 5: [.29, .1775, .1775, .1775, .1775],
                 6: [.12, .14, .20, .12, .19, .23]}[n]
    cols = "".join(r">{\raggedright\arraybackslash}p{\dimexpr " + str(f) + r"\textwidth-2\tabcolsep\relax}" for f in fractions)
    caption = CAPTIONS[table_index]
    table_index += 1
    head = " & ".join(r"\textbf{" + inline(v) + "}" for v in header) + r" \\"
    result = [r"\begingroup\small\setlength{\tabcolsep}{4pt}\renewcommand{\arraystretch}{1.15}",
              r"\begin{longtable}{" + cols + "}", r"\caption{" + caption + r"}\label{tab:generated" + str(table_index) + r"}\\",
              r"\toprule", head, r"\midrule\endfirsthead", r"\toprule", head,
              r"\midrule\endhead", r"\bottomrule\endfoot"]
    result += [" & ".join(inline(v) for v in row) + r" \\" for row in rows]
    result += [r"\end{longtable}", r"\endgroup"]
    return "\n".join(result)


def convert(text, abstract=False):
    lines = text.splitlines()
    result, i = [], 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("# "):
            title = re.sub(r"^Chương \d+\.\s*", "", line[2:])
            result.append((r"\chapter*{" if abstract else r"\chapter{") + inline(title) + "}")
            if abstract:
                result.append(r"\thispagestyle{empty}")
        elif line.startswith("## "):
            title = re.sub(r"^\d+\.\d+\.\s*", "", line[3:])
            result.append(r"\section{" + inline(title) + "}")
        elif line.startswith("#"):
            raise ValueError("Unsupported heading " + line)
        elif line == "$$":
            end = lines.index("$$", i + 1)
            result += [r"\begin{equation}", *lines[i+1:end], r"\end{equation}"]
            i = end
        elif line.startswith("```"):
            end = next(j for j in range(i + 1, len(lines)) if lines[j] == "```")
            result += [r"\begin{Verbatim}[breaklines=true,fontsize=\small,frame=single]", *lines[i+1:end], r"\end{Verbatim}"]
            i = end
        elif line.startswith("|"):
            end = i
            while end < len(lines) and lines[end].startswith("|"):
                end += 1
            result.append(table(lines[i:end]))
            i = end - 1
        elif line.startswith("!["):
            match = re.fullmatch(r"!\[(.+)\]\(([^)]+)\)", line)
            if not match or not (BASE / match[2]).is_file():
                raise ValueError("Missing/invalid figure: " + line)
            result += [r"\begin{figure}[htbp]", r"\centering", r"\includegraphics[width=\textwidth]{../" + match[2] + "}",
                       r"\caption{" + inline(match[1]) + "}", r"\label{fig:" + Path(match[2]).stem + "}", r"\end{figure}"]
        elif line.strip():
            result.append(inline(line))
        else:
            result.append("")
        i += 1
    return "\n".join(result) + "\n"


text = (BASE / "THESIS_DRAFT.md").read_text(encoding="utf-8")
parts = re.split(r"(?m)(?=^# Chương \d+\.)", text)
if len(parts) != 6:
    raise ValueError("Expected abstract and exactly five chapters")
for i, section in enumerate(parts):
    target = TEX / "content" / ("abstract.tex" if i == 0 else f"chapter{i}.tex")
    target.write_text("% Generated from THESIS_DRAFT.md.\n" + convert(section, abstract=i == 0), encoding="utf-8")
if table_index != len(CAPTIONS):
    raise ValueError("Update table captions after changing manuscript tables")
print(f"Generated abstract, 5 chapters, {table_index} tables, and copied template logo.")
