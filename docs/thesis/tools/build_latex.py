"""Convert this manuscript's deliberately small Markdown subset to LaTeX.

This is not a general Markdown renderer. Unknown heading/image constructs fail
loudly so that generated source cannot silently omit manuscript content.
"""
import argparse
from pathlib import Path
import re

BASE = Path(__file__).resolve().parents[1]


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
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", lambda m: r"\textit{" + m[1] + "}", text)
    for i, value in enumerate(saved):
        text = text.replace(f"ZZPLACEHOLDER{i}ZZ", value)
    return text


CAPTIONS = [
    "Các cách biểu diễn lịch sử và liên hệ với câu hỏi nghiên cứu",
    "Ngưỡng đo quan hệ giữa bát và đĩa gốc",
    "Các điều kiện đối chứng và mục đích đánh giá",
    "Đối chiếu sự kiện thị giác với mốc tiến độ trên quỹ đạo trình diễn",
    "Các điều kiện đối chiếu trong thực nghiệm vòng kín",
    "Phân hoạch quỹ đạo theo nhóm tác vụ",
    "Cấu hình thích ứng của F1 và phạm vi suy luận",
    "Kết quả điều khiển vòng kín theo tác vụ và điều kiện",
]
table_index = 0


def table(lines):
    global table_index
    cells = [[v.strip() for v in line.strip().strip("|").split("|")] for line in lines]
    header, rows = cells[0], cells[2:]
    n = len(header)
    if any(len(row) != n for row in rows):
        raise ValueError("Non-rectangular table")
    fractions = {2: [.31, .69], 3: [.25, .35, .40], 4: [.18, .23, .26, .33],
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


def main():
    parser = argparse.ArgumentParser(description="Export the historical Markdown to a scratch directory")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    target = args.output_dir.resolve()
    canonical = (BASE / "latex").resolve()
    if target == canonical or canonical in target.parents:
        parser.error("Export target must be outside the canonical LaTeX tree")
    text = (BASE / "THESIS_DRAFT.md").read_text(encoding="utf-8")
    parts = re.split(r"(?m)(?=^# Chương \d+\.)", text)
    if len(parts) != 6:
        raise ValueError("Expected historical abstract and five chapters")
    chapters = [convert(part) for part in parts[1:]]
    if table_index != len(CAPTIONS):
        raise ValueError("Update table captions after changing manuscript tables")
    target.mkdir(parents=True, exist_ok=False)
    for i, chapter in enumerate(chapters, 1):
        (target / f"chapter{i}.tex").write_text(chapter, encoding="utf-8")


if __name__ == "__main__":
    main()
