"""Convierte el articulo Markdown a .docx (estructura basica + figuras).

Uso: py -3.12 paper/md_to_docx.py <md> <docx> <repo>
No reproduce el template ACM exacto; produce un Word editable con titulos,
parrafos, negritas, listas, tablas y las figuras incrustadas, como punto de
partida para volcar al template acmart.
"""
import os
import re
import sys
from docx import Document
from docx.shared import Inches

FIG_DIRS = ["results", "paper/assets"]


def find_fig(name, repo):
    for d in FIG_DIRS:
        p = os.path.join(repo, d, name)
        if os.path.exists(p):
            return p
    return None


def add_runs(paragraph, text):
    text = text.replace("`", "")
    for part in re.split(r"(\*\*[^*]+\*\*)", text):
        if part.startswith("**") and part.endswith("**"):
            paragraph.add_run(part[2:-2]).bold = True
        else:
            paragraph.add_run(part)


def md_to_docx(md_path, docx_path, repo):
    with open(md_path, encoding="utf-8") as f:
        lines = f.read().split("\n")

    doc = Document()
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        if not line or line == "---":
            i += 1
            continue

        if line.startswith("|"):  # tabla
            buf = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                buf.append(lines[i].strip())
                i += 1
            rows = [[c.strip() for c in r.strip("|").split("|")] for r in buf]
            rows = [r for r in rows
                    if not all(set(c) <= set("-: ") for c in r)]
            if rows:
                t = doc.add_table(rows=len(rows), cols=len(rows[0]))
                t.style = "Light Grid Accent 1"
                for ri, row in enumerate(rows):
                    for ci, cell in enumerate(row):
                        add_runs(t.cell(ri, ci).paragraphs[0], cell)
            continue

        if line.startswith("### "):
            doc.add_heading(line[4:], level=2)
        elif line.startswith("## "):
            doc.add_heading(line[3:], level=1)
        elif line.startswith("# "):
            doc.add_heading(line[2:], level=0)
        elif line.startswith("- "):
            add_runs(doc.add_paragraph(style="List Bullet"), line[2:])
        elif re.match(r"^\d+\.\s", line):
            add_runs(doc.add_paragraph(style="List Number"),
                     re.sub(r"^\d+\.\s", "", line))
        elif line.startswith("*") and line.endswith("*") and not line.startswith("**"):
            doc.add_paragraph().add_run(line.strip("*")).italic = True
        else:
            add_runs(doc.add_paragraph(), line)
            for fig in re.findall(r"([\w]+\.png)", line):
                fp = find_fig(fig, repo)
                if fp:
                    doc.add_picture(fp, width=Inches(6))
        i += 1

    doc.save(docx_path)
    print(f"[OK] {docx_path}")


if __name__ == "__main__":
    md_to_docx(sys.argv[1], sys.argv[2], sys.argv[3])
