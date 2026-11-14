import json

nb = json.load(open("notebooks/working_notebooks/memsft-extend.ipynb", encoding="utf-8"))
print("Total cells:", len(nb["cells"]))
for i, c in enumerate(nb["cells"]):
    src = "".join(c["source"]).replace("\n", " ")[:80]
    print(f"Cell {i:2d} | {c['cell_type']:8s} | {src}")
