"""Read the 101 formulas out of your own copy of the paper (arXiv:1601.00991) into data/formulas_raw.json.
Usage: extract_formulas.py <path to 1601.00991.pdf>      (needs pypdf)"""
import json, os, re, sys
import pypdf

text = "\n".join(p.extract_text() for p in pypdf.PdfReader(sys.argv[1]).pages)
parts = re.split(r"Alpha#(\d+)\s*:", text[text.find("Alpha#1:"):])
alphas = {}
for k in range(1, len(parts) - 1, 2):
    n, e = int(parts[k]), parts[k + 1]
    if n == 101: e = e[:e.find(".001))") + 6]
    e = re.sub(r"\n\s*\d{1,2}\s*\n", "\n", e)              # page numbers that fall between lines
    alphas[n] = " ".join(e.split())
assert sorted(alphas) == list(range(1, 102)) and all(e.count("(") == e.count(")") for e in alphas.values())
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "formulas_raw.json")
json.dump(alphas, open(out, "w"), indent=0)
print("wrote", len(alphas), "formulas to", out)
