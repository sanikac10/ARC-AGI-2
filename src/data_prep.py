# =============================
# File: src/data_prep.py
# Purpose: Convert handcrafted *.py solvers into JSONL training rows
# =============================

"""Usage
python src/data_prep.py --in_dir problems/ --out data/train.jsonl
"""

import ast, json, argparse, glob, textwrap, os

TOK = {"VISEXP": "<VISEXP>", "REASON": "<REASON>", "CODE": "<CODE>"}

# -----------------------------
# extract_solver – fetch DSL function body from a .py file
# -----------------------------

def extract_solver(path: str) -> str:
    tree = ast.parse(open(path).read())
    func = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name.startswith("solve_")]
    if not func:
        return ""
    return textwrap.dedent(ast.get_source_segment(open(path).read(), func[0]))

# -----------------------------
# main pipeline
# -----------------------------

def main(in_dir: str, out_file: str):
    with open(out_file, "w") as w:
        for py in glob.glob(os.path.join(in_dir, "*.py")):
            solver_code = extract_solver(py)
            meta = {
                "id": os.path.basename(py).split(".")[0],
                "input_grid": "TODO",  # replace with real grids
                "output_grid": "TODO",
                "context": f"{TOK['VISEXP']} … {TOK['REASON']} … {TOK['CODE']}\n{solver_code}",
            }
            w.write(json.dumps(meta) + "\n")
    print(f"Wrote {out_file}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--in_dir"); ap.add_argument("--out"); args = ap.parse_args()
    main(args.in_dir, args.out)
