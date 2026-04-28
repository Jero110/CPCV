import json
from pathlib import Path


def main() -> None:
    nb_path = Path("full_experiments.ipynb")
    with nb_path.open() as f:
        nb = json.load(f)

    glb = {"__name__": "__main__"}
    for idx, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if not source.strip():
            continue
        print(f"[runner] Executing code cell {idx}")
        exec(compile(source, f"{nb_path.name}:cell_{idx}", "exec"), glb, glb)


if __name__ == "__main__":
    main()
