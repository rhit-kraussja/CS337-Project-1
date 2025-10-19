# main.py
import json
from pathlib import Path
from dataclasses import asdict
from candidate_pipeline import generate_from_text, dump_learned_awards

INPUT  = Path("gg2013.json")
OUT    = Path("candidates.json")

def load_texts(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit("Input JSON must be a list")
    for item in data:
        yield (item.get("text","") if isinstance(item, dict) else str(item))

def main():
    texts = list(load_texts(INPUT))
    out = []
    for t in texts:
        for c in generate_from_text(t, {}, "raw", 8, 2):
            out.append(asdict(c))
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    dump_learned_awards("learned_awards.json")  # optional report
    print(f"Wrote {len(out)} candidates to {OUT}")

if __name__ == "__main__":
    main()
