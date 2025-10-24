# main.py
import json
from pathlib import Path
from dataclasses import asdict
from candidate_pipeline import generate_from_text, dump_learned_awards, finalize_answers

INPUT  = Path("spikes_out/combined_spikes.json")
OUT    = Path("candidates.json")
ANSW   = Path("proj1_answers.json")

def load_texts(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit("Input JSON must be a list")
    for item in data:
        yield (item.get("text","") if isinstance(item, dict) else str(item))

def main():
    texts = list(load_texts(INPUT))
    cands = []
    for t in texts:
        for c in generate_from_text(t, {}, "raw", 8, 2):
            cands.append(asdict(c))
    OUT.write_text(json.dumps(cands, ensure_ascii=False, indent=2), encoding="utf-8")
    dump_learned_awards("learned_awards.json")

    answers = finalize_answers(texts, cands)       # ← new
    ANSW.write_text(json.dumps(answers, ensure_ascii=False, indent=4), encoding="utf-8")
    print(f"Wrote {len(cands)} candidates to {OUT}")
    print(f"Wrote answers to {ANSW}")

if __name__ == "__main__":
    main()
