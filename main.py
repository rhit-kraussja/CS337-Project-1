from dataclasses import asdict
import json
from pathlib import Path
from tqdm import tqdm
import spacy
from candidate_pipeline import (
    extract_red_carpet_batch,
    extract_performance_info_batch,
    aggregate_red_carpets,
    aggregate_performances, 
    generate_from_text, 
    dump_learned_awards
)

INPUT  = Path("gg2013.json")
OUT    = Path("candidates.json")

def load_texts(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit("Input JSON must be a list")
    for item in data:
        yield (item.get("text","") if isinstance(item, dict) else str(item))

def load_nlp_model():
    """Try to load your trained NER model, fallback to spaCy default."""
    model_path = Path("output/model-best")
    if model_path.exists():
        print("✅ Using trained spaCy model (output/model-best)")
        return spacy.load(model_path)
    else:
        print("⚠️ Using default model (en_core_web_sm)")
        return spacy.load("en_core_web_sm")

def main():
    global nlp
    nlp = load_nlp_model()

    texts = list(load_texts(INPUT))
    out = []

    print(f"📄 Processing {len(texts)} texts...")

    # --- Run all batch extractors ---
    extract_red_carpet_batch(texts)
    extract_performance_info_batch(texts)

    # --- Candidate extraction loop (if needed) ---
    for text in tqdm(texts, desc="Generating candidates"):
        for c in generate_from_text(text, {}, "raw", 8, 2):
            out.append(asdict(c))

    # --- Aggregate and save everything ---
    aggregate_red_carpets()
    aggregate_performances()
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    dump_learned_awards("learned_awards.json")

    print(f"🎬 Done! Wrote {len(out)} candidates to {OUT}")

if __name__ == "__main__":
    main()
