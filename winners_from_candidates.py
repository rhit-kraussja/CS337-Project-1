# winners_from_candidates.py
# Parse candidates.json and elect a single winner per award from tweet-derived candidates.
# No CLI. Edit config at the top if needed.

from pathlib import Path
import json
from collections import defaultdict, Counter
import regex as re

# ----------------- CONFIG -----------------
CANDIDATES_PATH = Path("candidates.json")
# Optional: variant->canonical map produced by your inference step (if present, it's used)
AWARD_MAP_PATH  = Path("award_variant_map.json")
WINNERS_JSON    = Path("winners.json")
WINNERS_MD      = Path("winners.md")

# If top candidate's share is below this, flag as "low confidence"
LOW_CONF_SHARE = 0.55   # 55%
# ------------------------------------------

# --- light normalization utilities (no heavy deps) ---
DASHES = r"[\u2012\u2013\u2014\u2212-]"
WS = re.compile(r"\s+")
QUOTES = re.compile(r"[\"'“”‘’`]")
PARENS = re.compile(r"[(){}\[\]]")

SMALL = {"and","or","of","the","a","an","in","for","by","to","on","at","with","series","film"}

def normalize_award_key(s: str) -> str:
    """Lower, unify dashes, collapse spaces; used for dict keys/lookup."""
    s = s or ""
    s = re.sub(DASHES, "-", s.lower())
    s = s.replace("&", "and").replace(" tv ", " television ")
    s = WS.sub(" ", s).strip()
    return s

def normalize_person(s: str) -> str:
    """Clean simple person variants (cases/quotes/extra spaces)."""
    s = s or ""
    s = QUOTES.sub("", s)
    s = PARENS.sub("", s)
    s = WS.sub(" ", s).strip()
    # Title-case words, but keep small words lowercase unless first
    parts = s.split()
    out = []
    for i, w in enumerate(parts):
        lw = w.lower()
        if i > 0 and lw in SMALL:
            out.append(lw)
        else:
            out.append(w[:1].upper() + w[1:])
    return " ".join(out)

def unify_dash_readable(s: str) -> str:
    """Make dashes consistent and readable with en-dash spacing."""
    s = re.sub(DASHES, "-", s)
    s = WS.sub(" ", s)
    s = s.replace(" - ", " – ")
    s = s.replace(" -", " – ")
    s = s.replace("- ", " – ")
    s = s.replace("-", " – ")
    return s.strip()

def smart_title_award(s: str) -> str:
    """Human-friendly title casing for awards; preserve key words like Television, Motion Picture."""
    s = s or ""
    s = unify_dash_readable(s)
    words = s.split()
    out = []
    for i, w in enumerate(words):
        lw = w.lower()
        if i > 0 and lw in SMALL:
            out.append(lw)
        else:
            out.append(w[:1].upper() + w[1:])
    return " ".join(out)

# --- optional award variant map (if present) ---
def load_award_map(path: Path):
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    # Build a lookup that tries both raw and normalized keys
    m = {}
    for k, v in raw.items():
        m[k] = v
        m[normalize_award_key(k)] = v
    return m

def canonicalize_award(name: str, variant_map: dict) -> str:
    """Return a canonical award name using variant map when possible; else readable normalized."""
    if not name or name.strip().lower() == "unrecognizable award":
        return ""
    # direct or normalized map hit
    if name in variant_map:
        return variant_map[name]
    key = normalize_award_key(name)
    if key in variant_map:
        return variant_map[key]
    # fallback: readable formatting of the original
    pretty = smart_title_award(name)
    return pretty

# --- main ---
def main():
    if not CANDIDATES_PATH.exists():
        raise SystemExit(f"Missing {CANDIDATES_PATH}. Run your extractor first.")

    candidates = json.loads(CANDIDATES_PATH.read_text(encoding="utf-8"))
    if not isinstance(candidates, list):
        raise SystemExit("candidates.json must be a JSON array")

    variant_map = load_award_map(AWARD_MAP_PATH)

    # Count votes per (award, person)
    votes = defaultdict(Counter)   # award -> Counter(person -> count)
    total_per_award = Counter()

    kept = 0
    for c in candidates:
        # expected keys: rule_id, award_name, anchor_text, subject
        award_raw = (c.get("award_name") or "").strip()
        person_raw = (c.get("subject") or "").strip()

        if not award_raw or award_raw.lower() == "unrecognizable award":
            continue
        if not person_raw:
            continue

        award = canonicalize_award(award_raw, variant_map)
        if not award:
            continue
        person = normalize_person(person_raw)

        votes[award][person] += 1
        total_per_award[award] += 1
        kept += 1

    if kept == 0:
        raise SystemExit("No usable candidates found (all unrecognizable or empty).")

    # Elect winners per award
    winners = []
    for award, ctr in votes.items():
        total = total_per_award[award]
        top_person, top_count = ctr.most_common(1)[0]
        share = top_count / total if total else 0.0
        winners.append({
            "award": unify_dash_readable(award),
            "winner": top_person,
            "votes_for_winner": int(top_count),
            "total_votes_for_award": int(total),
            "winner_share": round(share, 4),
            "low_confidence": bool(share < LOW_CONF_SHARE),
            "runner_up": [
                {"name": p, "votes": int(n)}
                for p, n in ctr.most_common(5)[1:]  # top 5 including winner; skip winner
            ]
        })

    # Sort by award name for stable output
    winners.sort(key=lambda r: r["award"])

    # Write outputs
    WINNERS_JSON.write_text(json.dumps(winners, ensure_ascii=False, indent=2), encoding="utf-8")

    # Pretty Markdown table
    lines = [
        "# Predicted Golden Globes Winners (from candidates.json)",
        "",
        "| Award | Winner | Votes | Share | Note |",
        "|---|---|---:|---:|---|",
    ]
    for r in winners:
        note = "⚠️ low confidence" if r["low_confidence"] else ""
        lines.append(
            f"| {r['award']} | {r['winner']} | {r['votes_for_winner']}/{r['total_votes_for_award']} | "
            f"{int(round(100*r['winner_share']))}% | {note} |"
        )
    WINNERS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {WINNERS_JSON} and {WINNERS_MD}")
    weak = [w for w in winners if w["low_confidence"]]
    if weak:
        print(f"{len(weak)} categories flagged low confidence (winner share < {int(LOW_CONF_SHARE*100)}%).")

if __name__ == "__main__":
    main()
