# infer_canonical_awards_from_learned.py
# Turn learned_awards.json (from your pipeline) into a clean, deduped list of
# canonical award names + a variant->canonical map — with ZERO hard-coded categories.

from pathlib import Path
import json
import regex as re
import difflib
from collections import Counter, defaultdict

# ---------- Inputs / Outputs (edit paths if you want) ----------
IN_PATH  = Path("learned_awards.json")
OUT_LIST = Path("award_names_inferred.json")   # canonical names sorted by support
OUT_MAP  = Path("award_variant_map.json")     # variant -> canonical (incl. normalized keys)
OUT_MD   = Path("award_names_review.md")      # optional pretty review

# ---------- Tuning knobs ----------
SIMILARITY_FOR_MERGE = 97   # 0..100, very strict to avoid merging Drama vs Musical/Comedy
MIN_SUPPORT = 3             # drop tiny clusters that are likely noise

# ---------- Light normalization (no hard-coded categories) ----------
DASHES = r"[\u2012\u2013\u2014\u2212-]"
WS = re.compile(r"\s+")
PUNCT_EDGES = re.compile(r"[ \t\r\n\-–—·•~]+")

def normalize_for_compare(s: str) -> str:
    """Lightweight canonicalization for comparing phrases (token-set based)."""
    s = s or ""
    s = re.sub(DASHES, "-", s.lower())
    s = WS.sub(" ", s).strip()
    return s

def unify_dash_readable(s: str) -> str:
    """Make dashes visually consistent: ' – ' with single spaces."""
    s = re.sub(DASHES, "-", s)
    s = s.replace(" - ", " – ")
    s = s.replace(" -", " – ")
    s = s.replace("- ", " – ")
    s = s.replace("-", " – ")
    s = WS.sub(" ", s).strip()
    return s

def token_set(s: str) -> str:
    """Token-set string for difflib: order-insensitive, dedups tokens."""
    toks = normalize_for_compare(s).split()
    return " ".join(sorted(set(toks)))

def sim_score(a: str, b: str) -> int:
    """Similarity in 0..100 using difflib over token sets (order-insensitive)."""
    ta, tb = token_set(a), token_set(b)
    return int(round(100 * difflib.SequenceMatcher(None, ta, tb).ratio()))

# ---------- Canonical selection within a cluster ----------
def pick_canonical(variants_with_counts):
    """
    Choose canonical string from observed variants in the cluster.
    Heuristic: highest count first, break ties by longer string.
    """
    # variants_with_counts: List[Tuple[str,int]]
    variants_with_counts = [(v.strip(), int(c)) for v,c in variants_with_counts if v and v.strip()]
    if not variants_with_counts:
        return None
    # sort by (-count, -len)
    variants_with_counts.sort(key=lambda vc: (vc[1], len(vc[0])), reverse=True)
    return variants_with_counts[0][0]

# ---------- Main ----------
def main():
    if not IN_PATH.exists():
        raise SystemExit(f"Missing {IN_PATH}. Generate it via dump_learned_awards() first.")

    data = json.loads(IN_PATH.read_text(encoding="utf-8"))
    # expected rows: {"normalized": "...", "total_count": N, "top_variants": [{"text": "...", "count": M}, ...]}

    # 1) Filter out trivial clusters, assemble candidate canonicals
    clusters = []
    for row in data:
        total = int(row.get("total_count", 0))
        if total < MIN_SUPPORT:
            continue
        norm_key = row.get("normalized", "").strip()
        tops = row.get("top_variants", []) or []
        variants_with_counts = [(d.get("text","").strip(), int(d.get("count", 0))) for d in tops if d.get("text")]
        # fallback: if we somehow don't have variants, use normalized key as a 1-count variant
        if not variants_with_counts and norm_key:
            variants_with_counts = [(norm_key, total)]
        canonical = pick_canonical(variants_with_counts)
        if not canonical:
            continue
        canonical = unify_dash_readable(canonical)
        clusters.append({
            "norm_key": norm_key,
            "total": total,
            "canonical": canonical,
            "variants": variants_with_counts,  # keep for mapping
        })

    # 2) Merge near-duplicates (punctuation/dash/minor tokenization) w/ strict threshold
    clusters.sort(key=lambda r: r["total"], reverse=True)
    merged = []
    for c in clusters:
        placed = False
        for m in merged:
            if sim_score(c["canonical"], m["canonical"]) >= SIMILARITY_FOR_MERGE:
                # merge into m
                m["total"] += c["total"]
                # union variants
                for v, cnt in c["variants"]:
                    m["variant_counts"][v] += cnt
                # keep the better canonical (by total; if tie, longer)
                cand = max(
                    [(m["canonical"], m["total"]), (c["canonical"], c["total"])],
                    key=lambda x: (x[1], len(x[0]))
                )[0]
                m["canonical"] = cand
                m["norm_keys"].add(c["norm_key"])
                placed = True
                break
        if not placed:
            merged.append({
                "canonical": c["canonical"],
                "total": c["total"],
                "variant_counts": Counter(dict(c["variants"])),
                "norm_keys": {c["norm_key"]} if c["norm_key"] else set(),
            })

    # 3) Build outputs
    merged.sort(key=lambda r: r["total"], reverse=True)

    canonical_list = [unify_dash_readable(m["canonical"]) for m in merged]
    variant_map = {}

    for m in merged:
        canon = unify_dash_readable(m["canonical"])
        # map all observed variants to canonical
        for v, _cnt in m["variant_counts"].most_common():
            variant_map[v] = canon
            variant_map[normalize_for_compare(v)] = canon  # normalized key for robust lookup
        # also map cluster normalized keys
        for nk in m["norm_keys"]:
            if nk:
                variant_map[nk] = canon
                variant_map[normalize_for_compare(nk)] = canon

    # 4) Write files
    OUT_LIST.write_text(json.dumps(canonical_list, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_MAP.write_text(json.dumps(variant_map, ensure_ascii=False, indent=2), encoding="utf-8")

    # Optional: review markdown
    lines = ["# Inferred Golden Globes Award Names (from tweets, no hard-coding)\n"]
    for i, m in enumerate(merged, 1):
        lines.append(f"## {i:02d}. {m['canonical']}  _(support: {m['total']})_")
        ex = [v for v,_ in m["variant_counts"].most_common(5)]
        if ex:
            lines.append("- examples: " + "; ".join(ex))
        lines.append("")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote {OUT_LIST} ({len(canonical_list)} names) and {OUT_MAP} (variant map).")
    print(f"Review file: {OUT_MD}")

if __name__ == "__main__":
    main()
