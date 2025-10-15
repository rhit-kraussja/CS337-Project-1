# src/candidate_pipeline.py
# Steps 4–5: Candidate Generation (weak rules) + First-Pass Filtering

from __future__ import annotations
import argparse, json, os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import regex as re
from ftfy import fix_text
from unidecode import unidecode

# ---------- Anchors & simple resources ----------

ANCHORS = {
    # Entity wins Award  -> left entity, right award
    "WIN_A": re.compile(r"(.+?)\s+(wins?|receives?|gets|takes\s+home|is\s+awarded)\s+(.+)", re.I),
    # Award goes to Entity -> left award, right entity
    "WIN_B": re.compile(r"(.+?)\s+(goes\s+to|awarded\s+to)\s+(.+)", re.I),
    # Presenters
    "PRESENT_FWD": re.compile(r"(?:present(?:s|ed)?|introduce(?:s|d)?)\s+(.+)", re.I),
    "PRESENT_BY": re.compile(r"presented\s+by\s+(.+)", re.I),
    # Nominees
    "NOMINEES": re.compile(r"(?:nominee[s]?\s+(?:are|for)|nominated\s+for)\s+(.+)", re.I),
    # Host trigger
    "HOST": re.compile(r"(?:host(?:s|ed|ing)?\s+(?:tonight|the\s+show|the\s+golden\s+globes)|our\s+host(?:s)?\s+is)\b", re.I),
    # Broad "Best ..." net from anywhere
    "BEST_NET": re.compile(r"\bbest\b.{0,120}", re.I | re.DOTALL),
}

AWARD_KEYWORDS = {
    "best","motion","picture","film","actor","actress","director","screenplay",
    "song","score","series","tv","television","drama","musical","comedy",
    "animated","foreign","supporting","cecil","demille"
}
EDGE_STOPWORDS = {"the","a","an","of","for","and","or","to","in"}
PUNCT_OR_BREAK = re.compile(r"[.!?,:;]| {2,}")

# ---------- Data model ----------

@dataclass
class Candidate:
    tweet_id: str
    entity_type: str                 # "award"|"winner"|"presenter"|"nominee"|"host"|"entity"
    rule_id: str
    span_text: str                   # raw substring (may have #/@)
    span_text_clean: str             # normalized, readable
    anchor_text: str
    side: Optional[str] = None       # "L"|"R"|None
    had_url: bool = False
    had_hashtag: bool = False
    had_at: bool = False
    is_rt: bool = False
    is_qt: bool = False
    rt_count: int = 0
    qt_count: int = 0
    original_author: Optional[str] = None
    timestamp_ms: Optional[int] = None
    segment: str = "raw"             # "raw"|"qt_added"|"qt_body"
    weak_superlative: bool = False
    weak_casing: bool = False
    from_hashtag: bool = False

# ---------- Small helpers ----------

def clean_text(s: str) -> str:
    s = fix_text(s or "")
    s = unidecode(s)
    return re.sub(r"\s+", " ", s).strip()

def has_url(s: str) -> bool:
    return bool(re.search(r"https?://\S+", s))

def has_hashtag(s: str) -> bool:
    return bool(re.search(r"#\w+", s))

def has_at(s: str) -> bool:
    return bool(re.search(r"@\w+", s))

def humanize_hashtag(tok: str) -> str:
    t = tok.lstrip("#")
    parts = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+", t) or [t]
    return " ".join(p.capitalize() for p in parts)

def strip_hashes_ats(s: str) -> str:
    s = re.sub(r"#\w+", "", s)
    s = re.sub(r"@\w+", "", s)
    return " ".join(s.split())

def titleize_award(s: str) -> str:
    if not s: return s
    return " ".join("Best" if w.lower()=="best" else w.capitalize() for w in s.split())

def enumerate_prefixes(tokens: List[str], max_len: int) -> List[str]:
    out, run = [], []
    for tok in tokens:
        if PUNCT_OR_BREAK.search(tok): break
        run.append(tok)
        if len(run) <= max_len: out.append(" ".join(run))
        else: break
    return out

def enumerate_suffixes(tokens: List[str], max_len: int) -> List[str]:
    out, run = [], []
    for i in range(1, min(max_len, len(tokens)) + 1):
        tok = tokens[-i]
        if PUNCT_OR_BREAK.search(tok): break
        run.insert(0, tok)
        out.append(" ".join(run))
    return out

def explode_nominee_list(s: str) -> List[str]:
    return [p.strip() for p in re.split(r",| and ", s) if p.strip()]

def split3(text: str, pat: re.Pattern) -> Optional[Tuple[str,str,str]]:
    m = pat.search(text)
    if not m: return None
    return m.group(1).strip(), m.group(2).strip(), m.group(3).strip()

def mk_candidate(base: Dict, *, entity_type: str, rule_id: str, span_text: str,
                 anchor_text: str, side: Optional[str], segment: str,):
    mirror = strip_hashes_ats(span_text)
    clean = clean_text(mirror)
    if "best" in clean.lower():
        clean = titleize_award(clean)
    return Candidate(
        tweet_id=str(base.get("id") or base.get("id_str") or ""),
        entity_type=entity_type,
        rule_id=rule_id,
        span_text=span_text,
        span_text_clean=clean,
        anchor_text=anchor_text,
        side=side,
        had_url=has_url(span_text),
        had_hashtag=has_hashtag(span_text),
        had_at=has_at(span_text),
        is_rt=bool(base.get("is_rt", False)),
        is_qt=bool(base.get("is_qt", False)),
        rt_count=int(base.get("rt_count") or 0),
        qt_count=int(base.get("qt_count") or 0),
        original_author=base.get("original_author"),
        timestamp_ms=base.get("timestamp_ms"),
        segment=segment,
        from_hashtag=bool(re.fullmatch(r"#\w+", span_text.strip()))
    )

# ---------- Step 4: Candidate generation ----------

def generate_from_text(text: str, base: Dict, segment: str,
                       max_left: int, max_right: int) -> List[Candidate]:
    text = clean_text(text)
    cands: List[Candidate] = []

    # Winners A: Entity wins Award
    x = split3(text, ANCHORS["WIN_A"])
    if x:
        L, anchor, R = x
        for suf in enumerate_suffixes(L.split(), max_left):
            cands.append(mk_candidate(base, entity_type="entity", rule_id="WIN_A",
                                      span_text=suf, anchor_text=anchor, side="L", segment=segment))
        for pre in enumerate_prefixes(R.split(), max_right):
            cands.append(mk_candidate(base, entity_type="award", rule_id="WIN_A",
                                      span_text=pre, anchor_text=anchor, side="R", segment=segment))

    # Winners B: Award goes to Entity
    x = split3(text, ANCHORS["WIN_B"])
    if x:
        L, anchor, R = x
        for suf in enumerate_suffixes(L.split(), max_left):
            cands.append(mk_candidate(base, entity_type="award", rule_id="WIN_B",
                                      span_text=suf, anchor_text=anchor, side="L", segment=segment))
        for pre in enumerate_prefixes(R.split(), max_right):
            cands.append(mk_candidate(base, entity_type="entity", rule_id="WIN_B",
                                      span_text=pre, anchor_text=anchor, side="R", segment=segment))

    # Presenters
    m = ANCHORS["PRESENT_FWD"].search(text)
    if m:
        obj = m.group(1).strip()
        cands.append(mk_candidate(base, entity_type="presenter", rule_id="PRESENT_FWD",
                                  span_text=obj, anchor_text="present", side=None, segment=segment))
    m = ANCHORS["PRESENT_BY"].search(text)
    if m:
        name = m.group(1).strip()
        cands.append(mk_candidate(base, entity_type="presenter", rule_id="PRESENT_BY",
                                  span_text=name, anchor_text="presented by", side=None, segment=segment))

    # Nominees
    m = ANCHORS["NOMINEES"].search(text)
    if m:
        for name in explode_nominee_list(m.group(1)):
            cands.append(mk_candidate(base, entity_type="nominee", rule_id="NOMINEES",
                                      span_text=name, anchor_text="nominee", side=None, segment=segment))

    # Host
    if ANCHORS["HOST"].search(text):
        cands.append(mk_candidate(base, entity_type="host", rule_id="HOST",
                                  span_text=text, anchor_text="host", side=None, segment=segment))

    # Best-net (award-like phrase anywhere)
    m = ANCHORS["BEST_NET"].search(text)
    if m:
        cands.append(mk_candidate(base, entity_type="award", rule_id="BEST_NET",
                                  span_text=m.group(0).strip(), anchor_text="best", side=None, segment=segment))
    return cands

def generate_candidates(tweet: Dict, max_left: int = 8, max_right: int = 10) -> List[Candidate]:
    out: List[Candidate] = []
    raw = tweet.get("text_clean") or tweet.get("text") or ""
    if raw:
        out.extend(generate_from_text(raw, tweet, "raw", max_left, max_right))
    if tweet.get("is_qt"):
        if tweet.get("qt_added_text"):
            out.extend(generate_from_text(tweet["qt_added_text"], tweet, "qt_added", max_left, max_right))
        if tweet.get("qt_body"):
            out.extend(generate_from_text(tweet["qt_body"], tweet, "qt_body", max_left, max_right))
    return out

# ---------- Step 5: First-pass filtering ----------

def drop_hard(c: Candidate, min_award_tokens: int, max_award_tokens: int, max_entity_tokens: int) -> bool:
    t = c.span_text_clean.strip()
    if not t: return True

    # Raw span had URL/hashtags/handles → we already mirrored; drop the raw
    if c.had_url: return True
    if c.had_hashtag or c.had_at:
        # if mirroring produced empty, drop
        if not t: return True

    toks = t.split()
    n = len(toks)

    if c.entity_type == "award":
        if n < min_award_tokens or n > max_award_tokens: return True
        low = t.lower()
        if "best" not in low and not any(k in low for k in AWARD_KEYWORDS):
            return True
        if toks[0].lower() in EDGE_STOPWORDS or toks[-1].lower() in EDGE_STOPWORDS:
            return True
    else:
        if n < 1 or n > max_entity_tokens: return True

    return False

def soft_flags(c: Candidate) -> Candidate:
    txt = c.span_text_clean
    if txt.lower() in {"best","supporting"}:
        c.weak_superlative = True
    if txt.islower():
        c.weak_casing = True
    return c

def first_pass_filter(cands: List[Candidate],
                      min_award_tokens: int = 2,
                      max_award_tokens: int = 10,
                      max_entity_tokens: int = 6) -> List[Candidate]:
    seen = set()
    out: List[Candidate] = []
    for c in cands:
        if drop_hard(c, min_award_tokens, max_award_tokens, max_entity_tokens):
            continue
        c = soft_flags(c)
        key = (c.tweet_id, c.rule_id, c.entity_type, c.span_text_clean.lower())
        if key in seen: 
            continue
        seen.add(key)
        out.append(c)
    return out

# ---------- IO / CLI ----------

def iter_jsonl(path: str):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def write_jsonl(path: str, cands: List[Candidate]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for c in cands:
            f.write(json.dumps(asdict(c)) + "\n")

def run(fin: str, fout: str, max_left: int, max_right: int,
        min_award_tokens: int, max_award_tokens: int, max_entity_tokens: int):
    all_out: List[Candidate] = []
    for tw in iter_jsonl(fin):
        gen = generate_candidates(tw, max_left=max_left, max_right=max_right)
        fil = first_pass_filter(gen,
                                min_award_tokens=min_award_tokens,
                                max_award_tokens=max_award_tokens,
                                max_entity_tokens=max_entity_tokens)
        all_out.extend(fil)
    write_jsonl(fout, all_out)
    print(f"Wrote {len(all_out)} candidates → {fout}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="fin", required=True)
    ap.add_argument("--out", dest="fout", required=True)
    ap.add_argument("--max-left", type=int, default=8)
    ap.add_argument("--max-right", type=int, default=10)
    ap.add_argument("--min-award-tokens", type=int, default=2)
    ap.add_argument("--max-award-tokens", type=int, default=10)
    ap.add_argument("--max-entity-tokens", type=int, default=6)
    args = ap.parse_args()
    run(args.fin, args.fout, args.max_left, args.max_right,
        args.min_award_tokens, args.max_award_tokens, args.max_entity_tokens)

if __name__ == "__main__":
    main()
