# src/candidate_pipeline.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import json
import regex as re
from collections import Counter
from ftfy import fix_text
from unidecode import unidecode
import spacy

# --- add near the top ---
import math
from collections import defaultdict, Counter
import regex as re
from unidecode import unidecode

def _canon(s: str) -> str:
    s = unidecode(s).lower().strip()
    s = re.sub(r"[“”\"'’`]", "", s)
    s = re.sub(r"[-–—]+", "-", s)
    s = re.sub(r"[^a-z0-9 &:\-\(\)/]", " ", s)
    s = s.replace("television", "tv").replace("&", "and")
    s = re.sub(r"\b(the|a|an)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _lev_ok(a: str, b: str, k: int = 3) -> bool:
    # tiny, fast, regex-based approx edit distance (OK for short strings)
    # fallback: token overlap heuristic
    if a == b: return True
    ta, tb = set(a.split()), set(b.split())
    if min(len(ta), len(tb)) and len(ta & tb) / max(1, len(ta | tb)) >= 0.6:
        return True
    return False

_AWARD_SPAN = re.compile(r"\b(best\s+[^\n\.!?]{5,120}?)\b(?=[\.\!\?,\n]|$)", re.I)
_WIN_VERB = re.compile(r"\b(wins?|won|winner is|goes to|awarded to|takes(?: home)?)\b", re.I)
_PRES_PAT = re.compile(
    r"\bpresent(?:ed|s|ing)?\b.*?\b(best [^.!\n]{3,120})\b.*?\b([A-Z][\w'.-]+(?: [A-Z][\w'.-]+)+)(?:\s*(?:,|and|&)\s*([A-Z][\w'.-]+(?: [A-Z][\w'.-]+)+))?",
    re.I
)
_NOMS_HEAD = re.compile(r"\b(best [^.!\n]{3,120})\b.*?\bnominee[s]?\b[:\-]\s*(.+)$", re.I)
_HOSTS = re.compile(r"\b(hosted by|hosts are|your hosts)\b\s*(.+)$", re.I)

def _normalize_award_surface(s: str) -> str:
    s = s.strip(" -:").strip()
    s = re.sub(r"\s*-\s*", " - ", s)
    s = re.sub(r"\s+", " ", s)
    return s

def _discover_awards_from_texts(texts):
    freq = Counter()
    variants = defaultdict(Counter)
    for t in texts:
        for m in _AWARD_SPAN.finditer(t):
            raw = _normalize_award_surface(m.group(1))
            c = _canon(raw)
            # keep only plausible category words; not a hard-coded list of awards
            if re.search(r"\b(actor|actress|director|screenplay|picture|series|tv|song|score|animated|foreign|supporting|drama|comedy|musical)\b", c):
                freq[c] += 1
                variants[c][raw] += 1
    # cluster by canon overlap (light-weight)
    clusters = []
    for c, cnt in freq.most_common():
        placed = False
        for cl in clusters:
            if _lev_ok(c, cl["canon"]):
                cl["total"] += cnt
                for k, v in variants[c].items():
                    cl["vars"][k] += v
                placed = True
                break
        if not placed:
            clusters.append({"canon": c, "total": cnt, "vars": variants[c].copy()})
    # choose most frequent surface form per cluster
    awards = []
    for cl in clusters:
        if cl["total"] >= 20:  # support threshold
            surf = max(cl["vars"].items(), key=lambda kv: kv[1])[0]
            awards.append(surf)
    return awards  # list of strings as your canonical award keys

def _find_award_in_text(text, discovered_awards):
    t = _canon(text)
    best, score = None, 0
    for a in discovered_awards:
        ca = _canon(a)
        if ca in t and len(ca) > score:
            best, score = a, len(ca)
    return best

def _cluster_strings(strings):
    # returns representative -> set(all variants)
    reps = []
    for s in strings:
        cs = _canon(s)
        placed = False
        for r in reps:
            if _lev_ok(cs, _canon(r["rep"])):
                r["all"].add(s)
                r["rep"] = max(r["all"], key=lambda x: len(x))  # prefer longer surface
                placed = True
                break
        if not placed:
            reps.append({"rep": s, "all": set([s])})
    return {r["rep"]: r["all"] for r in reps}

def _time_peak_minutes(timestamps):
    if not timestamps: return set()
    # naive: keep full set; you can refine if you have per-tweet ts
    return set(range(len(timestamps)))  # placeholder when no ts

def finalize_answers(texts, candidates):
    """
    texts: List[str]  (raw tweet texts)
    candidates: List[dict]  (asdict from your generate_from_text)
    Returns dict ready to json.dump for gg2013answers.json
    """
    # 1) discover award keys from TEXTS (no hard-coded awards)
    discovered_awards = _discover_awards_from_texts(texts)

    # 2) winners from candidates + fallback from text
    per_award = defaultdict(lambda: {"winner": Counter(), "presenters": [], "nominees": [], "ts": []})

    # winner votes from your existing candidate extractor
    for c in candidates:
        award = c.get("award") or ""
        target = c.get("subject") or c.get("target") or ""
        anchor = c.get("anchor") or ""
        if not award or not target: 
            continue
        # map candidate award to nearest discovered award
        aw = _find_award_in_text(award, discovered_awards) or award
        per_award[aw]["winner"][target] += 1
        per_award[aw]["ts"].append(c.get("timestamp"))

    # 3) presenters/nominees/hosts passes (regex-first) from TEXTS
    hosts = []
    for t in texts:
        # presenters
        m = _PRES_PAT.search(t)
        if m:
            aw_text = _normalize_award_surface(m.group(1))
            aw_key = _find_award_in_text(aw_text, discovered_awards) or aw_text
            for g in (2,3):
                name = m.group(g)
                if name:
                    per_award[aw_key]["presenters"].append(name.strip())

        # nominees
        m2 = _NOMS_HEAD.search(t)
        if m2:
            aw_text = _normalize_award_surface(m2.group(1))
            aw_key = _find_award_in_text(aw_text, discovered_awards) or aw_text
            lst = m2.group(2)
            parts = re.split(r"\s*,\s*|\s+and\s+", lst)
            for p in parts:
                p = re.sub(r"^[\"“”'’`]+|[\"“”'’`]+$", "", p).strip()
                if len(p) > 1:
                    per_award[aw_key]["nominees"].append(p)

        # hosts
        mh = _HOSTS.search(t)
        if mh:
            lst = mh.group(2)
            parts = re.split(r"\s*,\s*|\s+and\s+|\s*&\s*", lst)
            for p in parts:
                p = p.strip()
                if p: hosts.append(p)

        # winners by verb without award in candidate stage
        if _WIN_VERB.search(t):
            aw_guess = _find_award_in_text(t, discovered_awards)
            if aw_guess:
                # try to capture the entity after "to|is|:" or before "wins/won"
                mpost = re.search(r"(?:to|is|:)\s*\"?([A-Z][\w' .&-]{2,})", t)
                mpre  = re.search(r"([A-Z][\w' .&-]{2,})\s+(?:wins?|won)", t)
                name = (mpost.group(1) if mpost else (mpre.group(1) if mpre else None))
                if name:
                    per_award[aw_guess]["winner"][name.strip()] += 1

    # 4) cluster/normalize per field + pick winners
    award_keys = []
    for aw, bucket in list(per_award.items()):
        # map award to discovered exact surface if close
        aw2 = _find_award_in_text(aw, discovered_awards) or aw
        if aw2 not in per_award:
            per_award[aw2] = bucket
            if aw2 != aw:
                del per_award[aw]

    for aw, bucket in per_award.items():
        award_keys.append(aw)
        # winner
        if bucket["winner"]:
            # cluster winner strings
            reps = _cluster_strings(list(bucket["winner"].elements()))
            # choose cluster with highest total count
            best_rep, best_count = None, -1
            for rep, variants in reps.items():
                total = sum(bucket["winner"][v] for v in variants)
                if total > best_count:
                    best_rep, best_count = rep, total
            bucket["winner"] = best_rep
        else:
            bucket["winner"] = ""

        # presenters / nominees de-dupe + cluster
        def _dedup(lst):
            reps = _cluster_strings(lst)
            # choose representative surface string each
            return sorted(list(reps.keys()))
        bucket["presenters"] = _dedup(bucket["presenters"])
        bucket["nominees"]   = _dedup(bucket["nominees"])

    # 5) hosts cluster/dedup
    hosts = sorted(list(_cluster_strings(hosts).keys()))

    # 6) build final JSON (lowercase strings, to match prof file)
    def low(s): return _canon(s).strip()
    out = {"hosts": [low(h) for h in hosts], "award_data": {}}
    for aw in sorted(set(award_keys), key=_canon):
        b = per_award[aw]
        out["award_data"][low(aw)] = {
            "nominees": [low(x) for x in b["nominees"]],
            "presenters": [low(x) for x in b["presenters"]],
            "winner": low(b["winner"]) if b["winner"] else ""
        }
    return out
# ---------- Setup ----------

# Load spaCy English model once (install with: python -m spacy download en_core_web_sm)
nlp = spacy.load("en_core_web_sm")

# ---------- Regex resources for normalization and award extraction ----------

DASHES = r"[\u2012\u2013\u2014\u2212-]"  # normalize all dash variants to '-'
SPACE = re.compile(r"\s+")
URL = re.compile(r"https?://\S+")
HANDLE = re.compile(r"@\w+")
HASHTAG = re.compile(r"#\w+")
PUNCT_STRIP = re.compile(r"[\"'“”‘’`(){}\[\]]")
BEST_SPAN = re.compile(r"\b(best\s+[a-z0-9&/,\-.\s]{3,120})", re.I)  # capture "Best …"
TRIM_AT = re.compile(r"[.!?;:|]")

# Track learned award phrases across a run (normalized → counts, + original variants)
AWARD_FREQ: Counter = Counter()
AWARD_VARIANTS: Dict[str, Counter] = {}  # normalized → Counter(original → count)

# ---------- Anchors (award/winner patterns) ----------

ANCHORS = {
    # Entity wins Award  -> left entity, right award
    "WIN_A": re.compile(r"(.+?)\s+(wins?|receives?|gets|takes\s+home|is\s+awarded)\s+(.+)", re.I),
    # Award goes to Entity -> left award, right entity
    "WIN_B": re.compile(r"(.+?)\s+(goes\s+to|awarded\s+to)\s+(.+)", re.I),
}

# ---------- Data model ----------

@dataclass
class Candidate:
    rule_id: str        # "WIN_A" or "WIN_B"
    award_name: str     # extracted award phrase
    anchor_text: str    # the matched anchor verb phrase
    subject: str        # PERSON name (winner)

# ---------- Helpers ----------

def normalize_text(s: str) -> str:
    """
    Normalize text to improve matching:
    - fix unicode and strip diacritics
    - remove URLs, @handles, #hashtags
    - normalize dashes and strip certain punctuation
    - lowercase; '&'→'and'; 'tv'→'television'; collapse spaces
    """
    s = s or ""
    s = fix_text(unidecode(s))
    s = URL.sub(" ", s)
    s = HANDLE.sub(" ", s)
    s = HASHTAG.sub(" ", s)
    s = re.sub(DASHES, "-", s)
    s = PUNCT_STRIP.sub(" ", s)
    s = s.lower().replace("&", "and").replace(" tv ", " television ")
    s = SPACE.sub(" ", s).strip()
    return s

def extract_award_from_side(side_text: str) -> Optional[str]:
    """
    Pull a likely award phrase from one side of the anchor.
    Strategy:
    1) Prefer an explicit 'Best …' span.
    2) Trim at strong punctuation.
    3) Require 'best' to avoid junk.
    4) Record normalized + original; return the most common original variant.
    Returns None if no reasonable award phrase is found.
    """
    if not side_text:
        return None

    m = BEST_SPAN.search(side_text)
    cand = m.group(1) if m else side_text

    cut = TRIM_AT.search(cand)
    if cut:
        cand = cand[:cut.start()]

    cand = cand.strip(" -—~·•\n\t ")
    if not cand or "best" not in cand.lower():
        return None

    norm = normalize_text(cand)
    if len(norm) < 8:
        return None

    AWARD_FREQ[norm] += 1
    AWARD_VARIANTS.setdefault(norm, Counter())[cand] += 1
    return AWARD_VARIANTS[norm].most_common(1)[0][0]

def dump_learned_awards(path: str = "learned_awards.json") -> None:
    """
    Persist a summary of learned award phrases:
        [
            {
                "normalized": "best performance by an actor in a motion picture - drama",
                "total_count": 123,
                "top_variants": [{"text": "Best Performance by an Actor in a Motion Picture - Drama", "count": 80}, ...]
            },
        ...
        ]
    """
    out = []
    for norm, total in AWARD_FREQ.most_common():
        variants = AWARD_VARIANTS.get(norm, Counter())
        out.append({
            "normalized": norm,
            "total_count": int(total),
            "top_variants": [{"text": t, "count": int(c)} for t, c in variants.most_common(5)],
        })
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

def filter_name(text: str) -> List[str]:
    """Return all PERSON entity spans from text using spaCy."""
    doc = nlp(text)
    return [ent.text for ent in doc.ents if ent.label_ == "PERSON"]

def split3(text: str, pat: re.Pattern) -> Optional[Tuple[str, str, str]]:
    """Apply a 3-group regex to text; return (L, anchor, R) or None."""
    m = pat.search(text)
    if not m:
        return None
    return m.group(1).strip(), m.group(2).strip(), m.group(3).strip()

def mk_candidate(rule_id: str, award_name: str, anchor_text: str, subject: str) -> Candidate:
    """Construct a Candidate object."""
    return Candidate(rule_id=rule_id, award_name=award_name, anchor_text=anchor_text, subject=subject)

# ---------- Candidate generation ----------

def generate_from_text(text: str, base: Dict, segment: str, max_left: int, max_right: int) -> List[Candidate]:
    """
    Extract (award, winner) candidates from a single tweet/text.

    Flow:
    1) Try WIN_B: 'Award goes to Entity' → PERSON on right, award on left.
    2) Else try WIN_A: 'Entity wins Award' → PERSON on left, award on right.
    3) Use spaCy NER for PERSON, regex heuristics for 'Best …' award phrase.
    4) Skip if award is unrecognizable (i.e., extractor returns None).
    """
    cands: List[Candidate] = []

    # WIN_B: Award goes to Entity
    sb = split3(text, ANCHORS["WIN_B"])
    if sb:
        L, anchor, R = sb
        names = filter_name(R)
        if names:
            subject = names[0]
            award_name = extract_award_from_side(L)
            if award_name:  # omit unrecognizable awards
                cands.append(mk_candidate("WIN_B", award_name, anchor, subject))
                return cands  # prefer WIN_B when both might match

    # WIN_A: Entity wins Award
    sa = split3(text, ANCHORS["WIN_A"])
    if sa:
        L, anchor, R = sa
        names = filter_name(L)
        if names:
            subject = names[0]
            award_name = extract_award_from_side(R)
            if award_name:  # omit unrecognizable awards
                cands.append(mk_candidate("WIN_A", award_name, anchor, subject))

    return cands
