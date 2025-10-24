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

from typing import Optional
from collections import Counter
from rapidfuzz import process, fuzz

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
    # Presenters
    "PRESENT_A": re.compile(r"(.+?)\s+(present(?:s|ed)?|introduce(?:s|d)?)\s+(.+)", re.I),
    "PRESENT_B": re.compile(r"(.+?)\s+((presented|introduced)\s+by\s)(.+)", re.I),
    # Nominees
    # A: Nominees for <award>: <names>
    "NOMINEES_A": re.compile(r"(nominees?\s+for)\s+([A-Z][\w\s]+):\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:,\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)*)", re.I),
    # B: <Name> nominated for <award> OR <Name> up for <award>
    "NOMINEES_B": re.compile(r"([A-Z][a-z]+(?: [A-Z][a-z]+)*) (?:is|was|has been)? (nominated for|up for) ([A-Z][\w\s]+)", re.I),
    # Host trigger
    "HOST": re.compile(r"(?:host(?:s|ed|ing)?\s+(?:tonight|the\s+show|the\s+golden\s+globes)|our\s+host(?:s)?\s+is)\b", re.I),
    # Broad "Best ..." net from anywhere
    "BEST_NET": re.compile(r"\bbest\b.{0,120}", re.I | re.DOTALL),
}

# ---------- Data model ----------

@dataclass
class Candidate:
    rule_id: str        # "WIN_A" or "WIN_B"
    award_name: str     # extracted award phrase
    anchor_text: str    # the matched anchor verb phrase
    subject: str        # PERSON name (winner)

# ---------- Helpers ----------

# Load known awards
with open("awards.txt", encoding="utf-8") as f:
    KNOWN_AWARDS = [line.strip() for line in f if line.strip()]


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

# Precompute normalized known awards for matching
NORMALIZED_AWARDS = {normalize_text(a): a for a in KNOWN_AWARDS}

def best_fuzzy_match(candidate: str, known_dict, cutoff: float = 60) -> Optional[str]:
    """
    Return best fuzzy match from normalized known_dict or None if no good match.
    candidate: normalized string
    known_dict: {normalized_award: original_award}
    cutoff: minimum similarity score (0–100)
    """
    match = process.extractOne(candidate, known_dict.keys(), scorer=fuzz.token_sort_ratio)
    if match and match[1] >= cutoff:
        norm_match = match[0]
        return known_dict[norm_match]
    return None

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
    #return AWARD_VARIANTS[norm].most_common(1)[0][0] # Used if we want to use our learned awards rather than pulling from file

    # Try to match against known awards
    matched_award = best_fuzzy_match(norm, NORMALIZED_AWARDS)
    return matched_award

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

def filter_movie(text: str) -> List[str]:
    """Return all WORK_OF_ART entity spans from text using spaCy."""
    doc = nlp(text)
    return [ent.text for ent in doc.ents if ent.label_ == "WORK_OF_ART"]

def actor_award(award_name: str) -> bool:
    """Check if the award is looking for an actor"""
    award_name = award_name.lower()

    person_keywords = [
        "actor", "actress", "director", "score", "screenplay"
    ]
    # film_keywords = [ # For if we want to deliniate film also and have a third category of unknown to run another check on
    #     "picture", "film", "feature", "movie", "cinematography",
    #     "editing", "sound", "score", "design", "visual effects", "makeup"
    # ]

    if any(word in award_name for word in person_keywords):
        return True
    # elif any(word in award_name for word in film_keywords): # For if we want to deliniate film also and have a third category of unknown to run another check on
    #     return "film"
    else:
        return False

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
        award_name = extract_award_from_side(L)
        if award_name:  # omit unrecognizable awards
            if actor_award(award_name):
                subject = filter_name(R)
            else:
                subject = filter_movie(R)
            if subject:
                subject = subject[0]
                cands.append(mk_candidate("WIN_B", award_name, anchor, subject))
                return cands  # prefer WIN_B when both might match

    # WIN_A: Entity wins Award
    sa = split3(text, ANCHORS["WIN_A"])
    if sa:
        L, anchor, R = sa
        award_name = extract_award_from_side(R)
        if award_name:  # omit unrecognizable awards
            if actor_award(award_name):
                subject = filter_name(L)
            else:
                subject = filter_movie(L)
            if subject:
                subject = subject[0]
                cands.append(mk_candidate("WIN_A", award_name, anchor, subject))
                return cands

    # PRESENT_B: Award presented by Entity
    sb = split3(text, ANCHORS["PRESENT_B"])
    if sb:
        L, anchor, R = sb
        award_name = extract_award_from_side(L)
        if award_name:  # omit unrecognizable awards
            if actor_award(award_name):
                subject = filter_name(R)
            else:
                subject = filter_movie(R)
            if subject:
                subject = subject[0]
                cands.append(mk_candidate("PRESENT_B", award_name, anchor, subject))
                return cands 

    # PRESENT_A: Entity presents Award
    sa = split3(text, ANCHORS["PRESENT_A"])
    if sa:
        L, anchor, R = sa
        award_name = extract_award_from_side(R)
        if award_name:  # omit unrecognizable awards
            if actor_award(award_name):
                subject = filter_name(L)
            else:
                subject = filter_movie(L)
            if subject:
                subject = subject[0]
                cands.append(mk_candidate("PRESENT_A", award_name, anchor, subject))
                return cands


    # # Nominees
    sa = split3(text, ANCHORS["NOMINEES_A"])
    if sa:
        anchor, L, R = sa
        award_name = extract_award_from_side(L)
        if award_name:
            nominees = R.split(",")
            subject = []
            if actor_award(award_name):
                for nominee in nominees:
                    subject.append(filter_name(nominee))
            else:
                for nominee in nominees:
                    subject.append(filter_movie(nominee))
            if len(subject) > 0:
                cands.append(mk_candidate["NOMINEES_A", award_name, subject])

    sa = split3(text, ANCHORS["NOMINEES_B"])
    if sa:
        L, anchor, R = sa
        award_name = extract_award_from_side(R)
        if award_name:
            if actor_award(award_name):
                subject = filter_name(L)
            else:
                subject = filter_name(L)
            if len(subject) > 0:
                cands.append(mk_candidate["NOMINEES_B", award_name, subject])

    # # Host
    # if ANCHORS["HOST"].search(text):
    #     cands.append(mk_candidate(base, entity_type="host", rule_id="HOST",
    #                               span_text=text, anchor_text="host", side=None, segment=segment))

    # # Best-net (award-like phrase anywhere)
    # m = ANCHORS["BEST_NET"].search(text)
    # if m:
    #     cands.append(mk_candidate(base, entity_type="award", rule_id="BEST_NET",
    #                               span_text=m.group(0).strip(), anchor_text="best", side=None, segment=segment))
    return cands
