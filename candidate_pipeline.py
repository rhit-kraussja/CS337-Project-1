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
    # # Presenters
    # "PRESENT_FWD": re.compile(r"(?:present(?:s|ed)?|introduce(?:s|d)?)\s+(.+)", re.I),
    # "PRESENT_BY": re.compile(r"presented\s+by\s+(.+)", re.I),
    # # Nominees
    # "NOMINEES": re.compile(r"(?:nominee[s]?\s+(?:are|for)|nominated\s+for)\s+(.+)", re.I),
    # # Host trigger
    # "HOST": re.compile(r"(?:host(?:s|ed|ing)?\s+(?:tonight|the\s+show|the\s+golden\s+globes)|our\s+host(?:s)?\s+is)\b", re.I),
    # # Broad "Best ..." net from anywhere
    # "BEST_NET": re.compile(r"\bbest\b.{0,120}", re.I | re.DOTALL),
}

AWARD_KEYWORDS = {
    
}
EDGE_STOPWORDS = {"the","a","an","of","for","and","or","to","in"}
PUNCT_OR_BREAK = re.compile(r"[.!?,:;]| {2,}")

# ---------- Data model ----------

@dataclass
class Candidate:
    rule_id: str
    award_name: str                   # raw substring (may have #/@)
    anchor_text: str
    subject: str

# ---------- Small helpers ----------

def enumerate_prefixes(tokens: List[str], max_len: int):
    return ' '.join(tokens[0:max_len])

def enumerate_suffixes(tokens: List[str], max_len: int) -> List[str]:
    out, run = [], []
    for i in range(1, min(max_len, len(tokens)) + 1):
        tok = tokens[-i]
        if PUNCT_OR_BREAK.search(tok): break
        run.insert(0, tok)
        out.append(" ".join(run))
    return out

def filter_award_name(text: str):
    pattern = r"Best original score"
    match = re.search(pattern, text)
    if match:
        return match.group()
    return "false"

def split3(text: str, pat: re.Pattern) -> Optional[Tuple[str,str,str]]:
    m = pat.search(text)
    if not m: return None
    return m.group(1).strip(), m.group(2).strip(), m.group(3).strip()

def mk_candidate(rule_id: str, award_name: str, anchor_text: str, subject: str):
    return Candidate(
        award_name=award_name,
        rule_id=rule_id,
        anchor_text=anchor_text,
        subject=subject
    )

# ---------- Step 4: Candidate generation ----------

def generate_from_text(text: str, base: Dict, segment: str,
                       max_left: int, max_right: int) -> List[Candidate]:
    cands: List[Candidate] = []

    # Winners B: Award goes to Entity
    x = split3(text, ANCHORS["WIN_B"])
    print(x)
    if x:
        L, anchor, R = x
        # award_name = enumerate_suffixes(L.split(), max_left)[-1]
        subject = enumerate_prefixes(R.split(" "), max_right)
        award_name = filter_award_name(L)
        cands.append(mk_candidate(rule_id="WIN_B", award_name=award_name, anchor_text=anchor, subject=subject))
    else:
        # Winners A: Entity wins Award
        x = split3(text, ANCHORS["WIN_A"])
        
        if x:
            L, anchor, R = x
            subject = enumerate_suffixes(L.split(), max_left)[-1]
            # award_name = enumerate_prefixes(R.split(" "), max_right)
            award_name = filter_award_name(R)
            cands.append(mk_candidate(rule_id="WIN_A", award_name=award_name, anchor_text=anchor, subject=subject))

    return cands

print(generate_from_text("RT @CNNshowbiz: Best original score - motion picture is awarded to Mychael Danna for the \"Life of Pi\"", {}, "raw", 8, 2))


