# src/candidate_pipeline.py

from __future__ import annotations                 # Enable postponed evaluation of type annotations (helps with forward refs)
import argparse, json, os                          # Standard libs: CLI parsing (argparse), JSON handling (json), OS utilities (os)
from dataclasses import dataclass, asdict          # Dataclass for lightweight containers; asdict to serialize dataclass → dict
from typing import Dict, List, Optional, Tuple     # Type hints for clarity and static checking
import regex as re                                 # Use the third-party 'regex' module (more powerful than stdlib 're')
from ftfy import fix_text                          # ftfy: fixes mojibake/unicode issues in text
from unidecode import unidecode                    # unidecode: strip accents/diacritics, map unicode → closest ASCII
from collections import Counter                   # Counter: fast frequency table (string → count)

# import nltk #TODO NLTK ran worse than spacy, but leaving this in in case we change our minds
# from nltk import word_tokenize, pos_tag, ne_chunk

# nltk.download('punkt_tab')
# nltk.download('averaged_perceptron_tagger_eng')
# nltk.download('maxent_ne_chunker_tab')
# nltk.download('words')

import spacy #NOTE: "python -m spacy download en_core_web_sm" needs to be run to install the small english model
# Or we could pick a diffent model

# Load the small English model
nlp = spacy.load("en_core_web_sm")                # Load spaCy English model used for PERSON name extraction


# ====== Text normalization & award extraction (no awards.txt) ======
DASHES = r"[\u2012\u2013\u2014\u2212-]"          # Regex char class: normalize various dash characters to simple hyphen '-'
SPACE = re.compile(r"\s+")                        # Regex: collapse runs of whitespace to a single space
URL = re.compile(r"https?://\S+")                 # Regex: find URLs (http/https + non-space)
HANDLE = re.compile(r"@\w+")                      # Regex: Twitter handles like @username
HASHTAG = re.compile(r"#\w+")                     # Regex: hashtags like #GoldenGlobes
PUNCT_STRIP = re.compile(r"[\"'“”‘’`(){}\[\]]")   # Regex: strip quote-like and bracket punctuation (helps phrase cleanup)
BEST_SPAN = re.compile(                           # Regex: capture a broad "Best …" span (core heuristic for award phrases)
    r"\b(best\s+[a-z0-9&/,\-.\s]{3,120})", re.I)  # - starts with 'best', then 3–120 allowed chars; case-insensitive
TRIM_AT = re.compile(r"[.!?;:|]")                 # Regex: sentence-stopping punctuation where we truncate noisy tails

# frequency table of learned awards (normalized -> Counter of original variants)
AWARD_FREQ = Counter()                            # Tracks normalized award phrase frequency across the run
AWARD_VARIANTS = {}  # normalized -> Counter(original -> count)  # Map normalized form → counts of original textual variants

def normalize_text(s: str) -> str:
    # PURPOSE: Apply consistent text cleanup to increase recall and reduce false matches.
    #          - fix unicode
    #          - remove URLs/handles/hashtags
    #          - unify dashes
    #          - strip punctuation noise
    #          - lowercase and normalize 'tv'/'&'
    s = s or ""                                   # Guard against None
    s = fix_text(unidecode(s))                    # Normalize unicode, remove diacritics, fix broken encodings
    s = URL.sub(" ", s)                           # Remove URLs to avoid polluting award phrases
    s = HANDLE.sub(" ", s)                        # Remove @handles
    s = HASHTAG.sub(" ", s)                       # Remove #hashtags
    s = re.sub(DASHES, "-", s)                    # Convert all dash variants to '-'
    s = PUNCT_STRIP.sub(" ", s)                   # Strip selected punctuation that often fragments phrases
    s = s.lower().replace("&", "and").replace(" tv ", " television ")  # Normalize '&' and 'tv' to canonical words
    s = SPACE.sub(" ", s).strip()                 # Collapse spaces and trim
    return s                                      # Return cleaned, normalized text

def extract_award_from_side(side_text: str) -> Optional[str]:
    """Pull a likely award name from the given side of the anchor, no dictionary."""
    # PURPOSE: Identify a candidate award phrase from a chunk of text (L or R of anchor).
    # STRATEGY:
    #   1) If we can find an explicit "Best ..." span, use that.
    #   2) Otherwise, use the side as-is but trim at strong punctuation.
    #   3) Require the phrase to contain "best" to avoid random junk (tunable).
    #   4) Normalize and record frequency; return the most common original variant for that normalized key.
    if not side_text:
        return None                               # Nothing to extract from
    # Prefer explicit "Best …"
    m = BEST_SPAN.search(side_text)               # Look for "Best ..." span anywhere
    cand = m.group(1) if m else side_text         # If found, take that capture; else use the whole side
    # Trim at first hard punctuation
    cut = TRIM_AT.search(cand)                    # Stop at ., !, ?, ;, :, |
    if cut:
        cand = cand[:cut.start()]                 # Keep text up to (but not including) the punctuation
    cand = cand.strip(" -—~·•\n\t ").strip()      # Strip whitespace and common separators from ends
    if not cand:
        return None                               # Empty after trimming: no candidate
    # Require "best" to avoid junk (adjust if needed)
    if "best" not in cand.lower():
        return None                               # Heuristic gate: only consider phrases containing 'best'
    # Record normalized + original for later consolidation
    norm = normalize_text(cand)                   # Normalize the candidate for stable counting
    if len(norm) < 8:  # too short to be meaningful
        return None                               # Very short phrases are likely noise; skip
    AWARD_FREQ[norm] += 1                         # Increment frequency of the normalized form
    AWARD_VARIANTS.setdefault(norm, Counter())[cand] += 1  # Count this original variant under that normalized key
    # Return a simple canonical: most common original for this normalized form
    most_common_original = AWARD_VARIANTS[norm].most_common(1)[0][0]  # Representative original string
    return most_common_original                    # Return representative (human-readable) variant

def dump_learned_awards(path: str = "learned_awards.json"):
    """Optional: write learned award clusters after a run."""
    # PURPOSE: Persist a summary of all award phrases observed during the run so you can inspect what was learned.
    out = []                                       # Will hold a list of dicts describing normalized forms and top variants
    for norm, total in AWARD_FREQ.most_common():   # Iterate through normalized phrases sorted by frequency
        variants = AWARD_VARIANTS.get(norm, Counter())  # Get variant counts for this normalized form
        out.append({
            "normalized": norm,                    # The normalized key
            "total_count": int(total),             # How many times we observed it
            "top_variants": [{"text": t, "count": int(c)} for t, c in variants.most_common(5)]  # Top original spellings
        })
    with open(path, "w", encoding="utf-8") as f:   # Write JSON file to disk
        json.dump(out, f, ensure_ascii=False, indent=2)  # Pretty-print UTF-8 JSON

# ---------- Anchors & simple resources ----------

ANCHORS = {
    # Entity wins Award  -> left entity, right award
    "WIN_A": re.compile(r"(.+?)\s+(wins?|receives?|gets|takes\s+home|is\s+awarded)\s+(.+)", re.I),  # 3 groups: L anchor R
    # Award goes to Entity -> left award, right entity
    "WIN_B": re.compile(r"(.+?)\s+(goes\s+to|awarded\s+to)\s+(.+)", re.I),                           # 3 groups: L anchor R
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

AWARD_KEYWORDS = {}                                # Legacy dict for dictionary-driven matching (unused in the no-list flow)
EDGE_STOPWORDS = {"the","a","an","of","for","and","or","to","in"}  # Not used in current extractor; left for possible use
PUNCT_OR_BREAK = re.compile(r"[.!?,:;]| {2,}")     # Helper regex: punctuation or big whitespace gap (used in suffix enumerator)

# ---------- Data model ----------

@dataclass
class Candidate:
    rule_id: str                                   # Which rule fired: 'WIN_A' or 'WIN_B'
    award_name: str                   # raw substring (may have #/@)  # The extracted award phrase (raw/canonicalized variant)
    anchor_text: str                               # The verb phrase (e.g., 'is awarded', 'goes to', 'wins')
    subject: str                                   # The recognized PERSON (winner) from spaCy

# ---------- Small helpers ----------

def enumerate_prefixes(tokens: List[str], max_len: int):
    # PURPOSE: Return the first N tokens as a single string (legacy utility; unused in the current flow).
    return ' '.join(tokens[0:max_len])             # Join tokens[0:max_len] with spaces

def enumerate_suffixes(tokens: List[str], max_len: int) -> List[str]:
    # PURPOSE: Build incremental suffix strings from the end, stopping at punctuation.
    out, run = [], []                              # out accumulates suffixes, run is a working buffer
    for i in range(1, min(max_len, len(tokens)) + 1):  # Iterate from 1 up to max_len or number of tokens
        tok = tokens[-i]                           # Take the i-th token from the end
        if PUNCT_OR_BREAK.search(tok): break      # Stop at punctuation or big gaps
        run.insert(0, tok)                         # Prepend token to build suffix
        out.append(" ".join(run))                  # Append the current suffix string
    return out                                     # Return list of incremental suffixes (unused in current extractor)

# def filter_award_name(text: str):
#     for keyword in AWARD_KEYWORDS:
#         pattern = rf"\b{re.escape(keyword)}\b" #Now it's pulling from AWARD_KEYWORDS that is populated by the awrds.txt file
#         match = re.search(pattern, text, re.IGNORECASE) #Added ignorecase
#         if match:
#             return match.group()
    
#     return "unrecognizable award" #"false"

def filter_award_name(text: str) -> str:
    # PURPOSE: Dictionary-free award selection. Try to extract a "Best …" phrase from the given text (side of anchor).
    # RETURNS: A human-readable award phrase (most common original variant) or "unrecognizable award".
    # NOTE: This delegates to extract_award_from_side(), which also logs frequencies for learning.
    return extract_award_from_side(text) or "unrecognizable award"  # Fallback string if nothing reasonable was found

#TODO NLTK ran worse than spacy, but leaving this in in case we change our minds
# def filter_name(text: str): #Uses NLTK to find names
#     tokens = word_tokenize(text)
#     pos_tags = pos_tag(tokens)
#
#     # Named Entity Recognition
#     tree = ne_chunk(pos_tags)
#
#     # Extract named entities (like people, organizations, locations)
#     names = []
#     for subtree in tree:
#         if hasattr(subtree, 'label') and subtree.label() == 'PERSON':
#             name = " ".join([token for token, pos in subtree.leaves()])
#             names.append(name)
#
#     return names

def filter_name(text: str): #Uses spacy to find names
    # PURPOSE: Run spaCy NER and return all PERSON entity spans as text (can be one or multiple tokens).
    # Process the text
    doc = nlp(text)                                 # Run spaCy pipeline (tokenization, NER, etc.)
    # Extract entities recognized as PERSON
    names = [ent.text for ent in doc.ents if ent.label_ == "PERSON"] #Spacy can recognize 1 or 2 words as a single person entity
    # 'names' is a list of strings like ["Ben Affleck", "Kathryn Bigelow"]
    return names                                    # Return possibly empty list; caller should guard against [] when indexing

def split3(text: str, pat: re.Pattern) -> Optional[Tuple[str,str,str]]:
    # PURPOSE: Apply a 3-group regex to text and return (L, anchor, R) substrings if matched.
    m = pat.search(text)                            # Search pattern anywhere in the string
    if not m: return None                           # If no match, return None
    return m.group(1).strip(), m.group(2).strip(), m.group(3).strip()  # Strip whitespace and return tuple

def mk_candidate(rule_id: str, award_name: str, anchor_text: str, subject: str):
    # PURPOSE: Construct a Candidate dataclass instance (simple factory).
    return Candidate(
        award_name=award_name,                      # Award phrase (string)
        rule_id=rule_id,                            # Rule identifier ('WIN_A' or 'WIN_B')
        anchor_text=anchor_text,                    # The anchor verb phrase matched
        subject=subject                             # Winner/person extracted via spaCy
    )

# ---------- Step 4: Candidate generation ----------

# def generate_from_text(text: str, base: Dict, segment: str,
#                        max_left: int, max_right: int) -> List[Candidate]:
#     cands: List[Candidate] = []
#
#     # Winners B: Award goes to Entity
#     x = split3(text, ANCHORS["WIN_B"])
#     #print(x)
#     if x:
#         L, anchor, R = x
#         # award_name = enumerate_suffixes(L.split(), max_left)[-1]
#         subject = filter_name(R)[0] #enumerate_prefixes(R.split(" "), max_right)
#         award_name = filter_award_name(L)
#         cands.append(mk_candidate(rule_id="WIN_B", award_name=award_name, anchor_text=anchor, subject=subject))
#     else:
#         # Winners A: Entity wins Award
#         x = split3(text, ANCHORS["WIN_A"])
#         
#         if x:
#             L, anchor, R = x
#             subject = filter_name(L)[0] #enumerate_suffixes(L.split(), max_left)[-1]
#             # award_name = enumerate_prefixes(R.split(" "), max_right)
#             award_name = filter_award_name(R)
#             cands.append(mk_candidate(rule_id="WIN_A", award_name=award_name, anchor_text=anchor, subject=subject))
#
#     return cands

def generate_from_text(text: str, base: Dict, segment: str,
                       max_left: int, max_right: int) -> List[Candidate]:
    # PURPOSE: Given a tweet/text, apply anchor rules to extract (award, winner) candidates.
    # INPUTS:
    #   text: the raw tweet string
    #   base/segment/max_left/max_right: legacy args (unused in current logic), kept for interface compatibility
    # FLOW:
    #   1) Try WIN_B: "Award goes to Entity" → PERSON is on the RIGHT, award phrase on the LEFT.
    #   2) Else try WIN_A: "Entity wins Award" → PERSON is on the LEFT, award phrase on the RIGHT.
    #   3) Use spaCy NER to get PERSON; use regex heuristics to get award phrase ("Best …").
    #   4) Return zero, one, or multiple Candidate(s) (currently at most one per pattern tested).
    cands: List[Candidate] = []                     # Accumulator for any candidates found

    # Try "Award goes to Entity" first
    x = split3(text, ANCHORS["WIN_B"])             # Attempt to split into (L, anchor, R) using WIN_B pattern
    if x:
        L, anchor, R = x                           # L = likely award side; R = likely person side
        names = filter_name(R)  # PERSON should be on the right
        if not names:
            return cands  # no person found; skip     # Guard: if spaCy finds no PERSON, yield nothing for this text
        subject = names[0]                          # Choose the first PERSON entity (policy: first match wins)
        award_name = extract_award_from_side(L) or filter_award_name(L)  # Prefer explicit extractor; fallback to same
        cands.append(mk_candidate(rule_id="WIN_B", award_name=award_name, anchor_text=anchor, subject=subject))
        return cands                                # Early return since we found a WIN_B candidate

    # Else try "Entity wins Award"
    x = split3(text, ANCHORS["WIN_A"])             # If WIN_B failed, try WIN_A split
    if x:
        L, anchor, R = x                           # L = likely person side; R = likely award side
        names = filter_name(L)  # PERSON should be on the left
        if not names:
            return cands                            # No PERSON → no candidate for this text
        subject = names[0]                          # First PERSON match
        award_name = extract_award_from_side(R) or filter_award_name(R)  # Extract award phrase from the right side
        cands.append(mk_candidate(rule_id="WIN_A", award_name=award_name, anchor_text=anchor, subject=subject))

    return cands                                    # Return collected candidates (0 or 1 in current logic)


# #Populating AWARD_KEYWORDS from the file
# with open("awards.txt", "r", encoding="utf-8") as f: 
#     for line in f:
#         keyword = line.strip()
#         if keyword:  # skipping empty lines
#             AWARD_KEYWORDS[keyword] = ""  # setting default value
#
# #Testing ideal case (works well)
# print(generate_from_text("RT @CNNshowbiz: Best original score - motion picture is awarded to Mychael Danna for the \"Life of Pi\"", {}, "raw", 8, 2))
# #Testing if Spacy recognizes single name person (it does)
# print(generate_from_text("RT @moderndestiny: I really, really, really hope Adele wins a Globe tonight. She looks GORG in @Burberry #GoldenGlobes @ERedCarpet", {}, "raw", 8, 2))
# #Testing if Spacy gets confused by multiple names (it just takes the first one; we can talk about intended behavior)
# print(generate_from_text("RT @_thebrunetteone: Ben Affleck, you are hot. I hope you win. I hope Kathryn Bigelow wins too, but I hope you win. #GoldenGlobes", {}, "raw", 8, 2))


