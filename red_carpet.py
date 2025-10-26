import re
from collections import defaultdict, Counter
import spacy
import json

nlp = spacy.load("en_core_web_sm")

PATTERNS = {
    "best_dressed": re.compile(
        r"\b(best\s+dressed|looked\s+(?:amazing|incredible|stunning|gorgeous))\b", re.I
    ),
    "worst_dressed": re.compile(
        r"\b(worst\s+dressed|terrible\s+outfit|looked\s+(?:awful|bad|terrible|horrible))\b", re.I
    ),
    "most_discussed": re.compile(
        r"\b(everyone(?:'s| is)\s+talking\s+about|most\s+talked\s+about|trending|viral)\b", re.I
    ),
    "most_controversial": re.compile(
        r"\b(controversial|divisive|caused\s+a\s+stir|stirring\s+up|mixed\s+reactions)\b", re.I
    ),
}

def extract_persons(text):
    doc = nlp(text)
    persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    if not persons:
        # fallback: Titlecase names like "Zendaya", "Jared Leto"
        persons = re.findall(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*", text)
    return persons

def classify_tweet(tweet):
    labels = []
    for label, pattern in PATTERNS.items():
        if pattern.search(tweet):
            labels.append(label)
    persons = extract_persons(tweet)
    return [(p, label) for p in persons for label in labels]

def aggregate_labels(tweets):
    category_counts = defaultdict(Counter)
    for t in tweets:
        for person, label in classify_tweet(t):
            category_counts[label][person] += 1
    data = {label: counts.most_common(5) for label, counts in category_counts.items()}
    with open("red_carpet.json", "w") as file:
        json.dump(data, file, indent=4)
    return data