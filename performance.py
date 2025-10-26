import json
import spacy
import re

nlp = spacy.load("en_core_web_sm")

def extract_performance_info(tweet):
    doc = nlp(tweet)
    performers = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    opinions = re.findall(r"\b(loved|hated|amazing|terrible|incredible|boring|fantastic)\b", tweet, re.I)
    performance = {"performers": performers, "opinions": opinions}
    if performers and opinions:
        try:
            with open("performances.json", 'x') as f:
                json.dump([], f)  # Initialize with an empty list
        except FileExistsError:
            pass

        with open("performances.json", 'r') as f:
            data = json.load(f)
        data.append(performance)
        with open("performances.json", "w") as file:
            json.dump(data, file, indent=4)
    return performance