import pandas as pd
import spacy
from spacy.tokens import DocBin
from sklearn.model_selection import train_test_split
import random

# Load dataset
data = pd.read_csv("IMDb_All_Genres_etf_clean1.csv")

# Keep only relevant columns
data = data[['Movie_Title', 'Director', 'Actors']].fillna('')

# Function to split actors
def split_actors(actors_str, max_count=3):
    actors = [a.strip() for a in actors_str.split(',')]
    # pad with empty strings if less than max_count
    actors += [''] * (max_count - len(actors))
    return actors[:max_count]

data = data[data['Movie_Title'] != '']

# Load blank English model
nlp = spacy.blank("en")

# Function to generate multiple natural sentences for one movie
def generate_sentences(row, num_sentences=3):
    actors = split_actors(row['Actors'])
    actors = [a for a in actors if a]
    sentences = []

    for _ in range(num_sentences):
        # Randomly select an actor if available
        actor = random.choice(actors) if actors else None
        
        # Define award-show-style templates
        templates = [
            f"{actor} won an award for {row['Movie_Title']}",
            f"{row['Movie_Title']}, directed by {row['Director']} was nominated for an award.",
            f"I can't believe {row['Movie_Title']} didn't win",
            f"At the awards ceremony, {actor}'s performance in {row['Movie_Title']}, directed by {row['Director']}, stole the show.",
            f"Nominated for Best Actor, {actor} gave a powerhouse performance in {row['Movie_Title']}, directed by {row['Director']}.",
            f"When {row['Movie_Title']} was announced as a contender for Best Picture, the standout performance by {actor} and the visionary direction of {row['Director']} were key talking points.",
            f"As {actor} took the stage to accept their award for {row['Movie_Title']}, it was clear that {row['Director']}'s direction had shaped an unforgettable cinematic experience.",
            f"The chemistry between {actor} and {row['Director']} in {row['Movie_Title']} made it one of the top contenders at this year’s awards.",
            f"When {row['Movie_Title']} was nominated for Best Director, {actor}'s incredible performance, combined with {row['Director']}'s direction, made it a front-runner for multiple awards.",
            f"The buzz around {actor}'s performance in {row['Movie_Title']} only grew after the film's multiple nominations, thanks to {row['Director']}'s impeccable direction.",
            f"At the awards show, the applause was deafening when {actor} and {row['Director']} were recognized for their collaboration on {row['Movie_Title']}.",
            f"It wasn't just the direction that earned {row['Movie_Title']} its nominations—{actor}'s stellar performance was also in the spotlight at this year's awards.",
            f"When {actor} won Best Actor for {row['Movie_Title']}, the entire room knew it was due in no small part to {row['Director']}'s masterful vision.",
            f"In a year filled with strong contenders, {row['Movie_Title']} stood out—thanks to {actor}'s award-worthy performance and {row['Director']}'s direction.",
            f"Everyone's talking about the stunning win for {row['Movie_Title']} at this year's awards, with {actor}'s performance and {row['Director']}'s direction leading the charge.",

            # --- actor + title ---
            f"{actor} was nominated for their outstanding role in {row['Movie_Title']}." if actor else None,
            f"The crowd went wild when {actor} won for {row['Movie_Title']}." if actor else None,
            f"{actor}'s portrayal in {row['Movie_Title']} left critics and fans speechless during award season." if actor else None,
            f"Everyone expected {actor} to take home the award for {row['Movie_Title']} — and they did!" if actor else None,

            # --- title + director ---
            f"{row['Movie_Title']}, under the direction of {row['Director']}, dominated the awards this year.",
            f"{row['Director']} took home Best Director for their work on {row['Movie_Title']}.",
            f"The visual storytelling in {row['Movie_Title']} proved why {row['Director']} is one of the best in the industry.",
            f"{row['Movie_Title']} was a landmark achievement for {row['Director']} at the ceremony.",

            # --- title only ---
            f"{row['Movie_Title']} was the surprise winner of the night!",
            f"Fans cheered as {row['Movie_Title']} picked up multiple awards.",
            f"Can you believe {row['Movie_Title']} didn’t win Best Picture?",
            f"{row['Movie_Title']} was the talk of the entire awards show.",

            # --- actor only ---
            f"{actor} delivered one of the most emotional acceptance speeches of the night." if actor else None,
            f"Everyone’s still talking about {actor}’s red carpet appearance." if actor else None,
            f"{actor} finally got the recognition they deserved this award season." if actor else None,
            f"{actor} continues to dominate award shows year after year." if actor else None,

            # --- director only ---
            f"{row['Director']} received a standing ovation during the ceremony.",
            f"Critics praised {row['Director']} for pushing creative boundaries this award season.",
            f"{row['Director']}'s artistic vision earned them the top directing honor of the night.",
        ]
        templates = [t for t in templates if t]

        # Choose a random sentence
        template = random.choice(templates)

        # Build entities list dynamically
        entities = []
        if actor:
            entities.append(("PERSON", actor))
        if row.get('Director'):
            entities.append(("PERSON", row['Director']))
        if row.get('Movie_Title'):
            entities.append(("TITLE", row['Movie_Title']))    
        sentences.append((template, entities))

    return sentences

# Create Doc from sentence and entity list
def create_doc_from_sentence(sentence, entities):
    doc = nlp.make_doc(sentence)
    spans = []

    for label, text in entities:
        start = sentence.find(text)
        if start != -1:
            end = start + len(text)
            span = doc.char_span(start, end, label=label, alignment_mode="expand")
            if span:
                spans.append(span)

    # Filter overlapping spans: keep the first span that starts earliest
    non_overlapping = []
    spans = sorted(spans, key=lambda s: (s.start, -s.end))
    for span in spans:
        if all(span.end <= s.start or span.start >= s.end for s in non_overlapping):
            non_overlapping.append(span)

    doc.ents = non_overlapping
    return doc

# Split data
train_df, dev_df = train_test_split(data, test_size=0.2, random_state=42)

# Create DocBin
def create_docbin(df, filename, num_sentences=3):
    doc_bin = DocBin()
    for _, row in df.iterrows():
        sentence_entities_list = generate_sentences(row, num_sentences)
        for sentence, entities in sentence_entities_list:
            doc = create_doc_from_sentence(sentence, entities)
            if doc.ents:
                doc_bin.add(doc)
    doc_bin.to_disk(filename)
    print(f"Saved {len(df) * num_sentences} (approx) examples to {filename}")

# Save train and dev DocBins
create_docbin(train_df, "train.spacy", num_sentences=3)
create_docbin(dev_df, "dev.spacy", num_sentences=3)
