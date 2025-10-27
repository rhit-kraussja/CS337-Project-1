'''Version 0.5'''
from dataclasses import asdict
import json
from pathlib import Path
from tqdm import tqdm
import spacy
from candidate_pipeline import (
    extract_red_carpet_batch,
    extract_performance_info_batch,
    aggregate_red_carpets,
    aggregate_performances, 
    generate_from_text, 
    dump_learned_awards
)

INPUT  = Path("gg2013.json")
OUT    = Path("candidates.json")

# Year of the Golden Globes ceremony being analyzed
YEAR = "2013"

# Global variable for hardcoded award names
# This list is used by get_nominees(), get_winner(), and get_presenters() functions
# as the keys for their returned dictionaries
# Students should populate this list with the actual award categories for their year, to avoid cascading errors on outputs that depend on correctly extracting award names (e.g., nominees, presenters, winner)
AWARD_NAMES = [
    "best motion picture - drama",
    "best motion picture - comedy or musical",
    "best performance by an actor in a motion picture - drama",
    # Add or modify categories as needed for your year
    "your custom award category",
    # ... etc
]

data = None  # Initialize, but it will be set in main()

def get_hosts(year):
    '''Returns the host(s) of the Golden Globes ceremony for the given year.
    
    Args:
        year (str): The year of the Golden Globes ceremony (e.g., "2013")
    
    Returns:
        list: A list of strings containing the host names. 
              Example: ["Seth Meyers"] or ["Tina Fey", "Amy Poehler"]
    
    Note:
        - Do NOT change the name of this function or what it returns
        - The function should return a list even if there's only one host
    '''
    if not data or "hosts" not in data:
        raise ValueError(f"Data for year {year} not found or 'hosts' key is missing.")
    return data["hosts"]

def get_awards(year):
    '''Returns the list of award categories for the Golden Globes ceremony.
    
    Args:
        year (str): The year of the Golden Globes ceremony (e.g., "2013")
    
    Returns:
        list: A list of strings containing award category names.
              Example: ["Best Motion Picture - Drama", "Best Motion Picture - Musical or Comedy", 
                       "Best Performance by an Actor in a Motion Picture - Drama"]
    
    Note:
        - Do NOT change the name of this function or what it returns
        - Award names should be extracted from tweets, not hardcoded
        - The only hardcoded part allowed is the word "Best"
    '''
    try:
        with open("learned_awards.json", "r") as f:
            learned_awards = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError("learned_awards.json not found.")
    except json.JSONDecodeError:
        raise ValueError("learned_awards.json is not a valid JSON file.")
    
    if not learned_awards:
        raise ValueError("No award data found in learned_awards.json.")

    # Sort awards by total_count (highest first)
    learned_awards.sort(key=lambda a: a.get("total_count", 0), reverse=True)

    # Filter: include top 30 or those with count > 25 #Arbitrary numbers 26 awards would remove some close dupes
    filtered_awards = []
    for i, award_entry in enumerate(learned_awards):
        total_count = award_entry.get("total_count", 0)
        if i < 35 or total_count > 25:
            # Use top variant text if available, else fallback to normalized
            if "top_variants" in award_entry and award_entry["top_variants"]:
                best_variant = award_entry["top_variants"][0]["text"]
            else:
                best_variant = award_entry.get("normalized", "")
            
            # Clean and format the award name
            formatted_award = best_variant.replace("-", " ").strip().title()
            filtered_awards.append(formatted_award)
        else:
            # Stop early if all remaining awards have low counts
            if i >= 35 or total_count <= 25:
                break

    return filtered_awards

def get_nominees(year):
    '''Returns the nominees for each award category.
    
    Args:
        year (str): The year of the Golden Globes ceremony (e.g., "2013")
    
    Returns:
        dict: A dictionary where keys are award category names and values are 
              lists of nominee strings.
              Example: {
                  "Best Motion Picture - Drama": [
                      "Three Billboards Outside Ebbing, Missouri",
                      "Call Me by Your Name", 
                      "Dunkirk",
                      "The Post",
                      "The Shape of Water"
                  ],
                  "Best Motion Picture - Musical or Comedy": [
                      "Lady Bird",
                      "The Disaster Artist",
                      "Get Out",
                      "The Greatest Showman",
                      "I, Tonya"
                  ]
              }
    
    Note:
        - Do NOT change the name of this function or what it returns
        - Use the hardcoded award names as keys (from the global AWARD_NAMES list)
        - Each value should be a list of strings, even if there's only one nominee
    '''
    nominees = {}
    if not data or "award_data" not in data:
        raise ValueError(f"Data for year {year} not found or 'award_data' key is missing.")
    for award, details in data["award_data"].items():
        # Award categories as keys, presenters as the proxy for nominees
        nominees[award.replace("-", " ").title()] = details["presenters"]
    return nominees

def get_winner(year):
    '''Returns the winner for each award category.
    
    Args:
        year (str): The year of the Golden Globes ceremony (e.g., "2013")
    
    Returns:
        dict: A dictionary where keys are award category names and values are 
              single winner strings.
              Example: {
                  "Best Motion Picture - Drama": "Three Billboards Outside Ebbing, Missouri",
                  "Best Motion Picture - Musical or Comedy": "Lady Bird",
                  "Best Performance by an Actor in a Motion Picture - Drama": "Gary Oldman"
              }
    
    Note:
        - Do NOT change the name of this function or what it returns
        - Use the hardcoded award names as keys (from the global AWARD_NAMES list)
        - Each value should be a single string (the winner's name)
    '''
    winners = {}
    if not data or "award_data" not in data:
        raise ValueError(f"Data for year {year} not found or 'award_data' key is missing.")
    for award, details in data["award_data"].items():
        # Extract the winner for each award
        winners[award.replace("-", " ").title()] = details["winner"]
    return winners

def get_presenters(year):
    '''Returns the presenters for each award category.
    
    Args:
        year (str): The year of the Golden Globes ceremony (e.g., "2013")
    
    Returns:
        dict: A dictionary where keys are award category names and values are 
              lists of presenter strings.
              Example: {
                  "Best Motion Picture - Drama": ["Barbra Streisand"],
                  "Best Motion Picture - Musical or Comedy": ["Alicia Vikander", "Michael Keaton"],
                  "Best Performance by an Actor in a Motion Picture - Drama": ["Emma Stone"]
              }
    
    Note:
        - Do NOT change the name of this function or what it returns
        - Use the hardcoded award names as keys (from the global AWARD_NAMES list)
        - Each value should be a list of strings, even if there's only one presenter
    '''
    # Your code here
    presenters = {}
    if not data or "award_data" not in data:
        raise ValueError(f"Data for year {year} not found or 'award_data' key is missing.")
    for award, details in data["award_data"].items():
        # Award categories as keys, presenters as values
        presenters[award.replace("-", " ").title()] = details["presenters"]
    return presenters

def load_texts(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit("Input JSON must be a list")
    for item in data:
        yield (item.get("text","") if isinstance(item, dict) else str(item))

def load_nlp_model():
    """Try to load your trained NER model, fallback to spaCy default."""
    model_path = Path("output/model-best")
    if model_path.exists():
        print("Using trained spaCy model (output/model-best)")
        return spacy.load(model_path)
    else:
        print("Using default model (en_core_web_sm)")
        return spacy.load("en_core_web_sm")


def pre_ceremony():
    '''Pre-processes and loads data for the Golden Globes analysis.
    
    This function should be called before any other functions to:
    - Load and process the tweet data from gg2013.json
    - Download required models (e.g., spaCy models)
    - Perform any initial data cleaning or preprocessing
    - Store processed data in files or database for later use
    
    This is the first function the TA will run when grading.
    
    Note:
        - Do NOT change the name of this function or what it returns
        - This function should handle all one-time setup tasks
        - Print progress messages to help with debugging
    '''
    global nlp
    nlp = load_nlp_model()

    texts = list(load_texts(INPUT))
    out = []

    print(f"Processing {len(texts)} texts...")

    # --- Run all batch extractors ---
    # extract_red_carpet_batch(texts)
    # extract_performance_info_batch(texts)

    # --- Candidate extraction loop (if needed) ---
    for text in tqdm(texts, desc="Generating candidates"):
        for c in generate_from_text(text, {}, "raw", 8, 2):
            out.append(asdict(c))

    # --- Aggregate and save everything ---
    # aggregate_red_carpets()
    # aggregate_performances()
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    dump_learned_awards("learned_awards.json")

    print(f"Wrote {len(out)} candidates to {OUT}")
    print("Pre-ceremony processing complete.")
    return

def main():
    '''Main function that orchestrates the Golden Globes analysis.
    
    This function should:
    - Call pre_ceremony() to set up the environment
    - Run the main analysis pipeline
    - Generate and save results in the required JSON format
    - Print progress messages and final results
    
    Usage:
        - Command line: python gg_api.py
        - Python interpreter: import gg_api; gg_api.main()
    
    This is the second function the TA will run when grading.
    
    Note:
        - Do NOT change the name of this function or what it returns
        - This function should coordinate all the analysis steps
        - Make sure to handle errors gracefully
    '''
    pre_ceremony()
    try:
        with open(f"results.json", "r") as file:
            global data
            data = json.load(file)
        print(f"Data for {YEAR} loaded successfully.")
    except FileNotFoundError:
        print(f"Error: The file {YEAR}_results.json was not found.")
        return
    except json.JSONDecodeError:
        print("Error: Failed to decode the JSON file.")
        return
    
    print(get_hosts(YEAR))
    print(get_awards(YEAR))
    print(get_nominees(YEAR))
    print(get_winner(YEAR))
    print(get_presenters(YEAR))

    return

if __name__ == '__main__':
    main()