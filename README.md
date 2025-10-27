# CS337-Project-1
CS 337 Project 1 Repository

## Pre-running

### Imports
All imports can be added using `requirements.txt`. The easiest way to install them is using a virtual environment (`venv` folder).

To import run `pip install -r requirements.txt`. Then run `python -m spacy download en_core_web_sm` and python -m spacy download en_core_web_lg`.

### Train Model
Runtime: 10 minutes
We needed to train a spacy model beforehand in order to accurately recognize entities (such as getting nominees and award winners). To train, run `python -m spacy train config.cfg --output ./output --paths.train ./train.spacy --paths.dev ./dev.spacy` once all the installation has been completed.

## Running
Normal Functionality Runtime: 2-3 minutes
For A level features uncomment:
	extract_red_carpet_batch(texts)
	extract_performance_info_batch(texts)
	aggregate_red_carpets()
	aggregate_performances()
in pre_ceremony in `gg_api.py`. Note this increases runtime to 18-20 minutes

All of our functions can be run using `gg_api.py`. This should populate json files: `candidates.json`, `performance.json, `red_carpet.json`, and `results.json`.
- `candidates.json`: includes all the candidates (awards along with what the role/subject is- winners/nominees/hosts/etc.)
- `performance.json`: includes performers in the show, as well as general opinions on the performers
- `red_carpet.json`: includes best dressed, worst dressed, most discussed, most controversial during the show
- `results.json`: includes the final results along with winners, nominees, and presenters for each award, and hosts of the show
