import json

_data = []
def parse_data_file():
    try:
        # Open the JSON file in read mode ('r')
        with open('gg2013.json', 'r') as file:
            data = json.load(file)
            _data = [item["text"] for item in data]

    except FileNotFoundError:
        print("Error: The file 'data.json' was not found.")
    except json.JSONDecodeError:
        print("Error: Failed to decode JSON from the file. The file might be malformed.")

parse_data_file()