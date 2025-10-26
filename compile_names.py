import json


def get_freq_winners():
    dic = {}
    try:
        with open("candidates.json", "r") as file:
            data = json.load(file)
            for ob in data:
                if ob['award_name'] in dic:
                    if ob['subject'] in dic[ob['award_name']]:
                        dic[ob['award_name']][ob['subject']] += 1
                    else:
                        dic[ob['award_name']][ob['subject']] = 1
                else:
                    dic[ob['award_name']] = {ob['subject']: 1}
    except FileNotFoundError:
        print("Error: The file 'candidates.json' was not found.")

get_freq_winners()