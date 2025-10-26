import json

def parse_candidates():
    final = {}
    with open("candidates.json", "r") as file:
        data = json.load(file)
        for d in data:
            award = d["award_name"]
            rule = d["rule_id"]
            subject = d["subject"]
            if award in final:
                if d["rule_id"] in final[award]:
                    if subject in final[award][rule]:
                        final[award][rule][subject] += 1
                    else:
                        final[award][rule][subject] = 1
                else:
                    final[award][rule] = {subject: 1}
            else:
                final[award] = {rule: {subject: 1}}
    
    with open("frequencies.json", "w") as json_file:
        json.dump(final, json_file, indent=4)

    return final

def get_data():
    parsed = parse_candidates()
    final = {}
    hosts = []
    for k in parsed:
        print(k)
        if k == "": # is the host category 
            sorted_items = sorted(parsed[""]["HOST"].items(), key=lambda item: item[1], reverse=True)

            # Extract the keys of the top 2 highest values
            hosts = [item[0] for item in sorted_items[:2]]

        else:
            # get winner
            if "WIN_A" in parsed[k]:
                win_a = parsed[k]["WIN_A"].items()
            else:
                win_a = []

            if "WIN_B" in parsed[k]:
                win_b = parsed[k]["WIN_B"].items()
            else:
                win_b = []

            wins = list(win_a) + list(win_b)
            winner = max(wins, key=lambda item: item[1])[0]

            # get presenters
            if "PRESENT_A" in parsed[k]:
                present_a = list(parsed[k]["PRESENT_A"].keys())
            else:
                present_a = []
            if "PRESENT_B" in parsed[k]:
                present_b = list(parsed[k]["PRESENT_B"].keys())
            else:
                present_b = []
            presenters = list(set(present_a + present_b))

            final[k] = {
                "presenters": presenters,
                "winner": winner
            }
        

    data = {
        "hosts": hosts,
        "award_data": final
    }

    with open("results.json", "w") as json_file:
        json.dump(data, json_file, indent=4)


get_data()