import json

def parse_candidates():
    final = {}
    with open("candidates.json", "r", encoding="utf-8") as file:
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
    candidates_data = {}
    hosts = []
    for k in parsed:
        if k == "": # is the host category 
            sorted_hosts = sorted(parsed[""]["HOST"].items(), key=lambda item: item[1], reverse=True)

            # Extract the keys of the top 2 highest values
            hosts = [item[0] for item in sorted_hosts[:2]]
            host_candidates = [item[0] for item in sorted_hosts[:5]]

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
            winner_candidates = [w[0] for w in sorted(wins, key=lambda x: x[1], reverse=True)[:3]]

            # get presenters
            presenters_freq = {}
            if "PRESENT_A" in parsed[k]:
                presenters_freq.update(parsed[k]["PRESENT_A"])
            if "PRESENT_B" in parsed[k]:
                presenters_freq.update(parsed[k]["PRESENT_B"])

            # Get top 2 presenters
            presenter_names = sorted(presenters_freq, key=lambda x: presenters_freq[x], reverse=True)[:2]

            # Get top 5 presenter candidates
            presenter_candidates = sorted(presenters_freq, key=lambda x: presenters_freq[x], reverse=True)[:5]

            # get nominees
            nom_b = list(parsed[k].get("NOMINEES_B", {}).items()) if "NOMINEES_B" in parsed[k] else []
            nominees_all = nom_b + wins
            nominee_candidates = [n[0] for n in sorted(nominees_all, key=lambda x: x[1], reverse=True)[:5]]


            final[k] = {
                "presenters": presenter_names,
                "nominees": nominee_candidates,
                "winner": winner
            }

            candidates_data[k] = {
                "presenters": presenter_names,
                "presenter_candidates": presenter_candidates,
                "winner": winner,
                "winner_candidates": winner_candidates,
                "nominees": nominee_candidates,
                "nominee_candidates": nominee_candidates
            }
        

    data = {
        "hosts": hosts,
        "award_data": final
    }

    with open("results.json", "w") as json_file:
        json.dump(data, json_file, indent=4)

    results_candidates = {
        "hosts": {
            "confirmed": hosts,
            "candidates": host_candidates
        },
        "awards": candidates_data
    }

    with open("results_candidates.json", "w", encoding="utf-8") as json_file:
        json.dump(results_candidates, json_file, indent=4)


get_data()