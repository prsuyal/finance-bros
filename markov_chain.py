import csv
from collections import defaultdict

states = ["nothing", "titfortat", "subretaliation", "superretaliation", "subsidization"]

ALPHA = 2.0  # adjustable weight for prior transition probs
BETA = 5.0  # boost for observed counts

prior = defaultdict(dict)
for s1 in states:
    for s2 in states:
        prior[s1][s2] = 0.0

with open("prior_transition_matrix.csv", "r", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        action = row["action"].strip().strip('"')
        response = row["response"].strip().strip('"')
        prob = float(row["prob"])
        prior[action][response] = prob

counts = defaultdict(lambda: defaultdict(int))

with open("historical_observations.csv", "r", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        action = row["action"].strip().strip('"')
        response = row["response"].strip().strip('"')
        c = float(row["count"])
        counts[action][response] += c

final_matrix = defaultdict(dict)

for s1 in states:
    row_sum = 0.0
    for s2 in states:
        pseudocount = ALPHA * prior[s1][s2]
        observedcount = BETA * counts[s1][s2]
        totalcount = pseudocount + observedcount
        final_matrix[s1][s2] = totalcount
        row_sum += totalcount

    if row_sum > 0:
        for s2 in states:
            final_matrix[s1][s2] /= row_sum  # normalize to make probabilities

    else:
        for s2 in states:
            final_matrix[s1][s2] = 1.0 / len(states) # uniform distribution fallback

with open("final_transition_matrix.csv", "w", newline="", encoding="utf-8") as f: 
    writer = csv.writer(f)
    writer.writerow(["current_state", "next_state", "prob"])
    for s1 in states:
        for s2 in states:
            writer.writerow([s1, s2, final_matrix[s1][s2]])

print("done!")
