import csv
import networkx as nx
import matplotlib.pyplot as plt

transition_matrix = {}
with open("final_transition_matrix.csv", "r", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        source = row["current_state"].strip()
        target = row["next_state"].strip()
        prob = float(row["prob"])
        if source not in transition_matrix:
            transition_matrix[source] = {}
        transition_matrix[source][target] = prob

G = nx.DiGraph()
for source, targets in transition_matrix.items():
    for target, prob in targets.items():
        if prob > 0.0:
            G.add_edge(source, target, weight=prob)

plt.figure(figsize=(10, 8), dpi=100)

pos = nx.spring_layout(G, seed=42, k=0.3)

nx.draw_networkx_nodes(
    G,
    pos,
    node_size=1200,
    node_color="lightblue",
    edgecolors="gray",
)
nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=15, width=1.0)
nx.draw_networkx_labels(G, pos, font_size=9, font_family="sans-serif")

edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

plt.title("Markov Chain", fontsize=10, pad=10)
plt.axis("off")

plt.tight_layout()

plt.savefig("markov_chain_simple.png", dpi=150)
plt.show()
