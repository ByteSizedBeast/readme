import numpy as np

def initialize_pagerank(graph):
    num_nodes = len(graph)
    return {node: 1 / num_nodes for node in graph}

def compute_pagerank(graph, damping_factor=0.85, max_iterations=100, tolerance=1e-6):
    num_nodes = len(graph)
    pagerank = initialize_pagerank(graph)
    new_pagerank = pagerank.copy()

    for _ in range(max_iterations):
        for node in graph:
            new_pagerank[node] = (1 - damping_factor) / num_nodes + damping_factor * sum(
                pagerank[incoming] / len(graph[incoming]) for incoming in graph if node in graph[incoming]
            )

        if all(abs(new_pagerank[node] - pagerank[node]) < tolerance for node in pagerank):
            break
        pagerank = new_pagerank.copy()

    return pagerank

web_graph = {
    'A': {'B', 'C'},
    'B': {'C'},
    'C': {'A'},
    'D': {'C'}
}

pagerank_scores = compute_pagerank(web_graph)

print("\n=== PageRank Scores ===")
for page, score in sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True):
    print(f"Page {page}: {score:.6f}")