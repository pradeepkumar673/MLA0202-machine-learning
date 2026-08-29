import numpy as np

original_image = np.array([
    [1, 1, 1, 1],
    [1, 8, 1, 1],
    [1, 1, 9, 1],
    [1, 1, 1, 1]
])

rows, cols = original_image.shape
graph = {}

for i in range(rows):
    for j in range(cols):
        node = (i, j)
        neighbors = []
        if i > 0: neighbors.append((i - 1, j))
        if i < rows - 1: neighbors.append((i + 1, j))
        if j > 0: neighbors.append((i, j - 1))
        if j < cols - 1: neighbors.append((i, j + 1))
        graph[node] = neighbors

smoothed_image = np.zeros_like(original_image)

for node, neighbors in graph.items():
    i, j = node
    neighbor_values = [original_image[n] for n in neighbors]
    smoothed_image[i, j] = int(np.round(np.mean(neighbor_values)))

print("Original Image:\n", original_image)
print("\nSmoothed Image:\n", smoothed_image)