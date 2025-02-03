import matplotlib.pyplot as plt
import numpy as np

# Create a 10x10 grid of data points
data_points = np.arange(100).reshape(10, 10)

# Create color maps for 10 groups
colors = plt.cm.tab10(np.arange(10))

# Create a figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Function to set up each subplot
def setup_axes(ax, colors_grid, title):
    ax.imshow(colors_grid, interpolation='none', extent=[-0.5, 9.5, -0.5, 9.5])

    # Set labels and ticks
    ax.set_title(title)
    ax.set_xlabel('Ordering Permutation')
    ax.set_ylabel('Example Set')

    # Set major ticks at the center of each cell
    ax.set_xticks(np.arange(0, 10, 1))
    ax.set_xticklabels(range(1, 11))
    ax.set_yticks(np.arange(0, 10, 1))
    ax.set_yticklabels(range(10, 0, -1))  # Invert y-axis to match matrix representation

    # Set minor ticks at cell boundaries
    ax.set_xticks(np.arange(-0.5, 10, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 10, 1), minor=True)

    # Draw grid lines at minor ticks
    ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)

    # Turn off grid lines at major ticks
    ax.grid(which='major', color='none')

# Experiment 1: Grouped by example sets (rows)
colors_grid_exp1 = np.zeros((10, 10, 4))  # RGBA colors
for i in range(10):
    for j in range(10):
        colors_grid_exp1[i, j, :] = colors[i]  # Color by row (example set)

setup_axes(axes[0], colors_grid_exp1, 'Experiment 1: Grouped by Example Sets')

# Experiment 2: Grouped by ordering permutations (columns)
colors_grid_exp2 = np.zeros((10, 10, 4))  # RGBA colors
for i in range(10):
    for j in range(10):
        colors_grid_exp2[i, j, :] = colors[j]  # Color by column (ordering permutation)

setup_axes(axes[1], colors_grid_exp2, 'Experiment 2: Grouped by Ordering Permutations')

plt.tight_layout()
plt.savefig('visualization.png', dpi=300)
