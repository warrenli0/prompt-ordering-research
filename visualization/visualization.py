import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Sample data generation (replace this with your actual data)
# Each example set has 10 permutations with random values
np.random.seed(42)  # For consistent results
data = [np.random.normal(loc=np.random.rand()*10, scale=1.0, size=10) for _ in range(10)]

# Calculate standard deviations for each example set
std_devs = [np.std(permutations, ddof=1) for permutations in data]

# Calculate the average grouped standard deviation
avg_std = np.mean(std_devs)

# Create a figure with two subplots: one for boxplots and one for standard deviations
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Subplot 1: Boxplots of the distributions for each example set
sns.boxplot(data=data, ax=axes[0], palette="pastel")
axes[0].set_title('Distribution of Values for Each Example Set', fontsize=14)
axes[0].set_xlabel('Example Set', fontsize=12)
axes[0].set_ylabel('Values', fontsize=12)
axes[0].tick_params(axis='both', which='major', labelsize=10)

# Subplot 2: Bar chart of standard deviations with average line
axes[1].bar(range(1, 11), std_devs, color="skyblue", edgecolor='black')
axes[1].axhline(y=avg_std, color='red', linestyle='--', label=f'Average Std Dev = {avg_std:.2f}')
axes[1].set_title('Standard Deviation for Each Example Set', fontsize=14)
axes[1].set_xlabel('Example Set', fontsize=12)
axes[1].set_ylabel('Standard Deviation', fontsize=12)
axes[1].set_xticks(range(1, 11))
axes[1].legend(fontsize=12)
axes[1].tick_params(axis='both', which='major', labelsize=10)

plt.tight_layout()

# Save the figure to a file
plt.savefig('average_grouped_std_dev.png', dpi=300)

# Optional: Show the plot interactively
# plt.show()
