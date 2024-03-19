import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='plot')
parser.add_argument('--log', type=str, default='exp', help='name of the experiment')
parser.add_argument('--out', type=str, default='ratio.png', help='name of the experiment')
args = parser.parse_args()

# Step 1: Read the data from log.txt
with open(args.log, "r") as file:
    lines = file.readlines()

# Extract a, b, c, d values for each line
data = [list(map(float, line.strip().split())) for line in lines]

# Step 2: Compute values
x = np.arange(len(data))
avg_d = np.array([d[0] for d in data])
std_d = np.array([d[1] for d in data])
eq_ratio = np.array([d[2] for d in data])

fig = plt.figure()
ax = fig.add_subplot(111)
# Step 3: Plot each value
ax.plot(x*5+5, avg_d, color='k', linestyle='--')  # Optional, if you want to see the lines
ax.fill_between(x*5+5, avg_d-std_d, avg_d+std_d, color='blue', alpha=0.5)
# plt.fill_between(x*5+5, abc_vals, 1, color='green', alpha=0.7)

# # Annotations
# mid_x = x[len(x) // 2]  # Choose the middle x value for annotations
# annotations = [
#     (0.5 * a_vals[3], "Invalid"),
#     (0.5 * (a_vals[mid_x] + ab_vals[mid_x]), "Valid Unbalance"),
#     (0.5 * (ab_vals[mid_x] + abc_vals[mid_x]), "In Dataset"),
#     (0.5 * (1 + abc_vals[mid_x]), "Extrapolation")
# ]
# for y, label in annotations:
#     if label == 'Invalid':
#         plt.annotate(label, (x[3] * 5 + 0.5, y), ha="center", va="center", fontsize=10)
#     else:
#         plt.annotate(label, (mid_x*5+0.5, y), ha="center", va="center", fontsize=10)
#
ax2 = ax.twinx()
ax2.plot(x*5+5, eq_ratio, color='r', linestyle='--')  # Optional
ax.grid(True)

ax.set_xlabel('k iter')
ax.set_ylabel('Avg Edit Distance')
ax2.set_ylabel('Ratio of Equal')
plt.savefig(args.out, format="png", dpi=600)

