import re
import matplotlib.pyplot as plt

with open("simulation_output.txt", "r") as file:
    lines = file.readlines()

labels = []
mean_sir = []
outage = []

for i, line in enumerate(lines):
    match = re.match(
        r"\s*(2\.5|3|4)\s+\|\s+(\d+)\s+\|\s+(\d+)\s+\|\s+(\d+)\s+\|\s+([\d.]+)\s+\|\s+(\d+)",
        line
    )

    if match and i + 1 < len(lines):
        result = re.search(
            r"Mean SIR:\s*([\d.]+)\s*dB.*Outage:\s*([\d.]+)%",
            lines[i + 1]
        )

        if result:
            n, k, cluster = match.group(1), match.group(2), match.group(3)
            labels.append(f"n={n}\nk={k}\nN={cluster}")
            mean_sir.append(float(result.group(1)))
            outage.append(float(result.group(2)))

if not labels:
    raise ValueError("No simulation results found in simulation_output.txt")

fig, axes = plt.subplots(2, 1, figsize=(11, 8))

axes[0].bar(labels, mean_sir)
axes[0].set_title("Mean SIR for Different Network Configurations")
axes[0].set_ylabel("Mean SIR (dB)")
axes[0].grid(axis="y", alpha=0.3)

axes[1].bar(labels, outage)
axes[1].set_title("Outage Percentage for Different Configurations")
axes[1].set_ylabel("Outage (%)")
axes[1].grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("performance_graph.png", dpi=300)
print("Graph saved successfully as performance_graph.png")
print("Configurations plotted:", len(labels))
