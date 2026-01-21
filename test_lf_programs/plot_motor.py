import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("motor.csv")
plt.style.use('seaborn-v0_8-whitegrid')
#change according to width
plt.figure(figsize=(15, 6))

for name, group in df.groupby("Group"):
    actual_rpm = group["Actual RPM"].iloc[0]
    
    if actual_rpm == 80:
        power = 0.1
    elif actual_rpm == 60:
        power = 0.08
    elif actual_rpm == 33:
        power = 0.05
    else:
        power = 0
    
    plt.plot(
        group["Time"],
        group["Measured RPM"],
        marker='o',
        markersize=4,
        linewidth=1.5, 
        label=f"Actual RPM {actual_rpm:.0f}, Duty Cycle {power:.2f}"
    )
    
    plt.axhline(
        y=actual_rpm,
        color='gray',
        linestyle=':',
        linewidth=1.2,
        alpha=0.8
    )

plt.tick_params(axis='x', labelsize=28)
plt.tick_params(axis='y', labelsize=28)
#CHange for width
plt.xticks(np.arange(0, 61, 5))
plt.xlabel("Total Testing Time Elapsed (s)", fontsize=34)
plt.ylabel("Measured RPM", fontsize=34)
plt.legend(fontsize=24)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("rpm_graph.svg")
plt.savefig("rpm_graph.jpg")
#plt.show()