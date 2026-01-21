#!/usr/bin/env python3
"""
ACT for blinking -> test inside a Nix develop shell.
Hardcoded for Blink.lf.
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import csv
import os

os.environ["QT_QPA_PLAPTFORM"] = "offscreen"

def plot():
    #ACT_HOME = Path.home()/"pololu"
    
    df = pd.read_csv("blink_results.csv")

    plt.style.use('seaborn-v0_8-whitegrid'
                  )
    df["Deviation (%)"] = (df["Deviation: (s)"].abs() / df["Actual Period (s)"]) * 100

    # Plot: Total Time vs Deviation %
    plt.figure(figsize=(8,6))
    for actual_period, group in df.groupby("Actual Period (s)"):
        plt.plot(
            group["Total Time (s)"],
            group["Deviation (%)"],
            marker='s',
            linewidth=2,
            label=f"Actual Period : {actual_period:.1f}s"
        )

    plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
    plt.tick_params(axis='x', labelsize=20)
    plt.tick_params(axis='y', labelsize=20)
    #plt.title("Deviation (%) v Total Testing Time", fontsize=24)
    plt.xlabel("Total Testing Time Elapsed (s)", fontsize=24)
    plt.ylabel("Deviation (%)", fontsize=24)
    plt.legend(title="", fontsize=24)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("time_vs_deviation_percent.svg")
    #plt.savefig(ACT_HOME/"time_vs_deviation_percent.jpg")
    #plt.show()
    return

def main():
    plot()
    
if __name__ == "__main__":
    main()

