import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("all_results.csv")

for scene in df["Scene"].unique():
    sub = df[df["Scene"] == scene]

    plt.figure()
    plt.title(f"{scene} Trade-off")
    plt.scatter(sub["PLY Size (MB)"], sub["PSNR"])

    for _, row in sub.iterrows():
        plt.annotate(row["Method"], (row["PLY Size (MB)"], row["PSNR"]))

    plt.xlabel("Model Size (MB)")
    plt.ylabel("PSNR")
    plt.grid()
    plt.savefig(f"tradeoff_{scene}.png")

print("Figures saved.")