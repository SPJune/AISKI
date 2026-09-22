import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load files
voice_info = pd.read_csv("data/generated/voice_info.csv")  # name, voice_id, gender, velocity, C
pairwise = pd.read_csv("data/generated/pairwise_distances.csv")  # name1, name2, distance

# name -> gender map
gender_map = dict(zip(voice_info["name"], voice_info["gender"]))

# Pair classifier by gender combination
def classify_pair(row):
    g1 = gender_map.get(row["name1"], None)
    g2 = gender_map.get(row["name2"], None)
    if g1 is None or g2 is None:
        return None
    if g1 == "M" and g2 == "M":
        return "MM"
    elif g1 == "F" and g2 == "F":
        return "FF"
    else:
        return "MF"

pairwise["group"] = pairwise.apply(classify_pair, axis=1)

# Save each group CSV
for group in ["MM", "FF", "MF"]:
    df_group = pairwise[pairwise["group"] == group]
    df_group.to_csv(f"data/generated/{group}_distance.csv", index=False)

# threshold sweep
thresholds = np.arange(1.4, 2.5 + 0.001, 0.05)
counts = {"MM": [], "FF": [], "MF": [], "Total": []}
percents = {"MM": [], "FF": [], "MF": []}
unique_names_total = []
unique_names_MM = []
unique_names_FF = []
unique_names_MF = []

for th in thresholds:
    df_th = pairwise[pairwise["distance"] > th]
    n_MM = (df_th["group"] == "MM").sum()
    n_FF = (df_th["group"] == "FF").sum()
    n_MF = (df_th["group"] == "MF").sum()
    total = n_MM + n_FF + n_MF
    
    counts["MM"].append(n_MM)
    counts["FF"].append(n_FF)
    counts["MF"].append(n_MF)
    counts["Total"].append(total)
    
    percents["MM"].append(100 * n_MM / total if total > 0 else 0)
    percents["FF"].append(100 * n_FF / total if total > 0 else 0)
    percents["MF"].append(100 * n_MF / total if total > 0 else 0)
    
    # Calculate unique name counts.
    names_total = set(df_th["name1"]).union(set(df_th["name2"]))
    names_MM = set(df_th.loc[df_th["group"]=="MM", "name1"]).union(
                set(df_th.loc[df_th["group"]=="MM", "name2"]))
    names_FF = set(df_th.loc[df_th["group"]=="FF", "name1"]).union(
                set(df_th.loc[df_th["group"]=="FF", "name2"]))
    names_MF = set(df_th.loc[df_th["group"]=="MF", "name1"]).union(
                set(df_th.loc[df_th["group"]=="MF", "name2"]))
    
    unique_names_total.append(len(names_total))
    unique_names_MM.append(len(names_MM))
    unique_names_FF.append(len(names_FF))
    unique_names_MF.append(len(names_MF))

    print(f"Threshold={th:.2f} | "
          f"MM={n_MM} ({percents['MM'][-1]:.2f}%, UniqueNames={len(names_MM)}) | "
          f"FF={n_FF} ({percents['FF'][-1]:.2f}%, UniqueNames={len(names_FF)}) | "
          f"MF={n_MF} ({percents['MF'][-1]:.2f}%, UniqueNames={len(names_MF)}) | "
          f"Total={total}, UniqueNames_Total={len(names_total)}")

# --- Plot graphs ---

# 1) Count line plot
plt.figure(figsize=(10,6))
plt.plot(thresholds, counts["MM"], label="MM")
plt.plot(thresholds, counts["FF"], label="FF")
plt.plot(thresholds, counts["MF"], label="MF")
plt.plot(thresholds, counts["Total"], label="Total", linestyle="--", color="black")
plt.xlabel("Threshold")
plt.ylabel("Count")
plt.title("Counts above threshold")
plt.legend()
plt.grid(True)
plt.savefig("outputs/plots/distance_counts.png", dpi=300)

# 2) Percentage stacked bar
plt.figure(figsize=(10,6))
bar_width = 0.035
plt.bar(thresholds, percents["MM"], width=bar_width, label="MM %")
plt.bar(thresholds, percents["FF"], width=bar_width, bottom=percents["MM"], label="FF %")
bottom_MF = np.array(percents["MM"]) + np.array(percents["FF"])
plt.bar(thresholds, percents["MF"], width=bar_width, bottom=bottom_MF, label="MF %")
plt.xlabel("Threshold")
plt.ylabel("Percentage (%)")
plt.title("Percentage distribution above threshold (stacked)")
plt.legend()
plt.grid(True, axis="y")
plt.savefig("outputs/plots/distance_percentages_stacked.png", dpi=300)

# 3) Unique name count line plot
plt.figure(figsize=(10,6))
plt.plot(thresholds, unique_names_total, label="Total", linestyle="--", color="black")
plt.plot(thresholds, unique_names_MM, label="MM")
plt.plot(thresholds, unique_names_FF, label="FF")
plt.plot(thresholds, unique_names_MF, label="MF")
plt.xlabel("Threshold")
plt.ylabel("Unique Name Count")
plt.title("Unique names involved in pairs above threshold")
plt.legend()
plt.grid(True)
plt.savefig("outputs/plots/unique_names_by_group.png", dpi=300)

plt.close("all")
