import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


no_baseline = pd.read_csv("results/reinforce_without_baseline.csv")
baseline = pd.read_csv("results/reinforce_with_baseline.csv")

sns.set(style="dark", context="notebook", palette="rainbow")
sns.lineplot(x="Episodes", y="Rewards", label="No Baseline", data=no_baseline)
sns.lineplot(x="Episodes", y="Rewards", label="Baseline", data=baseline).set(title="REINFORCE with & w/o Baseline")
plt.savefig("figures/reinforce_comparison.png")