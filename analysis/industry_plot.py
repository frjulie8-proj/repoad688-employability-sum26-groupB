import pandas as pd
import matplotlib.pyplot as plt

industry = pd.read_csv(
    "data/processed/industry_codes.csv"
)

plt.figure(figsize=(10,6))

industry["IND"].hist(
    bins=20
)

plt.title("Distribution of Industry Codes")
plt.xlabel("Industry Code")
plt.ylabel("Frequency")

plt.savefig(
    "assets/figures/alina_industry_distribution.png",
    bbox_inches="tight"
)

print("Done")
