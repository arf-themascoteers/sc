import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r'../data_raw/ghisaconus.csv')
df_aggregated = df.groupby('crop').mean()
#cols = df.columns
cols = [col for col in df.columns if col != "crop"]
i_cols = list(range(len(cols)))

for row in df_aggregated.itertuples():
    val = row[0]
    ref = row[1:]
    plt.plot(i_cols, ref, label=val.replace("_", " ").title())

plt.legend()

plt.xlabel("Wavelength number", fontsize=15)
plt.ylabel("Reflectance (%)", fontsize=15)
plt.legend(fontsize=13)
plt.margins(0.01)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig("ghisaconus.png")
