import pandas as pd
import matplotlib.pyplot as plt

classes = """
Alfalfa
Corn-notill
Corn-mintill
Corn
Grass-pasture
Grass-trees
Grass-pasture-mowed
Hay-windrowed
Oats
Soybean-notill
Soybean-mintill
Soybean-clean
Wheat
Woods
Buildings-Grass-Trees-Drives
Stone-Steel-Towers
"""

cs = classes.split("\n")
css = []
for c in cs:
    if len(c.strip()) != 0:
        css.append(c)


def get_class(c):
    return css[c-1]

def get_idxs(i):
    if i < 103:
        return 400 + (i*9.5)
    if 103 <= i <= 149:
        i = i + 5
        return 400 + (i*9.5)
    if 150 <= i:
        i = i + 5 + 14
        return 400 + (i*9.5)


def get_Xs():
    bands = []
    for i in range(200):
        b = get_idxs(i)
        bands.append(b)
    return bands

xs = get_Xs()


df = pd.read_csv(r'../data_raw/indian_pines.csv')
ref = df.iloc[:, 0:-1]
v_max = ref.max().max()
v_min = ref.min().min()
df.iloc[:, 0:-1] = df.iloc[:, 0:-1] / (v_max - v_min)

df_aggregated = df.groupby('class').mean()
cols = df.columns
cols = [col for col in df.columns if col != "class"]
i_cols = list(range(len(cols)))

for row in df_aggregated.itertuples():
    val = row[0]
    if val == 0:
        continue
    ref = row[1:]
    plt.plot(i_cols, ref, label=get_class(int(val)))



plt.xlabel("Wavelength number", fontsize=15)
plt.ylabel("Reflectance (Normalized)", fontsize=15)
plt.legend(fontsize=11, ncol=1, loc='upper right', bbox_to_anchor=(1.6, 1))
plt.margins(0.01)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig("ip.png", bbox_inches='tight')
