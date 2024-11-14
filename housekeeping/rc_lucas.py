import pandas as pd
import matplotlib.pyplot as plt


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def get_bands(cols):
    cs = []
    for c in cols:
        if is_number(c):
            cs.append(c)
    return cs

data = pd.read_csv(r"../data_raw/lucas_r.csv")
print(data["oc"].min())
print(data["oc"].max())
bands = get_bands(data.columns)
f_bands = list(range(len(bands)))
cols = ["oc"] + bands
data = data[cols]

low = 0
print(len(data))
for i in range(6):
    high = low + 10
    if i == 5:
        high = 100000
    lab = f"{low} ≤ SOC < {high}"
    if i == 0:
        lab = f"SOC < {high}"
    if i == 5:
        lab = f"{low} ≤ SOC"
    data2 = data[(data['oc']>=low)&(data['oc']<high)]
    mn = data2[bands].mean()
    y = mn.to_numpy()
    plt.plot(f_bands,y,label=lab)
    print(low, high, len(data2))
    low = high

plt.xlabel("Wavelength number", fontsize=15)
plt.ylabel("Reflectance (Normalized)", fontsize=15)
plt.legend(fontsize=13)
plt.margins(0.01)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig("lucas.png")