import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


l1 = "all"
l2 = "split"
tries = 10


def calc_ind_gain(l):
    ind_gain = []
    for i in range(1, tries + 1):
        with open('output/ind_gain_level' + l + '_try' + str(i)) as f:
            ig = f.readlines()
            ind_gain.append(float(ig[0].strip()))
    return ind_gain


ind_gain_all = calc_ind_gain(l1)
ind_gain_split = calc_ind_gain(l2)
print(ind_gain_all)
print(ind_gain_split)

data = {'All': ind_gain_all,
        'Split': ind_gain_split,
        'Level': [''] * 10}


df = pd.DataFrame(data)

print(df.head())
dfl = pd.melt(df, id_vars='Level', value_vars=['All', "Split"])

print(dfl)
ax = sns.boxplot(x='Level', y='value', data=dfl, showfliers=True, hue='variable')
ax.set_xlabel("Level", fontdict = {'family': "serif", "size": 12})
ax.set_ylabel("Value", fontdict = {'family': "serif", "size": 12})
plt.xticks(fontsize=10, fontfamily="serif")
plt.yticks(fontsize=10, fontfamily="serif")
plt.legend(loc='lower right', prop={"family":"serif"})
plt.title("Fig. 3: Individual Gain by Level", fontfamily="serif",  fontsize=20, pad=14) 
plt.savefig('output/boxplot1.png')
plt.show()