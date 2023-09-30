import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ind_gain_cma_level1 = []
ind_gain_cma_level3 = []
ind_gain_cma_level7 = []

ind_gain_ea_level1 = []
ind_gain_ea_level3 = []
ind_gain_ea_level7 = []


for i in range(1,11):
    with open('output/ind_gain_cma_level1_try'+str(i)) as f:
        ind_gain = f.readlines()
        ind_gain_cma_level1.append(float(ind_gain[0].strip()))

for i in range(1,11):
    with open('output/ind_gain_cma_level3_try'+str(i)) as f:
        ind_gain = f.readlines()
        ind_gain_cma_level3.append(float(ind_gain[0].strip()))


for i in range(1,11):
    with open('output/ind_gain_cma_level7_try'+str(i)) as f:
        ind_gain = f.readlines()
        ind_gain_cma_level7.append(float(ind_gain[0].strip()))



for i in range(1, 11):
    with open('output/ind_gain_own_ea_level1_try' + str(i)) as f:
        ind_gain = f.readlines()
        ind_gain_ea_level1.append(float(ind_gain[0].strip()))

for i in range(1, 11):
    with open('output/ind_gain_own_ea_level3_try' + str(i)) as f:
        ind_gain = f.readlines()
        ind_gain_ea_level3.append(float(ind_gain[0].strip()))

for i in range(1, 11):
    with open('output/ind_gain_own_ea_level7_try' + str(i)) as f:
        ind_gain = f.readlines()
        ind_gain_ea_level7.append(float(ind_gain[0].strip()))


print(ind_gain_cma_level1)
print(ind_gain_ea_level1)
print(ind_gain_cma_level3)
print(ind_gain_ea_level3)
print(ind_gain_cma_level7)
print(ind_gain_ea_level7)


data = {'Own EA': [ind_gain_ea_level1[0], ind_gain_ea_level1[1],ind_gain_ea_level1[2], ind_gain_ea_level1[3],ind_gain_ea_level1[4], ind_gain_ea_level1[5],ind_gain_ea_level1[6], ind_gain_ea_level1[7],ind_gain_ea_level1[8], ind_gain_ea_level1[9],ind_gain_ea_level3[0], ind_gain_ea_level3[1],ind_gain_ea_level3[2], ind_gain_ea_level3[3],ind_gain_ea_level3[4], ind_gain_ea_level3[5],ind_gain_ea_level3[6], ind_gain_ea_level3[7],ind_gain_ea_level3[8], ind_gain_ea_level3[9], ind_gain_ea_level7[0], ind_gain_ea_level7[1],ind_gain_ea_level7[2], ind_gain_ea_level7[3],ind_gain_ea_level7[4], ind_gain_ea_level7[5],ind_gain_ea_level7[6], ind_gain_ea_level7[7],ind_gain_ea_level7[8], ind_gain_ea_level7[9]],
        'CMA': [ind_gain_cma_level1[0], ind_gain_cma_level1[1],ind_gain_cma_level1[2], ind_gain_cma_level1[3],ind_gain_cma_level1[4], ind_gain_cma_level1[5],ind_gain_cma_level1[6], ind_gain_cma_level1[7],ind_gain_cma_level1[8], ind_gain_cma_level1[9],ind_gain_cma_level3[0], ind_gain_cma_level3[1],ind_gain_cma_level3[2], ind_gain_cma_level3[3],ind_gain_cma_level3[4], ind_gain_cma_level3[5],ind_gain_cma_level3[6], ind_gain_cma_level3[7],ind_gain_cma_level3[8], ind_gain_cma_level3[9], ind_gain_cma_level7[0], ind_gain_cma_level7[1],ind_gain_cma_level7[2], ind_gain_cma_level7[3],ind_gain_cma_level7[4], ind_gain_cma_level7[5],ind_gain_cma_level7[6], ind_gain_cma_level7[7],ind_gain_cma_level7[8], ind_gain_cma_level7[9]],
        'Level': ['Level_1', 'Level_1','Level_1', 'Level_1','Level_1', 'Level_1','Level_1', 'Level_1','Level_1', 'Level_1', 'Level_3','Level_3','Level_3','Level_3','Level_3','Level_3','Level_3','Level_3','Level_3','Level_3', 'Level_7','Level_7', 'Level_7','Level_7','Level_7','Level_7','Level_7','Level_7','Level_7','Level_7',]}

df = pd.DataFrame(data)

print(df.head())
dfl = pd.melt(df, id_vars='Level', value_vars=['Own EA', "CMA"])

print(dfl)
ax = sns.boxplot(x='Level', y='value', data=dfl, showfliers=True, hue='variable')
ax.set_xlabel("Level", fontdict = {'family': "serif", "size": 12})
ax.set_ylabel("Value", fontdict = {'family': "serif", "size": 12})
plt.xticks(fontsize=10, fontfamily="serif")
plt.yticks(fontsize=10, fontfamily="serif")
plt.legend(loc='lower right', prop={"family":"serif"})
plt.title("Individual Gain by Level", fontfamily="serif",  fontsize=16, pad=10) 
plt.savefig('output/boxplot1.png')
plt.show()