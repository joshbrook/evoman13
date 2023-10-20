import numpy as np
import matplotlib.pyplot as plt

l1 = "all"
l2 = "split"
runs = 10
gens = 61

#################### FUNCTION ####################


def import_data(level):
    fit = []
    for o in range(1, runs + 1):
        with open("output/best_f_level" + str(level) + "_try" + str(o)) as f:
            best_f = f.readlines()

        for i in range(len(best_f)):
            best_f[i] = best_f[i].split(",")
            for j in range(len(best_f[i])):
                best_f[i][j] = best_f[i][j].strip()
                best_f[i][j] = float(best_f[i][j])

        fit.append(best_f)


    avg_best_fit = []
    for j in range(len(fit[0])):
        sum_fit = 0
        for e in range(len(fit)):
            sum_fit += fit[e][j][0]
        avg_best_fit.append(sum_fit / len(fit))
        if len(avg_best_fit) == gens:
            break


    mean_fit = []
    for o in range(1, runs + 1):
        with open("output/mean_f_level" + str(level) + "_try" + str(o)) as f:
            mean_f = f.readlines()

        for i in range(len(mean_f)):
            mean_f[i] = mean_f[i].split(",")
            for j in range(len(mean_f[i])):
                mean_f[i][j] = mean_f[i][j].strip()
                mean_f[i][j] = float(mean_f[i][j])

        mean_fit.append(mean_f)


    avg_mean_fit = []
    std_fit = []
    for j in range(len(mean_fit[0])):
        values_std = []
        sum_fit = 0

        for e in range(len(mean_fit)):
            sum_fit += mean_fit[e][j][0]
            values_std.append(mean_fit[e][j][0])

        std_fit.append(np.std(values_std))
        avg_mean_fit.append(sum_fit / len(mean_fit))

        if len(avg_mean_fit) == gens:
            break

    return avg_best_fit, avg_mean_fit, std_fit


#################### CMA ####################

avg_best_fit_f1, avg_mean_fit_f1, std_fit_f1 = import_data(l1)
avg_best_fit_f2, avg_mean_fit_f2, std_fit_f2 = import_data(l2)
print(len(avg_best_fit_f1))
print(len(avg_best_fit_f2))

avg_best_fit_f1 = avg_best_fit_f1[1:]
avg_mean_fit_f1 = avg_mean_fit_f1[1:]
std_fit_f1 = std_fit_f1[1:]

print(len(avg_best_fit_f1))

avg_best_fit_f2 = avg_best_fit_f2[:60]
avg_mean_fit_f2 = avg_mean_fit_f2[:60]
std_fit_f2 = std_fit_f2[:60]

print(len(avg_best_fit_f2))

std_f1_plus = np.add(np.array(avg_mean_fit_f1), np.array(std_fit_f1))
std_f1_minus = np.subtract(np.array(avg_mean_fit_f1), np.array(std_fit_f1))
std_f2_plus = np.add(np.array(avg_mean_fit_f2), np.array(std_fit_f2))
std_f2_minus = np.subtract(np.array(avg_mean_fit_f2), np.array(std_fit_f2))

#################### PLOTTING ####################


line1 = plt.plot(avg_best_fit_f1, label="Best Fitness ALL")
line2 = plt.plot(avg_mean_fit_f1, label="Mean Fitness ALL", color="#15a9e8")
line3 = plt.plot(avg_best_fit_f2, label="Best Fitness SPLIT")
line4 = plt.plot(avg_mean_fit_f2, label="Mean Fitness SPLIT", color="#f2b422")
line5 = plt.plot(std_f1_plus, "--", linewidth=0.3, color="#15a9e8")
line6 = plt.plot(std_f1_minus, "--", linewidth=0.3, color="#15a9e8")
line7 = plt.plot(std_f2_plus, "--", linewidth=0.3, color="#f2b422")
line8 = plt.plot(std_f2_minus, "--", linewidth=0.3, color="#f2b422")


np_gens = np.arange(0, gens-1, 1)
plt.fill_between(np_gens, std_f1_plus, std_f1_minus, alpha=0.3)
plt.fill_between(np_gens, std_f2_plus, std_f2_minus, alpha=0.3)


plt.legend(prop={"family":"serif"}, loc='lower right')
plt.xlabel("Generations", fontfamily="serif", fontsize=14)
plt.ylabel("Fitness Value", fontfamily="serif", fontsize=14)
# plt.yticks(range(0, 101, 20))
plt.title("Fig. 2: Fitness vs. Generation", fontsize=20, pad=14, fontfamily="serif")
plt.savefig("plots/plot.png")
plt.show()


import pandas as pd
import statsmodels.api as sm 
from statsmodels.formula.api import ols 

from scipy.stats import f_oneway

print(f_oneway(avg_best_fit_f1, avg_best_fit_f2))