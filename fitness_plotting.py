import numpy as np
import matplotlib.pyplot as plt


level = "all"  # input("Which level to plot: ")
runs = 1
gens = 60


# inverse fitness function for cma results
def min_to_max(x):
    f = (((x - 100) * 200) / 100) + 100
    return abs(f)


#################### OWN EA ####################

fitness_own_ea = []

for o in range(1, runs + 1):
    with open("output/best_f_level" + str(level) + "_try" + str(o)) as f:
        best_f = f.readlines()[:-1]

    for i in range(len(best_f)):
        best_f[i] = best_f[i].split(",")
        for j in range(len(best_f[i])):
            best_f[i][j] = best_f[i][j].strip()
            best_f[i][j] = float(best_f[i][j])

    fitness_own_ea.append(best_f)


avg_best_f_own_ea = []
for j in range(len(fitness_own_ea[0])):
    sum_fit = 0
    for e in range(len(fitness_own_ea)):
        sum_fit += fitness_own_ea[e][j][0]
    avg_best_f_own_ea.append(sum_fit / len(fitness_own_ea))
    if len(avg_best_f_own_ea) == gens:
        break


mean_fitnesses_own_ea = []

for o in range(1, runs + 1):
    with open("output/mean_f_level" + str(level) + "_try" + str(o)) as f:
        mean_f = f.readlines()[:-1]

    for i in range(len(mean_f)):
        mean_f[i] = mean_f[i].split(",")
        for j in range(len(mean_f[i])):
            mean_f[i][j] = mean_f[i][j].strip()
            mean_f[i][j] = float(mean_f[i][j])

    mean_fitnesses_own_ea.append(mean_f)


avg_mean_f_own_ea = []
std_own_ea = []

for j in range(len(mean_fitnesses_own_ea[0])):
    values_std = []
    sum_fit = 0

    for e in range(len(mean_fitnesses_own_ea)):
        sum_fit += mean_fitnesses_own_ea[e][j][0]
        values_std.append(mean_fitnesses_own_ea[e][j][0])

    std_own_ea.append(np.std(values_std))
    avg_mean_f_own_ea.append(sum_fit / len(mean_fitnesses_own_ea))

    if len(avg_mean_f_own_ea) == gens:
        break


#################### CMA ####################


fitness_cma = []
# LOOP OVER 10 RUNS
for o in range(1, runs + 1):
    # OPEN CORRESPONDING FOLDER OF BEST F
    with open("output/f_best_cma_level" + str(level) + "_try" + str(o)) as f:
        best_f = f.readlines()
    # LOOP OVER CURRENT FILE TRANSFORM STR --> FLOAT + MIN_TO_MAX
    for i in range(len(best_f)):
        best_f[i] = best_f[i].split(",")
        for j in range(len(best_f[i])):
            best_f[i][j] = best_f[i][j].strip()
            best_f[i][j] = float(best_f[i][j])
            best_f[i][j] = min_to_max(best_f[i][j])
    # APPEND FINAL FILE AS LIST
    fitness_cma.append(best_f)

avg_best_f_cma = []
# CALCULATE AVG
# LOOP OVER THE FIRST LIST

for j in range(len(fitness_cma[0])):
    sum_fit = 0
    # FOR EACH ELEMENT OF THE FIRST LIST
    # CALCULATE THE SUM --> ADD TO THE CURRENT ELEMENT ALL THE OTHER ELEMENT AT THE SAME INDEX
    # 1ST RUN INDEX 0 + 2ND RUN INDEX 0 + 3RD RUN INDEX 0....
    for e in range(len(fitness_cma)):
        sum_fit += fitness_cma[e][j][0]
    # CALCULATE AVG
    avg_best_f_cma.append(sum_fit / len(fitness_cma))
    if len(avg_best_f_cma) == gens:
        break


# import CMA mean fitness values from file
mean_fitnesses_cma = []
for o in range(1, runs + 1):
    with open("output/avg_fit_cma_level" + str(level) + "_try" + str(o)) as f:
        mean_f = f.readlines()

    for i in range(len(mean_f)):
        mean_f[i] = mean_f[i].split(",")
        for j in range(len(mean_f[i])):
            mean_f[i][j] = mean_f[i][j].strip()
            mean_f[i][j] = float(mean_f[i][j])

    mean_fitnesses_cma.append(mean_f)


# CALCULATE AVG OF THE MEAN FITNESS VALUES
mean_of_mean_f_cma = []
std_cma = []
for j in range(len(mean_fitnesses_cma[0])):
    values_std = []
    sum_fit = 0
    for e in range(len(mean_fitnesses_cma)):
        sum_fit += mean_fitnesses_cma[e][j][0]
        values_std.append(mean_fitnesses_own_ea[e][j][0])
    std_cma.append(np.std(values_std))
    mean_of_mean_f_cma.append(sum_fit / len(mean_fitnesses_cma))
    if len(mean_of_mean_f_cma) == gens:
        break

std_own_ea_plus = np.add(np.array(avg_mean_f_own_ea), np.array(std_own_ea))
std_own_ea_minus = np.subtract(np.array(avg_mean_f_own_ea), np.array(std_own_ea))
std_cma_plus = np.add(np.array(mean_of_mean_f_cma), np.array(std_cma))
std_cma_minus = np.subtract(np.array(mean_of_mean_f_cma), np.array(std_cma))


#################### PLOTTING ####################


line1 = plt.plot(avg_best_f_own_ea, label="Avg best fitness Own EA")
line2 = plt.plot(avg_best_f_cma, label="Avg best fitness CMA")
line3 = plt.plot(avg_mean_f_own_ea, label="Avg mean fitness Own EA", color="#15a9e8")
line4 = plt.plot(mean_of_mean_f_cma, label="Avg mean fitness CMA", color="#f2b422")
line5 = plt.plot(std_own_ea_plus, "--", linewidth=0.3)
line6 = plt.plot(std_own_ea_minus, "--", linewidth=0.3)
line7 = plt.plot(std_cma_minus, "--", linewidth=0.3)
line8 = plt.plot(std_cma_plus, "--", linewidth=0.3)


np_gens = np.arange(0, gens, 1)
plt.fill_between(np_gens, std_own_ea_plus, std_own_ea_minus, alpha=0.3)
plt.fill_between(np_gens, std_cma_plus, std_cma_minus, alpha=0.3)


plt.legend(prop={"family":"serif"})
plt.xlabel("Generations", fontfamily="serif", fontsize=12)
plt.ylabel("Fitness Value", fontfamily="serif", fontsize=12)
plt.yticks(range(0, 101, 20))
plt.title("Fitness vs. Generation, Level " + str(level), fontsize=16, pad=10, fontfamily="serif")
plt.savefig("plots/plot_level" + str(level) + ".png")
plt.show()
