import random
import sys

from evoman.environment import Environment
from demo_controller import player_controller

# imports other libs
import numpy as np
import os
import cma
import matplotlib.pyplot as plt



#inverse fitness function for cma results
def min_to_max(x):
    f = (((x - 100)*200)/100)+100
    return abs(f)


fitness = []

for o in range(1,10):
    with open('pop_fit_bestf_mean_f/level1/best_f_level1_try'+str(o)) as f:
        best_f = f.readlines()

    for i in range(len(best_f)):
        best_f[i] = best_f[i].split(',')
        for j in range(len(best_f[i])):
            best_f[i][j] = best_f[i][j].strip()
            best_f[i][j] = float(best_f[i][j])
            best_f[i][j] = best_f[i][j]

    fitness.append(best_f)

avg_best_f = []
for j in range(len(fitness[0])):
    sum_fit = 0
    for e in range(len(fitness)):
        sum_fit += fitness[e][j][0]
    avg_best_f.append(sum_fit/len(fitness))
    if j == 0:
        print(avg_best_f)
    if len(avg_best_f) == 999:
        break


mean_fitnesses=[]
for o in range(1,10):
    with open('pop_fit_bestf_mean_f/level1/mean_f_level1_try'+str(o)) as f:
        mean_f = f.readlines()

    for i in range(len(mean_f)):
        mean_f[i] = mean_f[i].split(',')
        for j in range(len(mean_f[i])):
            mean_f[i][j] = mean_f[i][j].strip()
            mean_f[i][j] = float(mean_f[i][j])

    mean_fitnesses.append(mean_f)

mean_of_mean_f = []


for j in range(len(mean_fitnesses[0])):
    sum_fit = 0
    for e in range(len(mean_fitnesses)):
        sum_fit += mean_fitnesses[e][j][0]
    mean_of_mean_f.append(sum_fit/len(mean_fitnesses))
    if j == 0:
        print(mean_of_mean_f)
    if len(mean_of_mean_f) == 999:
        break

line1 = plt.plot(avg_best_f, label='Average of the best fitness over 10 runs')
line2 = plt.plot(mean_of_mean_f, label='Average of the mean fitness over 10 runs')
plt.legend()
plt.xlabel('Generations')
plt.ylabel('Fitness Value')
plt.savefig('plot_own_EA_level1.png')
plt.show()
