import random
import sys

from evoman.environment import Environment
from demo_controller import player_controller

# imports other libs
import numpy as np
import os
import cma
import matplotlib.pyplot as plt


def simulation(env,x):
    f,p,e,t = env.play(pcont=x)
    f = (f - (-100)) / (100 - (-100)) * 100
    return abs(f - 100)

def evaluate(env, x):
    return np.array(list(map(lambda y: simulation(env,y), x)))


#inverse fitness function for cma resultsx
def min_to_max(x):
    f = (((x - 100)*200)/100)+100
    return abs(f)


fitness = []
#LOOP OVER 10 RUNS
for o in range(1,5):
    #OPEN CORRESPONDING FOLDER OF BEST F
    with open('f_best_level1_test') as f:
        best_f = f.readlines()
    #LOOP OVER CURRENT FILE TRANSFORM STR --> FLOAT + MIN_TO_MAX
    for i in range(len(best_f)):
        best_f[i] = best_f[i].split(',')
        for j in range(len(best_f[i])):
            best_f[i][j] = best_f[i][j].strip()
            best_f[i][j] = float(best_f[i][j])
            best_f[i][j] = min_to_max(best_f[i][j])
    #APPEND FINAL FILE AS LIST
    fitness.append(best_f)


avg_best_f = []
#CALCULATE AVG
#LOOP OVER THE FIRST LIST
for j in range(len(fitness[0])):
    sum_fit = 0
    #FOR EACH ELEMENT OF THE FIRST LIST
    #CALCULATE THE SUM --> ADD TO THE CURRENT ELEMENT ALL THE OTHER ELEMENT AT THE SAME INDEX
    # 1ST RUN INDEX 0 + 2ND RUN INDEX 0 + 3RD RUN INDEX 0....
    for e in range(len(fitness)):
        sum_fit += fitness[e][j][0]
    #CALCULATE AVG
    avg_best_f.append(sum_fit/len(fitness))
    if len(avg_best_f) == 999:
        break


#SAME AS ABOVE BUT FOR THE AVG FITNESS INSTEAD OF BEST
mean_fitnesses=[]
for o in range(1,5):
    with open('avg_fit_cma_level1_test') as f:
        mean_f = f.readlines()

    for i in range(len(mean_f)):
        print(len(mean_f))
        mean_f[i] = mean_f[i].split(',')
        for j in range(len(mean_f[i])):
            mean_f[i][j] = mean_f[i][j].strip()
            mean_f[i][j] = float(mean_f[i][j])

    mean_fitnesses.append(mean_f)


#CALCULATE AVG OF THE MEAN F
mean_of_mean_f = []
for j in range(len(mean_fitnesses[0])):
    sum_fit = 0
    for e in range(len(mean_fitnesses)):
        sum_fit += mean_fitnesses[e][j][0]
    mean_of_mean_f.append(sum_fit/len(mean_fitnesses))
    if len(mean_of_mean_f) == 999:
        break


#PLOT
line1 = plt.plot(avg_best_f, label='Average of the best fitness')
line2 = plt.plot(mean_of_mean_f, label='Average of the mean fitness')
plt.legend()
plt.xlabel('Generations')
plt.ylabel('Fitness Value')
plt.show()
plt.savefig('plot_cma_level1_try3.png')
