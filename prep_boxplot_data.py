import sys

from evoman.environment import Environment
from demo_controller import player_controller

# imports other libs
import time
import numpy as np
from math import fabs, sqrt
import glob, os


experiment_name = "weight_test"
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

n_hidden_neurons = 10

# change values here
level = 7
tries = 10

# initializes simulation in individual evolution mode, for single static enemy.
env = Environment(
    experiment_name=experiment_name,
    enemies=[level],  # CHANGE LEVEL HERE
    playermode="ai",
    player_controller=player_controller(n_hidden_neurons),
    speed="fastest",
    enemymode="static",
    level=2,
    visuals=False,
)


def simulation(env, x, min=False):
    f, p, e, t = env.play(pcont=x)
    if min == False:
        return f
    else:
        f = (f - (-100)) / (100 - (-100)) * 100
        return abs(f - 100)


# evaluation
def evaluate(env, x):
    return np.array(list(map(lambda y: simulation(env, y), x)))


for attempt in range(1, tries+1):

    with open("output/pop_level" + str(level) + "_try" + str(attempt)) as f:
        pop = f.readlines()
    print(len(pop[0]))
    for i in range(len(pop)):
        pop[i] = pop[i].split(",")
        for j in range(len(pop[i])):
            pop[i][j] = pop[i][j].strip()
            pop[i][j] = float(pop[i][j])


    with open("output/fit_level" + str(level) + "_try" + str(attempt)) as f:
        fit = f.readlines()

    for i in range(len(fit)):
        fit[i] = fit[i].split(",")
        for j in range(len(fit[i])):
            fit[i][j] = fit[i][j].strip()
            fit[i][j] = float(fit[i][j])


    max_i = np.argmax(fit)
    pop = pop[max_i]
    individual_gain = 0

    for i in range(5):
        f, p, e, t = env.play(pcont=np.array(pop))
        individual_gain += p - e
        # print(individual_gain)
    np.savetxt(
        "output/ind_gain_own_ea_level" + str(level) + "_try" + str(attempt),
        [individual_gain / 5],
        delimiter=",",
    )


for attempt in range(1, tries+1):

    with open('output/best_x_cma_level' + str(level) + "_try" + str(attempt)) as f:
        pop = f.readlines()

    for i in range(len(pop)):
        pop[i] = pop[i].split(',')
        for j in range(len(pop[i])):
            pop[i][j] = pop[i][j].strip()
            pop[i][j] = float(pop[i][j])


    with open("output/f_best_cma_level" + str(level) + "_try" + str(attempt)) as f:
        fit = f.readlines()

    for i in range(len(fit)):
        fit[i] = fit[i].split(",")
        for j in range(len(fit[i])):
            fit[i][j] = fit[i][j].strip()
            fit[i][j] = float(fit[i][j])


    individual_gain = 0

    for i in range(5):
        f, p, e, t = env.play(pcont=np.array(pop))
        individual_gain += p - e
        # print(individual_gain)
    np.savetxt(
        "output/ind_gain_cma_level" + str(level) + "_try" + str(attempt),
        [individual_gain / 5],
        delimiter=",",
    )