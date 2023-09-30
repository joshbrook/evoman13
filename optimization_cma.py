import os
import time
import cma

import matplotlib.pyplot as plt
import numpy as np

from math import fabs, sqrt
from demo_controller import player_controller
from evoman.environment import Environment

# SETUP
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

experiment_name = "cma_opt_test"
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

n_hidden_neurons = 10
level = 7
runs = 10
npop = 40
gens = 50

env = Environment(
    experiment_name=experiment_name,
    enemies=[level],
    playermode="ai",
    player_controller=player_controller(n_hidden_neurons),
    enemymode="static",
    level=2,
    speed="fastest",
    visuals=False,
)

ini = time.time()  # sets time marker

# genetic algorithm params

run_mode = "train"  # train or test

# number of weights for multilayer with 10 hidden neurons
n_vars = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5

dom_u = 1
dom_l = -1
mutation = 0.2
last_best = 0


def simulation(env, x, min):
    f, p, e, t = env.play(pcont=x)
    if min:
        f = (f - (-100)) / (100 - (-100)) * 100
        return abs(f - 100)
    else:
        return f


def evaluate(env, x, min):
    return np.array(list(map(lambda y: simulation(env, y, min), x)))


for i in range(1, runs + 1):
    # init weights
    pop = np.random.uniform(dom_l, dom_u, (1, n_vars))

    # init cma
    es = cma.CMAEvolutionStrategy(pop, 40)

    f_best = []
    avg_fit = []

    # find solution for weights
    while not es.stop():
        # ask set of new solutions
        solutions = es.ask()

        # evaluate each new solution
        es.tell(solutions, [simulation(env, pop, min=True) for pop in solutions])
        es.logger.add()
        es.disp()

        # append best fitness
        f_best.append(es.result_pretty()[1])

        # get the current population
        pop = es.pop_sorted

        # evaluate the current population - with fitness maximisation
        pop_fit = evaluate(env, pop, min=False)

        # calculate the avg fitness
        avg_pop_fit = np.sum(pop_fit) / len(pop_fit)

        # append it
        avg_fit.append(avg_pop_fit)

        # same number of generations as in own EA
        if es.countiter == gens:
            break

    # save best_weights
    best_weights = es.result_pretty()[0].copy()
    np.savetxt(
        "output/best_x_cma_level" + str(level) + "_try" + str(i),
        best_weights,
        delimiter=",",
    )
    np.savetxt(
        "output/f_best_cma_level" + str(level) + "_try" + str(i), f_best, delimiter=","
    )
    np.savetxt(
        "output/avg_fit_cma_level" + str(level) + "_try" + str(i),
        avg_fit,
        delimiter=",",
    )
