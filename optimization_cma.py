import sys

from evoman.environment import Environment
from demo_controller import player_controller

# imports other libs
import time
import numpy as np
from math import fabs,sqrt
import glob, os
import cma
import matplotlib.pyplot as plt


headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"


experiment_name = 'cma_opt_test'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

n_hidden_neurons = 10
env = Environment(experiment_name=experiment_name,
                  enemies=[1],
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest",
                  visuals=False)

ini = time.time()  # sets time marker


# genetic algorithm params

run_mode = 'train' # train or test

# number of weights for multilayer with 10 hidden neurons
n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5


dom_u = 1
dom_l = -1
npop = 100
gens = 30
mutation = 0.2
last_best = 0

def simulation(env,x):
    f,p,e,t = env.play(pcont=x)
    f = (f - (-100)) / (100 - (-100)) * 100
    return abs(f - 100)


#init weights
pop = np.random.uniform(dom_l, dom_u, (1, n_vars))
#init cma
es = cma.CMAEvolutionStrategy(pop, 40)
#set options
#es.opts.set({'tolflatfitness' : 5})
es.opts.set({'ftarget': 0})

#find solution for weights
while not es.stop():
    #ask set of new solutions
    solutions = es.ask()
    #evluate each new solutions
    es.tell(solutions,[simulation(env, pop) for pop in solutions])
    es.logger.add()
    es.disp()

#save plot
es.logger.plot()
plt.savefig('plot_cma_level1.png')
#save best_weights
best_weights = es.result_pretty()[0].copy()
np.savetxt('best_x_cma_level1', best_weights, delimiter=',')
