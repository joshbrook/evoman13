import sys

from evoman.environment import Environment
from demo_controller import player_controller

# imports other libs
import time
import numpy as np
from math import fabs,sqrt
import glob, os


experiment_name = 'weight_test'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

n_hidden_neurons = 10



# initializes simulation in individual evolution mode, for single static enemy.
env = Environment(experiment_name=experiment_name,
				  playermode="ai",
				  player_controller=player_controller(n_hidden_neurons),
			  	  speed="normal",
				  enemymode="static",
				  level=2,
				  visuals=True)


ini = time.time()  # sets time marker


# genetic algorithm params

run_mode = 'test' # train or test

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
    return abs(f*-1)

with open('best_x___||') as f:
    pop = f.readlines()

for i in range(len(pop)):
    pop[i] = pop[i].strip()
    pop[i] = float(pop[i])

env.play(pcont=np.array(pop))