import os
import numpy as np

from evoman.environment import Environment
from demo_controller import player_controller


experiment_name = "weight_test"
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

n_hidden_neurons = 10


# initializes simulation in individual evolution mode, for single static enemy.
env = Environment(
    experiment_name=experiment_name,
    enemies=[1],
    playermode="ai",
    player_controller=player_controller(n_hidden_neurons),
    speed="normal",
    enemymode="static",
    level=2,
    visuals=True,
)


# TESTING WEIGHTS OBTAINED FROM OWN EA
# MAKE SURE THE FILENAME FOR WEIGHTS AND FITNESS CORRESPONDS TO THE WEIGHTS AND FIT YOU WANT TO TEST
# SEE BELOW TO TEST WEIGHTS FROM CMA
with open("output/pop_level1_try4") as f:
    pop = f.readlines()

for i in range(len(pop)):
    pop[i] = pop[i].split(",")
    for j in range(len(pop[i])):
        pop[i][j] = pop[i][j].strip()
        pop[i][j] = float(pop[i][j])


with open("output/fit_level1_try4") as f:
    fit = f.readlines()

for i in range(len(fit)):
    fit[i] = fit[i].split(",")
    for j in range(len(fit[i])):
        fit[i][j] = fit[i][j].strip()
        fit[i][j] = float(fit[i][j])


max_i = np.argmax(fit)
pop = pop[max_i]


f, _, _, _ = env.play(pcont=np.array(pop))
print("Fitness: ", f)
