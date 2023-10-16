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
    enemies=[1, 2, 3, 4, 5, 6, 7, 8],
    multiplemode="yes",
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
with open("output/best_x_cma_levelall_try1") as f:
    pop = [float(n.strip()) for n in f.readlines()]

print(pop)

f, _, _, _ = env.play(pcont=np.array(pop))
print("Fitness: ", f)
