import os
import numpy as np

from evoman.environment import Environment
from demo_controller import player_controller


experiment_name = "weight_test"
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)
8136741
n_hidden_neurons = 10
level = "1682357"
f_total = []

for l in range(1, 9):
# initializes simulation in individual evolution mode, for single static enemy.
    env = Environment(
        experiment_name=experiment_name,
        enemies=[l],
        multiplemode="no",
        playermode="ai",
        player_controller=player_controller(n_hidden_neurons),
        speed="fastest",
        enemymode="static",
        level=2,
        visuals=True,
    )


    # TESTING WEIGHTS OBTAINED FROM OWN EA
    # MAKE SURE THE FILENAME FOR WEIGHTS AND FITNESS CORRESPONDS TO THE WEIGHTS AND FIT YOU WANT TO TEST
    # SEE BELOW TO TEST WEIGHTS FROM CMA
    with open("output/pop_level" + level + "_try1") as f:
        pop = f.readlines()

    for i in range(len(pop)):
        pop[i] = pop[i].split(",")
        for j in range(len(pop[i])):
            pop[i][j] = float(pop[i][j].strip())


    with open("output/fit_level" + level + "_try1") as f:
        fit = [float(l) for l in f.readlines()]


    max_i = np.argmax(fit)
    ind = pop[max_i]

    f, p, e, _ = env.play(pcont=np.array(ind))
    print("Enemy: ", l)
    print("Fitness: ", f)
    print("Player life: ", p)
    print("Enemy life: ", e)

    f_total.append(f)

print("Mean fitness: ", sum(f_total)/len(f_total))

np.savetxt("output/weights" + level + ".txt", ind, delimiter=",")
