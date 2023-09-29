###############################################################################
# EvoMan FrameWork - V1.0 2016  			                                  #
# DEMO : Neuroevolution - Genetic Algorithm  neural network.                  #
# Author: Karine Miras        			                                      #
# karine.smiras@gmail.com     				                                  #
###############################################################################

# import framework
from evoman.environment import Environment
from demo_controller import player_controller

# import other libs
import numpy as np
import random
import os


def simulation(env, x, min=False):
    """Simulates one individual and returns its fitness"""
    f, p, e, t = env.play(pcont=x)
    if not min:
        return f
    else:
        f = (f - (-100)) / (100 - (-100)) * 100
        return abs(f - 100)


def evaluate(env, x):
    """Evaluates fitness of entire population"""
    return np.array(list(map(lambda y: simulation(env, y), x)))


def parent_selection(n_top, pop, fit_pop):
    """Extracts top n candidates"""
    return pop[np.argpartition(np.array(fit_pop), -n_top)[-n_top:]]


def mutation(parents, mut, exit_local_optimum):
    """gaussian noise mutation"""
    mutated = []
    for parent in parents:
        # duplicate parent
        mut_parent = parent.copy()

        # iter over the genes and randomly add gaussian noise
        # mut is the mutation criteria between 0 & 1 - higher --> more prob of mutation
        for e in range(len(parent)):
            if random.uniform(0, 1) < mut:
                # if no offspring > parent --> increase step size to find solutions outside local optimum
                if exit_local_optimum:
                    mut_e = mut_parent[e] + np.random.normal(0, 10)
                else:
                    # else add noise btw -1 and 1
                    mut_e = mut_parent[e] + np.random.normal(0, 1)
                mut_parent[e] = mut_e

        mutated.append(mut_parent)

    return np.array(mutated)


def crossover(mutated, n_top, npop):
    """simple arithmetic random-index-based crossover"""
    children = []
    for n in range(npop - n_top):
        # take two random candidate parents from top_n individuals
        p1, p2 = random.sample(list(mutated), 2)

        # generate random index between 2nd element & one before last element
        i = random.randint(1, len(p1))

        # concatenate p1 until i with p2 going from i
        # EXAMPLE
        # p1 = ------------
        # p2 = &&&&&&&&&&&&
        # i = 3
        # c1 = ---&&&&&&&&&
        # c2 = &&&---------

        children.append(np.concatenate((p1[:i], p2[i:])))

    return children


def survivor_selection(mutated, children, pop, fit_pop, n_top, npop, env, rep):
    """
    Modified Elitism selection strategy.
    Keep top_n parents in next generation
    Create 80 children from pairs of top_n parents.
    Test each child against random individual from previous generation.
    If better, replace previous individual with new child in next generation.
    """
    bottom_n = pop[
        np.argpartition(np.array(fit_pop), -(npop - n_top))[-(npop - n_top) :]
    ]
    newpop = list(mutated)

    for child in children:
        # evaluate child with random individual from previous generation
        parent = random.choice(bottom_n)
        f_c = simulation(env, child)
        f_p = simulation(env, parent)

        # if f of child > f of chosen individual, then replace
        if f_c > f_p:
            newpop.append(child)
            rep += 1
        else:
            newpop.append(parent)

    pop = np.array(newpop)

    # evaluate new pop fitness
    fit = evaluate(env, pop)
    return pop, fit, rep


def main():
    level = 1
    print("\nInitializing simulation, level " + str(level))

    for i in range(1, 11):
        print("\nRun: " + str(i))

        # setup
        headless = True
        if headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"

        experiment_name = "optimization_test"
        if not os.path.exists(experiment_name):
            os.makedirs(experiment_name)

        n_hidden_neurons = 10

        # initialise simulation
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

        # no. weights for multilayer with 10 hidden neurons
        n_vars = (env.get_num_sensors() + 1) * n_hidden_neurons + (
            n_hidden_neurons + 1
        ) * 5

        # variables
        n_top = 10
        dom_u = 1
        dom_l = -1
        npop = 40
        gens = 100
        mutation_crit = 0.2
        rep = 0
        exit_local_optimum = False
        pop = np.random.uniform(dom_l, dom_u, (npop, n_vars))
        pop_fit = evaluate(env, pop)
        best_f = [np.amax(np.array(pop_fit))]
        mean_f = [np.mean(np.array(pop_fit))]

        # generations
        for g in range(gens + 1):
            # keep track of f best
            best_f.append(np.amax(np.array(pop_fit)))
            mean_f.append(np.mean(np.array(pop_fit)))

            if np.amax(np.array(pop_fit)) > best_f[-1]:
                print("-----NEW BEST FITNESS----")

            # print info about gen and best fit
            if g % 5 == 0:
                print("########################")
                print("GEN: ", g)
                print("BEST FITNESS: ", np.amax(np.array(pop_fit)))
                print("MEAN FITNESS: ", np.mean(np.array(pop_fit)))

            # Parent Selection
            parents = parent_selection(n_top, pop, pop_fit)

            # Mutations
            mutated = mutation(parents, mutation_crit, exit_local_optimum)

            # Crossover
            children = crossover(mutated, n_top, npop)

            # Survivor selection
            pop, pop_fit, rep = survivor_selection(
                mutated, children, pop, pop_fit, n_top, npop, env, rep
            )

            # check for no evolution
            if g % 10 == 0 and g != 0 and best_f.count(best_f[-1]) > 10:
                if rep < 5:
                    exit_local_optimum = True
                    print("-----ATTEMPT EXIT LOCAL OPTIMUM-----")
                else:
                    exit_local_optimum = False
                rep = 0

        # save results
        np.savetxt(
            "output/pop_level" + str(level) + "_try" + str(i), pop, delimiter=","
        )
        np.savetxt(
            "output/fit_level" + str(level) + "_try" + str(i), pop_fit, delimiter=","
        )
        np.savetxt(
            "output/best_f_level" + str(level) + "_try" + str(i), best_f, delimiter=","
        )
        np.savetxt(
            "output/mean_f_level" + str(level) + "_try" + str(i), mean_f, delimiter=","
        )


if __name__ == "__main__":
    main()
