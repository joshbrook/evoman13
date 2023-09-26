###############################################################################
# EvoMan FrameWork - V1.0 2016  			                                  #
# DEMO : Neuroevolution - Genetic Algorithm  neural network.                  #
# Author: Karine Miras        			                                      #
# karine.smiras@gmail.com     				                                  #
###############################################################################

# imports framework
import random
import sys

from evoman.environment import Environment
from demo_controller import player_controller

# imports other libs
import numpy as np
import os
import cma
import matplotlib.pyplot as plt

# runs simulation
def simulation(env,x,min=False):
    f, p, e, t = env.play(pcont=x)
    if min == False:
        return f
    else:
        f = (f - (-100)) / (100 - (-100)) * 100
        return abs(f - 100)

# evaluation
def evaluate(env, x):
    return np.array(list(map(lambda y: simulation(env,y), x)))


def parent_selection(pop, fit_pop):
    #pick best parent in pop
    best_fit_index = np.argmax(np.array(fit_pop))
    best_parent = pop[best_fit_index]

    #for now random second parent
    rand_i = random.randint(0,len(pop)-1)
    second_parent = pop[rand_i]

    #check if they are not the same
    """ 
    while set(best_parent) == set(second_parent.sort()):
        print('SAME PARENTS FOUND')
        print('###############################')
        print('1: ',best_parent[:5])
        print('2: ',second_parent[:5])
        print('###############################')

        rand_i = random.randint(0, len(pop) + 1)
        second_parent = pop[rand_i]
    """

    return best_parent, second_parent, rand_i


def mutation(parent, mut, exit_local_optimum):

    #gaussian noise mutation
    #duplicate parent
    mut_parent = parent.copy()

    #iter over the genes and randomly add goissian noise
    #mut is the mutation criteria btw 0 & 1 - higher --> more prob of mutation
    for e in range(len(parent)):
        if random.randint(0, 1) < mut:
            #if no offspring > parent --> increase step size to find solutions outside local optimum
            if exit_local_optimum == True:
                mut_e = mut_parent[e] + np.random.normal(0, 10)
            else:
            #else add noise btw -1 and 1
                mut_e = mut_parent[e] + np.random.normal(0, 1)
            mut_parent[e] = mut_e

    #return the mutated parent
    return mut_parent

def crossover(mut_1, mut_2):
    
    #simple arithmetic crossover
    #generate random index btw 2nd element & one before last element
    i = random.randint(1, len(mut_1))

    #concatenate mut_1 until i with mut_2 going from i
    #------------    &&&&&&&&&&
    #             i
    #       ------&&&&&&&
    recombined_offspring = np.concatenate((mut_1[:i], mut_2[i:]))

    return recombined_offspring

def survivor_selection(second_parent, offspring, pop, env, id_second, replacement):


    #evaluate second parent and offspring
    f_second_parent = simulation(env, second_parent)
    f_offspring = simulation(env, offspring)

    #if f of off-spring > f of second_parent then replace
    if f_offspring > f_second_parent:
        pop[id_second] = offspring
        print('###REPLACEMENT###')
        print('OLD FITNESS: ', f_second_parent)
        print('NEW FITNESS', f_offspring)
        replacement += 1

    #evaluate new pop fit
    fit = evaluate(env, pop)
    return pop, fit, replacement



def main():
    # choose this for not using visuals and thus making experiments faster
    for i in range(2,11):
        headless = True
        if headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"


        experiment_name = 'optimization_test'
        if not os.path.exists(experiment_name):
            os.makedirs(experiment_name)

        n_hidden_neurons = 10

    # initializes simulation in individual evolution mode, for single static enemy.
        env = Environment(experiment_name=experiment_name,
                    enemies=[3],
                    playermode="ai",
                    player_controller=player_controller(n_hidden_neurons), # you  can insert your own controller here
                    enemymode="static",
                    level=2,
                    speed="fastest",
                    visuals=False)


    # number of weights for multilayer with 10 hidden neurons
        n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5

    # start writing your own code from here


        dom_u = 1
        dom_l = -1
        npop = 100
        gens = 1000
        mutation_crit = 0.2
        replacement = 0
        exit_local_optimum = False
        pop  = np.random.uniform(dom_l, dom_u, (npop, n_vars))
        pop_fit = evaluate(env, pop)
        best_f = [np.amax(np.array(pop_fit))]
        mean_f = [np.mean(np.array(pop_fit))]
    #print(mean_f)


    #generations
        for g in range(gens):
        #keep track of f best
            best_f.append(np.amax(np.array(pop_fit)))
            mean_f.append(np.mean(np.array(pop_fit)))

            if np.amax(np.array(pop_fit)) > best_f[-1]:
                print('NEW BEST FITNESS EVER!')

        #print info about gen and best fit
            if g % 5 == 0:
                print('########################')
                print('GEN: ', g)
                print('BEST_FITNESS: ', np.amax(np.array(pop_fit)))
                print('########################')


        #PARENT SELECTION
            best_parent, second_parent, index_second = parent_selection(pop, pop_fit)

        #Mutations
            mut_best = mutation(best_parent, mutation_crit, exit_local_optimum)
            mut_second = mutation(second_parent, mutation_crit, exit_local_optimum)

        #Cross-over
            offspring = crossover(mut_best, mut_second)

        #Survivor selection
            pop, pop_fit, replacement = survivor_selection(second_parent, offspring, pop, env, index_second, replacement)

        #check for no evolution
            if g % 20 == 0 and g != 0:
                if replacement < 5:
                    exit_local_optimum = True
                    print('--------FINDIND EXIT OF LOCAL OPTIMUM-----------')
                else:
                    exit_local_optimum = False
                replacement = 0

            # plot
        #best_f.append(simulation(env, best_parent))

    #save results
        np.savetxt('pop_level3_try'+str(i), pop, delimiter=',')
        np.savetxt('fit_level3_try'+str(i), pop_fit, delimiter=',')
        np.savetxt('best_f_level3_try'+str(i), best_f, delimiter=',')
        np.savetxt('mean_f_level3_try'+str(i), mean_f, delimiter=',')










if __name__ == '__main__':
    main()