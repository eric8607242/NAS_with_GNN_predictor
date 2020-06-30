import numpy as np

def select_mating_pool(pop, fitness, num_parents):
    # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
    parents = np.empty((num_parents, pop.shape[1]))
    for parent_num in range(num_parents):
        max_fitness_idx = np.argmax(fitness)
        parents[parent_num, :] = pop[max_fitness_idx, :]
        fitness[max_fitness_idx] = -99999999999
    return parents

def crossover(evolution_id, parents, offspring_size, macs_budget):
    max_layers = parents.shape[1]
    offspring = np.empty((1, max_layers), dtype=np.int32)

    parents_size = parents.shape[0]


    crossover_point = np.random.randint(low=0, high=max_layers)
    parent1_idx = np.random.randint(low=0, high=parents_size)
    parent2_idx = np.random.randint(low=0, high=parents_size)
    #parent1_idx = evolution_id%parents_size
    #parent2_idx = (evolution_id+1)%parents_size

    offspring[0, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
    offspring[0, crossover_point:] = parents[parent2_idx, crossover_point:]

    return offspring

def mutation(offspring_crossover, num_mutations=1):
    max_layers = offspring_crossover.shape[1]
    mutations_counter = int(offspring_crossover.shape[1] / num_mutations)

    for l in range(max_layers):
        mut_pro = np.random.choice([0, 1], p=[0.9, 0.1])
        random_value = np.random.randint(low=0, high=2)
        if mut_pro == 1:
            offspring_crossover[0, l] = random_value
        
    return offspring_crossover

    

