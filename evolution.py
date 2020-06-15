import json
import argparse

import numpy as np

import torch
import torch.nn as nn

import matplotlib.pyplot as plt

import utils.ga as ga
from utils.util import get_logger
from utils.config import get_config
from utils.graph import calculate_nodes
from train_predictor import get_input_data, get_edge_index, wrap_data

def decode_population(architecture, nodes_num):
    adj_matrix = np.identity(nodes_num, nodes_num)
    for l in range(5):
        adj_matrix[l*7, l*7+1:(l+1)*7] = architecture[6*l:6*(l+1)]
        adj_matrix[l*7+1:(l+1)*7, (l+1)*7] = architecture[6*l:6*(l+1)]

    return adj_matrix

def predict_accuracy(new_population, model, nodes_num):
    fitness = []
    for p in new_population:
        adj_matrix = decode_population(p, nodes_num)

        X = get_input_data(adj_matrix)
        edge_index = get_edge_index(adj_matrix)

        X = wrap_data(X)
        edge_index = wrap_data(edge_index, dtype=torch.long)

        outs = model(X, edge_index)
        fitness.append(outs.item())
    return np.array(fitness)


def evolution():
    parser = argparse.ArgumentParser()
    paraser.add_argument("--cfg", type=str, required=True)
    macs_budget = 15

    CONFIG = get_config(args.cfg)
    
    nodes_num = calculate_nodes(CONFIG)

    model = Predictor(nodes_num)
    model.load_state_dict(torch.load("./gcn_weight.pth"))
    model = model.cuda()

    sol_per_pop = 20
    num_parents_mating = 10

    pop_size = (sol_per_pop, 5*6)
    new_population = np.random.randint(low=0, high=1, size=pop_size)

    best_outputs = []
    num_generations = 1000
    for generation in range(num_generations):
        print("Generation : ", generation)
        fitness = predict_accuracy(new_population, model, nodes_num)

        best_outputs.append(np.max(fitness))
        print("Best result : ", np.max(fitness))

        parents = ga.select_mating_pool(new_population, fitness, num_parents_mating)

        offspring_size = (sol_per_pop-num_parents_mating, 30)
        evolution_id = 0
        offspring = np.empty(offspring_size, dtype=np.int32)
        while evolution_id < offspring_size[0]:
            offspring_crossover = ga.crossover(evolution_id, parents, offspring_size, macs_budget)

            offspring_mutation = ga.mutation(offspring_crossover, num_mutations=2)
            offspring_mutation_macs = cal_macs(offspring_mutation[0])

            if offspring_mutation_macs <= macs_budget:
                offspring[evolution_id] = offspring_mutation
                evolution_id += 1
        new_population[0:parents.shape[0], :] = parents
        new_population[parents.shape[0]:, :] = offspring_mutation

    fitness = predict_accuracy(new_population, model, nodes_num)
    best_idx = np.argmax(fitness)
    print("Best solution : ", new_population[best_idx, :])
    print("Best predict accuracy : ", fitness[best_idx])

    architecture_metric = []
    architecture_metric.append(decode_population(new_population[best_idx, :], nodes_num).reshape(-1))
    df_architecture = pd.DataFrame(architecture_metric)
    df_architecture.to_csv("./evolution_architecture", index=False)
