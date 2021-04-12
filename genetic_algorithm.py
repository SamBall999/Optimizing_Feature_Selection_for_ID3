import numpy as np
import pandas as pd
from custom_id3 import ID3_Decision_Tree
from testing import calculate_rates, accuracy


#----GENETIC ALGORITHM (GA)----#


# potential downside of GA compared to tabu -> more function evals



# calculates fitness of current position = ability of individual to survive
# Objectives: maximize classification accuracy of the feature subset. 
def fitness_function(individual, training_data, validation_data):
 
    # check that feature subset contains at least one feature
    if (np.sum(individual) == 0): # if no features
        return 0
    
    # get selected features from the bitvector encoding
    feature_indices = [(i+1) for i in range(len(individual)) if individual[i]==1]
    feature_list =  training_data.columns.values[feature_indices] # wholelist or only some?
    
    # convert x and y data to numpy arrays
    x = np.array(training_data.drop("Target", axis=1).copy())
    y = np.array(training_data["Target"])
    
    # create decision tree with given subset
    tree = ID3_Decision_Tree(x, y) # intialize decision tree
    tree.id_3(feature_list) # build tree

   
    # verify if training accuracy is 100% (overfitting is occurring)
    #predictions = tree.predict_batch(training_data)
    #predictions = tree.predict_batch(x)
    #targets = training_data["Target"].values
    #TP, FP, TN, FN = calculate_rates(targets, predictions)
    #training_accuracy = accuracy(TP, FP, TN, FN)
    #print("Training Accuracy: {}".format(training_accuracy))


    # calculate accuracy on the validation set
    X_val = np.array(validation_data.drop("Target", axis=1).copy())
    validation_predictions = tree.predict_batch(X_val)
    validation_targets = validation_data["Target"].values
    TP, FP, TN, FN = calculate_rates(validation_targets, validation_predictions)
    validation_accuracy = accuracy(TP, FP, TN, FN)
    #print("Validation Accuracy: {}".format(validation_accuracy))

    # fitness is measured by validation accuracy
    fitness = validation_accuracy

    return fitness



# single point 
def single_point_crossover_mask(n_x):

    crossover_point = np.random.randint(1, n_x, 1)

    #print(crossover_point)

    mask = np.zeros(n_x, dtype=int)
    j = crossover_point
    while (j < n_x):
        mask[j] = 1 # assign this range to 1's
        j+=1
    #print(mask)
    return mask





# also try single point or uniform crossover
def two_point_crossover_mask(n_x):

    possible_points = np.arange(1, n_x)
    crossover_points = np.random.choice(possible_points,  2, replace = False) #np.random.randint(1, n_x, 2)# select crossover points, no replacement
    e_1 = min(crossover_points)
    e_2 = max(crossover_points)
    print(crossover_points)
    print(e_1)
    print(e_2)
    mask = np.zeros(n_x, dtype=int)
    j = e_1 + 1
    while (j < e_2):
        mask[j] = 1 # assign this range to 1's
        j+=1
    print(mask)
    return mask



# produce new offspring 
# also try single point crossover
def bitstring_crossover(parent_1, parent_2, n_x):

    print("Crossover")

    # init children
    child_1 = parent_1.copy() # copies are not independent?
    child_2 = parent_2.copy()

    # get crossover mask
    #mask = two_point_crossover_mask(n_x)
    mask = single_point_crossover_mask(n_x)

    # apply mask
    for j in range(0, n_x): # do we start at 1 or 0?
        if (mask[j] == 1):
            child_1[j] = parent_2[j]
            child_2[j] = parent_1[j]

    return [child_1, child_2]


# introduce new genetic material
# random mutation -> apply p_m to each bit
def mutate(chromosome, p_m):

    print("Mutation")

    for i in range(len(chromosome)):
        r = np.random.uniform(0, 1)
        if (r < p_m):
            chromosome[i] = not chromosome[i] 

    return chromosome



def proportional_probability(individual, fitness_sum, population_fitnesses):
    # NB scaled fitnesses??
    fitness = 0 # if [0 0 0 0 ] # not allowed therefore probability to be selected must be zero
    if (np.sum(individual) != 0):
        #fitness = fitness_function(individual, training_data, validation_data) # scaled?
        individual_string = str(individual).replace(" ", "")
        fitness = population_fitnesses[individual_string]
    #all_fitnesses = [fitness_function(population[i], training_data, validation_data) for i in range(len(population))]
    prob = fitness/fitness_sum # probability that individual x_i will be selected
    #print(prob)
    return prob



# Roulette wheel selection -> can also try more balanced variant: stochastic universal sampling
def roulette_wheel_selection(population, fitness_sum, population_fitnesses):

    print("Roulette selection")


    i = 0 # first chromosome
    p_i = proportional_probability(population[i], fitness_sum, population_fitnesses) # probability of first chromosome
    #probs = [proportional_probability(i, fitness_sum, population_fitnesses) for i in population]
    #print(probs)
    #print(np.sum(probs))
    summation = p_i # points to the top of the first region
    r = np.random.uniform(0, 1) # generate random number sampled uniformly between 0 and 1 - do we need to include 1 somehow??
    #print("R {}:".format(r)) # sometimes we dont reach r before i is out of range?
    while (summation < r):
        i = i+1
        #print(i)
        summation = summation + proportional_probability(population[i], fitness_sum, population_fitnesses)
        #print("Sum {}:".format(summation))

    #print(population[i])
    #print(proportional_probability(population[i], all_fitnesses, training_data, validation_data))
    return population[i] # selected individual/parent


# choose fitter individuals for reproduction -> should also try rank-based at some point
# (i) Roulette wheel selection - with or without replacement??
def parent_selection(population, population_fitnesses):

    print("Parent selection")
    # how many parents to select?? - for now we select 2 parents -> 2 children
    # therefore select no. of parents = half of pop?
    num_parents = int(len(population)/2)
    if(num_parents%2 != 0):
        num_parents = num_parents+1
    p = [] # list of parents

    # compute sum of all fitnesses in population once
    #all_fitnesses = [fitness_function(population[i], training_data, validation_data) for i in range(len(population))]
    #fitness_sum = np.sum(all_fitnesses) # sum of fitnesses across all individuals
    fitness_sum = sum(population_fitnesses.values())
    #print(fitness_sum)

    for i in range(num_parents):
        p.append(roulette_wheel_selection(population, fitness_sum, population_fitnesses))

    return p # selected parents


# try to have fewest fucntion evals possible
def select_new_gen(parents, children, population, population_fitnesses, training_data, validation_data):

    new_generation = []
    no_children =  len(children)
    sorted_parents = sorted(population_fitnesses, key=population_fitnesses.get, reverse=False)

    print(sorted_parents)
    
    #print(parent, population_fitnesses[parent])
    
    i = 0
    for c in children:
        child_fitness = fitness_function(c, training_data, validation_data)
        for i in range(len(parents)):
            if (child_fitness > population_fitnesses[sorted_parents[i]]):
                new_generation.append(c)
                break
    
    num_new_individuals = len(new_generation)
    print(num_new_individuals)
    print(sorted_parents[num_new_individuals:])
    for parent in sorted_parents[num_new_individuals:]:
        parent_array = np.array(list(parent[1:-1]), dtype=int)
        new_generation.append(parent_array)
    print(new_generation) # what about remaining population who were not parents??
    remaining_pop = [i for i in population if str(i).replace(" ", "") not in sorted_parents]
    print(remaining_pop)
    new_generation.append(remaining_pop)

    return new_generation

    #child_fitness = fitness_function(child, training_data, validation_data)



def reproduction(population, population_fitnesses, p_c, p_m, training_data, validation_data):
    
    # select parents using selection operator
    selected_parents = parent_selection(population, population_fitnesses)
    print("No. of parents {}".format(len(selected_parents)))


    n_x = len(population[0]) # number of features
    #print("No. of features {}".format(n_x))

    all_children = []

    # perform crossover using selected parents
    #new_generation = []
    i = 0
    #print(int(len(selected_parents)/2) + 1)
    while (i < (int(len(selected_parents)/2) + 1)): # not iterating enough
        r = np.random.uniform(0, 1) 
        #print(r)
        if (r < p_c): # is this correct way to implement crossover probability
            #print("selected parents {} and {}".format(selected_parents[i], selected_parents[i+1]))
            children = bitstring_crossover(selected_parents[i], selected_parents[i+1], n_x) # NB should ensure this is not the same individual
             # apply mutation
            #print(children)
            children = [mutate(c, p_m) for c in children]
            #print(children)
            # determine whether offspring are accepted into population
            for c in children:
                all_children.append(c)
                parent_offspring_competition(c, selected_parents[i], selected_parents[i+1], population, population_fitnesses, training_data, validation_data)
    
        i += 2
    

    # add back the rest of the population not involved in reproduction
    return population


        
    
    # could use Boltzmann selection to decide if offspring replaces parent




def parent_offspring_competition(child, parent_1, parent_2, population, population_fitnesses, training_data, validation_data):

    print("Replacement strategy")
    print("Population size {}".format(len(population)))

    #fitness_p_1 = fitness_function(parent_1, training_data, validation_data)
    p1_string = str(parent_1).replace(" ", "")
    fitness_p_1= population_fitnesses[p1_string]
    #print(fitness_p_1)
    p2_string = str(parent_2).replace(" ", "")
    fitness_p_2= population_fitnesses[p2_string]
    #print(fitness_p_2)
    #fitness_p_2 = fitness_function(parent_2, training_data, validation_data)
    worst_parent = parent_1
    worst_parent_string = str(worst_parent).replace(" ", "")
    worst_fitness = fitness_p_1

    if(fitness_p_1 > fitness_p_2):
        worst_parent = parent_2
        worst_fitness = fitness_p_2
 

    # if child better than worst parent, remove worst parent and add child
    child_fitness = fitness_function(child, training_data, validation_data)
    #print(child_fitness)
    if (fitness_function(child, training_data, validation_data) > worst_fitness):
        #population.replace(worst_parent, child)
        #population = np.where(population == worst_parent, child, population)
        print("Improvement")
        if (worst_parent_string in population_fitnesses.keys()):
            #population = [ individual for individual in population if not (individual==worst_parent).all()]  # better way??
            print("Worst parent")
            for i in range(len(population)): # is the -1 necessary?
                #print(i)
                #print(population[i])
                #print(worst_parent)
                #print((population[i]==worst_parent).all())
                if((population[i]==worst_parent).all()):
                    population.pop(i)
                    break
        else:
            worst_individual_string = min(population_fitnesses, key= population_fitnesses.get)
            worst_individual = np.array(list(worst_individual_string[1:-1]))
            for i in range(len(population)): # is the -1 necessary?
                #print(i)
                #print(population[i])
                #print(worst_parent)
                #print((population[i]==worst_parent).all())
                print("Worst individual")
                if((population[i]==worst_individual).all()):
                    population.pop(i)
                    break

        population.append(child)
        #print("\n")
        #print(population)

    # else do nothing
    print("Population size {}".format(len(population)))

    return population



    




# potential stopping conditions 
# limited number of generations or function evaluation
# when population has converged e.g. no improvement observed

def genetic_algorithm(training_data, validation_data):

    
    n_x = 50 # 100
    gen_counter = 0 # generation counter
    max_gens = 2
    population_size = 10 # maybe 10*n_x but start small
    population = [np.random.randint(0, 2, n_x) for individual in range(population_size)] # initialize nx-dimensional population of given population size ns


    p_c = 0.8 # high prob of crossover
    p_m = 0.5 # lower prob of mutation

    population_fitnesses = {}

    while(gen_counter < max_gens): # choose stopping condition

        print("\n-------Generation {}--------".format(gen_counter))
        
        # calculate population fitnesses
        population_fitnesses.clear()
        for individual in population:
            fitness = fitness_function(individual, training_data, validation_data)
            print(fitness)
            # convert to string to use as dictionary key
            individual_string = str(individual)
            individual_string = individual_string.replace(" ", "")
            population_fitnesses[individual_string] = fitness # add dictionary entry

        # find new population through parent selection, crossover, mutation and a replacement strategy
        population = reproduction(population, population_fitnesses, p_c, p_m, training_data, validation_data) # new generation
        gen_counter += 1 # move to next generation

    for individual in population:
        fitness = fitness_function(individual, training_data, validation_data)
        print(fitness)
        individual_string = str(individual)
        individual_string = individual_string.replace(" ", "")
        #print(individual_string)
        population_fitnesses[individual_string] = fitness # add dictionary entry
    
    # what do we have at the end? the best individual in the population?
    best_individual = max(population_fitnesses, key=population_fitnesses.get)
    print(best_individual)
    print(population_fitnesses[best_individual])




def read_text_file_alt(filename):
    
    data = pd.read_csv(filename, header = None) 
    # first separate features and targets
    features_and_target = data[0].str.split(pat=" ", expand=True)
    data["Features"] = features_and_target[0]
    # split features into their own columns
    split_features = data["Features"].str.split(pat="", expand=True)
    for i in range(split_features.shape[1]-1):
        data[i] = split_features[i] # does this index the column??

    data.drop(columns =["Features"], inplace = True) # drop old features column
    data["Target"] = features_and_target[1]
    #print(data.head())
    return data


def main():
    print("\nFeature Optimization for ID3")
    print("--------Genetic Algorithm---------")

    print("\nReading in training dataset...")
    training_data = read_text_file_alt("./Data/Training_Data.txt") # can we make these datasets a class variable??


    print("Reading in Validation Data") 
    validation_data = read_text_file_alt("./Data/Validation_Data.txt")



    genetic_algorithm(training_data, validation_data)



if __name__ == "__main__":
    main()
