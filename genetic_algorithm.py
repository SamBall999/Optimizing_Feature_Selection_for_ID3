import numpy as np

#----GENETIC ALGORITHM (GA)----#

# Design decisions:
# 1. Type of selection operators - (i) select parents, (ii) select individuals to keep in population
# 2. Type of crossover
# 3. Type of mutation
# 4. How to encode chromosomes

# Control parameters
# 1. Population size
# 2. Mutation rate pm
# 3. Crossover rate pc
# 4. Dynamic mutation rates


# potential downside of GA compared to tabu -> more function evals


# ENCODING PROBLEM
# Each solution is encoded as a bitstring of length no_features
# Each bit in the bit string indicates whether that features is chosen or not

# many combinatorial problems can easily be encoded as bitstrings
def encode_chromosome():
    return chromosome

# calculates fitness of current position = ability of individual to survive
# The genetic algorithm is designed to optimize two objectives: maximize classification accuracy of the feature subset and minimize the number of features selected. 
# To do so, we define the following fitness function:
# F = w ∗ c(x) + (1 − w) ∗ (1/s(x)) # or just validation (generalisation) error??
def fitness_function(individual):
    w = 0.9
    # num features
    num_features = np.sum(individual) # no of 1's = no of features
    # how do we obtain accuracy? - need to call the decision tree using this subset of features!
    fitness = w*accuracy + (1 − w)*(1/num_features)
    return fitness





# also try single point or uniform crossover
def two_point_crossover_mask(n_x):

    crossover_points = np.random.randint(1, n_x, 2)# select crossover points
    e_1 = min(crossover_points)
    e_2 = max(crossover_points)
    print(crossover_points)
    print(e_1)
    print(e_2)
    mask = np.zeroes(n_x)
    j = e_1 + 1
    while (j < e_2):
        mask[j] = 1 # assign this range to 1's
    print(mask)
    return mask



# produce new offspring 
# also try single point crossover
def bitstring_crossover(parent_1, parent_2, n_x):

    # init children
    child_1 = parent_1
    child_2 = parent_2

    # get crossover mask
    mask = two_point_crossover_mask(n_x)

    # apply mask
    for j in range(1, n_x):
        if (mask[j] = 1):
            child_1[j] = parent_2[j]
            child_2[j] = parent_1[j]

    return [child_1, child_2]


# introduce new genetic material
# random mutation -> apply p_m to each bit
# is the chromosome a string or an array?
def mutate(chromosome, p_m):

    for i in range(len(chromosome)):
        r = np.random.rand.uniform(0, 1)
        if (r < p_m):
            chromosome[i] = ~chromosome[i] 

    return chromosome



def proportional_probability(individual, population):
    # NB scaled fitnesses??
    fitness = fitness_function(individual) # scaled?
    all_fitnesses = [fitness_function(i) for i in population]
    fitness_sum = np.sum(all_fitnesses) # sum of fitnesses across all individuals
    prob = fitness/fitness_sum # probability that individual x_i will be selected
    return prob



# Roulette wheel selection -> can also try more balanced variant: stochastic universal sampling
def roulette_wheel_selection(population):
    i = 1 # first chromosome
    p_i = proportional_probability(i, population) # probability of first chromosome
    summation = p_i # points to the top of the first region
    r = np.random.rand.uniform(0, 1) # generate random number sampled uniformly between 0 and 1 - do we need to include 1 somehow??
    while (summation < r):
        i = i+1
        summation = summation + proportional_probability(i, population)
    return i # selected individual/parent


# choose fitter individuals for reproduction -> should also try rank-based at some point
# (i) Roulette wheel selection - with or without replacement??
def parent_selection(population):
    # how many parents to select?? - for now we select 2 parents -> 2 children
    # therefore select no. of parents = half of pop?
    num_parents = len(population)/2
    p = {} # list of parents
    for i in range(num_parents):
        p.append(roulette_wheel_selection(population))

    return p # selected parents




def reproduction(population, p_c, p_m):
    
    # select parents using selection operator
    selected_parents = parent_selection(population)

    # perform crossover using selected parents
    new_generation = {}
    for i in range(len(selected_parents)/2):
        r = np.random.rand.uniform(0, 1) 
        if (r < p_c): # is this correct way to implement crossover probability
            children = bitstring_crossover(selected_parents[i], selected_parents[i+1], n_x) # NB should ensure this is not the same individual
             # apply mutation
            children = [mutate(children[c], p_m) for c in children]
            # determine whether offspring are accepted into population
            new_generation.append(parent_offspring_competition(c, selected_parents[i], selected_parents[i+1]) for c in children)
        i += 2
    
    return new_generation

        
    
    # could use Boltzmann selection to decide if offspring replaces parent




def parent_offspring_competition(child, parent_1, parent_2):

    retained_individuals = {}
    worst_parent = parent_1
    if(fitness_function(parent_1) > fitness_function(parent_2)):
        worst_parent = parent_2
    else:
        worst_parent = parent_1

    if (fitness_function(child) > worst_parent):
        retained_individuals.append(child)
    else:
        retained_individuals.append(worst_parent)

    return retained_individuals


    




# potential stopping conditions 
# limited number of generations or function evaluation
# when population has converged e.g. no improvement observed

def genetic_algorithm():

    
    n_x = 100 # no. of features
    gen_counter = 0 # generation counter
    max_gens = 50
    population_size = 10*n_x
    #individual = np.random.randint(0, 2, n_x) # upper bound is excluded
    population = [np.random.randint(0, 2, n_x) for individual in population_size] # initialize nx-dimensional population of given population size ns
    # discrete - either 0 or 1 only?? - each member of the population should be a bitstring of length n_x
    print(population)

    while(gen_counter < max_gens): # choose stopping condition
        population_fitnesses = {}
        for individual in population:
            fitness = fitness_function(individual)
            population_fitnesses[individual] = fitness # add dictionary entry

        #parents = parent_selection()
        #offspring = crossover() # where does mutation fit in?
        population = reproduction(population, p_c, p_m) # new generation
        #new_generation = replacement_strategy(parents, offspring)
        gen_counter += 1


