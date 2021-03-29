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



# many combinatorial problems can easily be encoded as bitstrings
def encode_chromosome():
    return chromosome

# calculates fitness of current position = ability of individual to survive
def fitness_function(individual):
    fitness = 1
    return fitness


# produce new offspring 
def crossover():
    return offspring


# introduce new genetic material
def mutation():
    return mutated_chromosome


# choose fitter individuals for reproduction
def parent_selection():
    return parents


# decide which individuals to retain
def replacement_strategy():
    return new_generation




def genetic_algorithm(population_size):

    

    gen_counter = 0 # generation counter
    population = 0 # initialize nx-dimensional population of given population size ns

    while(stopping_condition= False): # choose stopping condition
        population_fitnesses = {}
        for individual in population:
            fitness = fitness_function(individual)
            population_fitnesses[individual] = fitness # add dictionary entry

        parents = parent_selection()
        offspring = crossover() # where does mutation fit in?
        new_generation = replacement_strategy(parents, offspring)
        gen_counter += 1


