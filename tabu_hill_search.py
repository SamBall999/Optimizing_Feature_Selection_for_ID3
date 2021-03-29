#----TABU HILL SEARCH----#

# NB are we maximizing or minimizing?
# Motivate all design decisions
# NB to understand what search space looks like cost we can have any combination of features
# our problem is combinatorial and discrete since two classes A and B?? is this true?

# calculates fitness of current position
def objective_function():
    fitness = 1
    return fitness


# can we increase efficiency e.g. list comprehension??
def get_fitnesses(neighbourhood):
    fitnesses = {}
    for neighbour in neighbourhood:
        fitness = objective_function(neighbour)
        fitnesses[neighbour] = fitness # add dictionary entry
    return fitnesses


# define all possible moves from current location
# how do you know what you can move to???
def find_neighbourhood(current_x):
    neighbourhood = [2, 3]
    return neighbourhood



# check if local optimum reached
# local optimum reached when fitness of current x is strictly better than all of its neighbours
def check_local_optimum(current_x, fitnesses):
    is_local_optimum = 1
    current_fitness = objective_function(current_x)
    for neighbour in fitnesses.keys:
        if (fitnesses[neighbour] > current_fitness):
            is_local_optimum = 0
            break

    return is_local_optimum


# mark certain candidate solutions as forbidden - do we really need a method for this?
# make tabu list a class variable?
# CHOICE: Frequency based memory or Recency based - try both and decide? -motivate
def mark_tabu(tabu_list, element):
    tabu_list =  tabu_list.append(element)
    return tabu_list
    


# possibly implement just hill search first and then add tabu part
# note consider that function evaluations are usually your computational currency - don't repeat them
# note this is a 'best improvement (greedy)' hill climber
def tabu_search():
    
    local_optimum = 0 # set local optimum to be false to begin within

    # define starting point
    x_current = 0 # choose randomnly from a distribution (all poss starting points)?? - how many dimensions since could be many different sizes of groups of features???

    while (is_local_optimum(x_current) == 0): # while a local optimum has not been reached
        current_fitness = objective_function(x_current)
        # calculate fitness for all elements in neighbourhood
        neighbours = find_neighbourhood(x_current)
        fitnesses = get_fitnesses(neighbours) # includes current x or not?
        #x_candidate = argmax(fitnesses) # candidate solutions must be mapped to fitnesses properly
        x_candidate = max(fitnesses, key=fitnesses.get) # candidate solution with best fitness
        print(x_candidate)
        if (fitnesses[x_candidate] > current_fitness):
            x_current = x_candidate


