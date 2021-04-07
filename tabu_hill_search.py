#----TABU HILL SEARCH----#
import numpy as np
import pandas as pd
from id3 import ID3_Decision_Tree # better way to do this?
from testing import calculate_rates, accuracy

# Are these both wrapper methods???

# Feature selection in classication can be modeled as a combinatorial optimization problem.
# High number of features complicates the learning of the model, and, as a result, makes dicult the correct prediction of new observations.
# Feature selection problem in classification can be modeled as a combinatorial optimization problem because
# (i) consists of choosing a subset of features among N (2N possible subsets exist)
# (ii) because the quality of a subset may be evaluated (by the quality of the classification model constructed with this subset, for example) - is this the performance on the test or validation data?

# NB are we maximizing or minimizing?
# Motivate all design decisions
# NB to understand what search space looks like cost we can have any combination of features
# our problem is combinatorial and discrete since two classes A and B?? is this true?



# Design decisions
# 1. How to encode problem/solution
# 2. How to define neighbourhood
# 3. What to use as fitness function??



# ENCODING PROBLEM
# Each solution is encoded as a bitstring of length no_features
# Each bit in the bit string indicates whether that features is chosen or not

# calculates fitness of current position
# we want to find the set of features which allows ID3 to obtain the best generalization ability.
# therefore do we use accuracy on the validation set?? (recall test set must in no way be used to create the model)
# i.e. given a candidate subset x -> build model using training set and evaluate accuracy on validation set and use this as fitness??
# NB rather make training and validation data global??
def objective_function(candidate_x, training_data, validation_data): 

    tree = ID3_Decision_Tree() # intialize decision tree

    # TO DO: need to process the bitstring candidate x into the indices of which features to include
    # convert from string to array if it is not in array form?
    print("\nCandidate x: {}".format(candidate_x))


    # find positions of 1's in the bitstring
    feature_indices = [(i+1) for i in range(len(candidate_x)) if candidate_x[i]==1]# be careful whether it is the index or the name of the feature
    feature_indices.append(-1) # append -1 in order to get Target column
    # +1 added to account for features starting at 1 not 0 

    #print(training_data.iloc[: , feature_indices]) # need to also add target column!!

    tree.id_3(training_data.iloc[:, feature_indices]) # how to input the correct feature subset??
   
    predictions = tree.predict_batch(training_data)
    targets = training_data["Target"].values
    TP, FP, TN, FN = calculate_rates(targets, predictions)
    training_accuracy = accuracy(TP, FP, TN, FN)
    print("Training Accuracy: {}".format(training_accuracy))

    validation_predictions = tree.predict_batch(validation_data)
    validation_targets = validation_data["Target"].values
    TP, FP, TN, FN = calculate_rates(validation_targets, validation_predictions)
    validation_accuracy = accuracy(TP, FP, TN, FN)
    print("Validation Accuracy: {}".format(validation_accuracy))

    fitness = validation_accuracy

    return fitness


# can we increase efficiency e.g. list comprehension??
def get_fitnesses(neighbourhood, training_data, validation_data):
    fitnesses = {}
    for neighbour in neighbourhood:
        fitness = objective_function(neighbour, training_data, validation_data)
        neighbour_string = str(neighbour)
        #print(neighbour_string) # need to convert neighbour array to a string in order to be a key
        #print(fitness)
        fitnesses[neighbour_string] = fitness # add dictionary entry
    return fitnesses




def one_flip(x, index):

    #print(x)
    x_new = np.empty_like (x)
    x_new[:] = x # new object
    #x ^= 1 << index # but will this work on an array?
    x_new[index] = not x_new[index]
    #print(x_new)

    return x_new



# define all possible moves from current location
# how do you know what you can move to???
# possible the one flip neighbourhood? justify/explore other possibilities?
# possibly the k flip neighbourhood? -> all bitstrings that have k=2 bit flips (NB this is an extra hyperparameter)
# NB why is current_x changing?????
def find_neighbourhood(current_x):

    print("Neighbourhood")
    neighbourhood = []
    for i in range(len(current_x)):
        x_copy = current_x
        #print(i)
        neighbourhood.append(one_flip(x_copy, i))

    return neighbourhood



# check if local optimum reached
# local optimum reached when fitness of current x is strictly better than all of its neighbours
def is_local_optimum(current_x, fitnesses):
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
    # frequency
    if(frequency(element) > 2):
        tabu_list =  tabu_list.append(element)

    # recency - sliding window of recently visited entities
    tabu_list =  tabu_list.append(element) # just call this method on the most recent elements?
    return tabu_list
    


# possibly implement just hill search first and then add tabu part
# note consider that function evaluations are usually your computational currency - don't repeat them
# note this is a 'best improvement (greedy)' hill climber - should we use first improvement to save function evals??
def tabu_search(training_data, validation_data):
    
    local_optimum = 0 # set local optimum to be false to begin within
    n_x = 10 # number of features = 100 (for now we use 10)

    # define starting point - bitstring of 0's and 1's of length n_x
    x_current = np.random.randint(0, 2, n_x) # choose randomnly from a distribution (all poss starting points)
    print("Starting position: {} \n".format(x_current))

    neighbours = find_neighbourhood(x_current)
    fitnesses = get_fitnesses(neighbours, training_data, validation_data) # initial neighbouring fitnesses

    while (is_local_optimum(x_current, fitnesses) == 0): # while a local optimum has not been reached - is this the correct stopping condition?
        current_fitness = objective_function(x_current, training_data, validation_data)
        print("\nCurrent fitness: {}".format(current_fitness))
        # calculate fitness for all elements in neighbourhood
        neighbours = find_neighbourhood(x_current)
        fitnesses = get_fitnesses(neighbours, training_data, validation_data) # includes current x or not?
        #x_candidate = argmax(fitnesses) # candidate solutions must be mapped to fitnesses properly
        x_candidate = max(fitnesses, key=fitnesses.get) # candidate solution with best fitness
        print("\nCandidate fitness: {}".format(fitnesses[x_candidate]))
        if (fitnesses[x_candidate] > current_fitness):
            # need to covert back to array from string??
            if(x_candidate not in tabu_list):
                x_current = x_candidate
                print("New current x: {}".format(x_current))



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
    print(data.head())
    return data



def main():
    print("Feature Optimization for ID3")
    print("--------Tabu Search---------")

    print("\nReading in training dataset...")
    training_data = read_text_file_alt("./Data/Training_Data.txt") # can we make these datasets a class variable??


    print("Reading in Validation Data") 
    validation_data = read_text_file_alt("./Data/Validation_Data.txt")



    tabu_search(training_data, validation_data)


    

if __name__ == "__main__":
    main()