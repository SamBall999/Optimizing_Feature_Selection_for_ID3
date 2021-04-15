#----TABU SEARCH----#

import numpy as np
import pandas as pd
from custom_id3 import ID3_Decision_Tree
from testing import calculate_rates, accuracy





def objective_function(candidate_x, training_data, validation_data): 

    """
    Calculates the fitness of the current position based on the validation accuracy of the ID3 classifier for a given feature subset.

    Arguments:
    - Candidate solution in the form of bitvector representing a feature subset
    - Set of training data used to build ID3 decision tree
    - Set of validation data used to measure accuracy on unseen data

    Returns:
    - Validation accuracy representing a measure of generalisation ability.
    """

    # get selected features from the bitvector encoding
    feature_indices = [(i+1) for i in range(len(candidate_x)) if candidate_x[i]==1]
    feature_list =  training_data.columns.values[feature_indices] 
  
    # convert x and y data to numpy arrays
    x = np.array(training_data.drop("Target", axis=1).copy())
    y = np.array(training_data["Target"])

    # create decision tree with given subset
    tree = ID3_Decision_Tree(x, y) # intialize decision tree
    tree.id_3(feature_list) # build tree

   
    # verify if training accuracy is 100% (overfitting is occurring)
    #predictions = tree.predict_batch(x)
    #targets = training_data["Target"].values
    #TP, FP, TN, FN = calculate_rates(targets, predictions)
    #training_accuracy = accuracy(TP, FP, TN, FN)
    #print("Training Accuracy: {}".format(training_accuracy))

    # calculate accuracy on the validation set
    X_val = np.array(validation_data.drop("Target", axis=1).copy())
    validation_predictions = tree.predict_batch(X_val)
    validation_targets = validation_data["Target"].values
    TP, FP, TN, FN = calculate_rates(validation_targets, validation_predictions) # calculate number of true positives, false positives etc.
    validation_accuracy = accuracy(TP, FP, TN, FN) # validation accuracy measures generalisation ability
    print("Validation Accuracy: {}".format(validation_accuracy))

    # fitness is measured by validation accuracy and should be maximized
    fitness = validation_accuracy

    return fitness



def get_fitnesses(neighbourhood, training_data, validation_data, tabu_list):

    """
    Calculates the value of the objective function for all solutions in the neighbourhood.

    Arguments:
    - Set of neighbourhood solutions in the form of bitvectors
    - Set of training data used to build ID3 decision tree
    - Set of validation data used to measure accuracy on unseen data
    - List of tabu elements to verify if a solution should be considered

    Returns:
    - A dictionary mapping each solution in the nieghbourhood to their objective function value.
    """

    fitnesses = {}
    #print("No. of neighbours: {}".format(len(neighbourhood)))
    for neighbour in neighbourhood:
        neighbour_string = str(neighbour).replace(" ", "") # remove spaces
        neighbour_string = neighbour_string.replace("\n", "") # remove newline if string is very long
        if ((neighbour_string not in tabu_list) and (np.sum(neighbour) != 0)): # only evaluate neighbours that are not on the tabu list and that have at least one feature
            fitness = objective_function(neighbour, training_data, validation_data) # calculate fitness for given neighbour
            fitnesses[neighbour_string] = fitness # add dictionary entry
 
    return fitnesses






def one_flip(x, index):

    """
    Creates a new candidate solution by flipping the bit at the given index.

    Arguments:
    - Candidate solution in the form of bitvector representing a feature subset
    - Index of the bit to be flipped

    Returns:
    - A new bitvector with the same values as the current solution except for one flipped bit at given index
    """

    #print(x)
    x_new = np.empty_like (x)
    x_new[:] = x # new object
    x_new[index] = not x_new[index] # flip bit at desired index
    #print(x_new)

    return x_new



def k_flip(x, indices):

    """
    Creates a new candidate solution by flipping the bits at the given indices.

    Arguments:
    - Candidate solution in the form of bitvector representing a feature subset
    - Indices of the bits to be flipped

    Returns:
    - A new bitvector with the same values as the current solution except for k flipped bits at given indices
    """

    #print(x)
    x_new = np.empty_like (x)
    x_new[:] = x # new object
    for index in indices:
        x_new[index] = not x_new[index] # flip bit at desired index
    #print(x_new)

    return x_new




def find_neighbourhood(current_x):

    """
    Generates a set of neighbourhood solutions based on the one flip neighbourhood.

    Arguments:
    - Current candidate solution in the form of bitvector representing a feature subset

    Returns:
    - A set of neighbourhood bitvectors representing closely related feature subsets
    """
    
    subset_size = 30 # another hyperparameter!!
    neighbourhood = []
    for i in range(len(current_x)):
        x_copy = current_x
        #neighbourhood.append(one_flip(x_copy, i))
        interval = int(len(current_x)/3)
        indices = [i, i+interval, i+(2*interval)]
        for j in range(len(indices)):
          if (indices[j] > len(current_x)-1):
            indices[j] = indices[j] - len(current_x)

        neighbourhood.append(k_flip(x_copy, indices))
    
    random_subset = list(np.random.randint(0, len(neighbourhood)-1, subset_size)) 
    #print(random_subset)
    neighbourhood_subset = [neighbourhood[i] for i in random_subset]

    #return neighbourhood
    return neighbourhood_subset

    




def tabu_search(training_data, validation_data):

    """
    Performs tabu search in order to find the feature subset that maximizes validation accuracy.

    Arguments:
    - Set of training data used to build ID3 decision tree
    - Set of validation data used to measure accuracy on unseen data

    Returns:
    - Feature subset with highest validation accuracy obtained during the search.
    """
    

    n_x = 100 # number of features

    # define starting point - bitstring of 0's and 1's of length n_x
    x_current = np.random.randint(0, 2, n_x) # choose randomnly from a distribution (all poss starting points)
    print("Starting position: {} \n".format(x_current))
    x_best = x_current # initialize absolute best

    t = 0  # initialize number of iterations 
    t_max = 10 # max iterations

    tabu_list = [] # initialize tabu list
    x_current_string = str(x_current).replace(" ", "")
    tabu_list.append(x_current_string.replace("\n", ""))  # add curent solution to tabu list
    #print(tabu_list)
    max_tabu_size = 8 # set maximum tabu list size (hyperparameter)

    # while max number of iterations has not been reached
    while (t < t_max):

        print("\n-------Iteration {}--------".format(t))
        
        # calculate fitness for all elements in neighbourhood
        neighbours = find_neighbourhood(x_current) # find neighbourhood of current candidate solution
        fitnesses = get_fitnesses(neighbours, training_data, validation_data, tabu_list) # calculate fitnesses of all bitvectors in the neighbourhood

        # calculate the fitness of the current best candidate solution
        best_fitness = objective_function(x_best, training_data, validation_data)
        print("\nBest fitness: {}".format(best_fitness))

        # find the candidate solution in the neighbourhood with the best fitness
        x_candidate_string = max(fitnesses, key=fitnesses.get) # candidate solution with best fitness
        print("\nCandidate fitness: {}".format(fitnesses[x_candidate_string]))

        # (has tabu list already been checked)
        # select the best local candidate, even if it has worse fitness than x_best, in order to escape local optima
        x_current = np.array(list(x_candidate_string[1:-1]), dtype=int) 
        print("New current x: {}".format(x_current))

        # update absolute best if candidate fitness is greater than best fitness
        if (fitnesses[x_candidate_string] > best_fitness):
            if(x_candidate_string not in tabu_list): # is this check redundant?
                # change back to numpy array
                x_best = np.array(list(x_candidate_string[1:-1]), dtype=int)
                print("New best x: {}".format(x_best))

        # update tabu list with current solution
        tabu_list.append(x_candidate_string)
        # check if max tabu list size exceeded
        if (len(tabu_list) > max_tabu_size):
            tabu_list = tabu_list[1:] # remove the first element of the list
        #print(tabu_list)
        t +=1  # move to next iteration

    return x_best # the best solution seen during the search process is returned



def read_text_file_alt(filename):

    """
    Reads in the dataset from a text file and stores in a dataframe.

    Arguments:
    - Filename of text file containing dataset

    Returns:
    - Dataframe containing features and target values split into columns.
    """
    
    data = pd.read_csv(filename, header = None) # read data into a dataframe
    features_and_target = data[0].str.split(pat=" ", expand=True) # separate features and targets

    # split features into their own columns
    data["Features"] = features_and_target[0]
    split_features = data["Features"].str.split(pat="", expand=True) 
    for i in range(split_features.shape[1]-1):
        data[i] = split_features[i] 

    data.drop(columns =["Features"], inplace = True) # drop old features column
    data["Target"] = features_and_target[1] # create target column
    #print(data.head())
    return data



def main():

    """
    Runs the tabu search algorithm for the given training and validation dataset.

    Arguments:


    """

    print("Feature Optimization for ID3")
    print("--------Tabu Search---------")

    print("\nReading in training dataset...")
    training_data = read_text_file_alt("./Data/Training_Data.txt") # read in the training data


    print("Reading in validation dataset...") 
    validation_data = read_text_file_alt("./Data/Validation_Data.txt")  # read in the validation data


    # run tabu search on the given datasets
    best_solution = tabu_search(training_data, validation_data)
    best_solution_string = str(best_solution).replace(" ", "").replace("\n", "")
    print(best_solution_string)
    best_fitness = objective_function(best_solution, training_data, validation_data)
    print("Best Validation Fitness: {}".format(best_fitness))


    print("Reading in Test Data") 
    test_data = read_text_file_alt("./Data/Test_Data.txt")


    test_fitness = objective_function(best_solution, training_data, test_data) # check that this is correct
    print("Test Fitness: {}".format(test_fitness))


    

if __name__ == "__main__":
    main()

