#----TABU HILL SEARCH----#
import numpy as np
import pandas as pd
#from id3 import ID3_Decision_Tree 
from custom_id3 import ID3_Decision_Tree
from testing import calculate_rates, accuracy



# Feature selection in classication can be modeled as a combinatorial optimization problem.
# High number of features complicates the learning of the model, and, as a result, makes dicult the correct prediction of new observations.
# Feature selection problem in classification can be modeled as a combinatorial optimization problem because
# (i) consists of choosing a subset of features among N (2N possible subsets exist)
# (ii) because the quality of a subset may be evaluated (by the quality of the classification model constructed with this subset, for example) - is this the performance on the test or validation data?

# NB are we maximizing or minimizing?
# Motivate all design decisions
# NB to understand what search space looks like cost we can have any combination of features
# our problem is combinatorial and discrete since two classes A and B?? is this true?


# calculates fitness of current position
# we want to find the set of features which allows ID3 to obtain the best generalization ability.
# therefore do we use accuracy on the validation set?? (recall test set must in no way be used to create the model)
# i.e. given a candidate subset x -> build model using training set and evaluate accuracy on validation set and use this as fitness??
# NB rather make training and validation data global??
def objective_function(candidate_x, training_data, validation_data): 
    """
    Calculates the validation accuracy of the ID3 classifier for a given feature subset.

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
    #validation_predictions = tree.predict_batch(validation_data)
    X_val = np.array(validation_data.drop("Target", axis=1).copy())
    validation_predictions = tree.predict_batch(X_val)
    validation_targets = validation_data["Target"].values
    TP, FP, TN, FN = calculate_rates(validation_targets, validation_predictions) # calculate number of true positives, false positives etc.
    validation_accuracy = accuracy(TP, FP, TN, FN) # calculate validation accuracy
    print("Validation Accuracy: {}".format(validation_accuracy))

    # fitness is measured by validation accuracy
    fitness = validation_accuracy

    return fitness


# can we increase efficiency e.g. list comprehension??
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
    print("No. of neighbours: {}".format(len(neighbourhood)))
    for neighbour in neighbourhood:
        neighbour_string = str(neighbour)
        neighbour_string = neighbour_string.replace(" ", "")
        if ((neighbour_string not in tabu_list) and (np.sum(neighbour) != 0)): # only evaluate neighbours that are not on the tabu list and that have at least one feature
            fitness = objective_function(neighbour, training_data, validation_data)
            fitnesses[neighbour_string] = fitness # add dictionary entry
    #print("No. of neighbours: {}".format(len(fitnesses)))
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
    #x ^= 1 << index # but will this work on an array?
    x_new[index] = not x_new[index]
    #print(x_new)

    return x_new



# define all possible moves from current location
# possible the one flip neighbourhood? justify/explore other possibilities?
# possibly the k flip neighbourhood? -> all bitstrings that have k=2 bit flips (NB this is an extra hyperparameter)
def find_neighbourhood(current_x):
    """
    Creates a new candidate solution by flipping the bit at the given index.

    Arguments:
    - Candidate solution in the form of bitvector representing a feature subset
    - Index of the bit to be flipped

    Returns:
    - A new bitvector with the same values as the current solution except for one flipped bit at given index
    """

    #neighbourhood = [one_flip(current_x, i) for i in range(len(current_x))]
    
    #print("Neighbourhood")
    #subset_size = 20 # another hyperparameter!!
    neighbourhood = []
    for i in range(len(current_x)):
        x_copy = current_x
        #print(i)
        neighbourhood.append(one_flip(x_copy, i))

    #random_subset = np.random.randint(0, len(neighbourhood)-1, n_x) 
    return neighbourhood

    



# note consider that function evaluations are usually your computational currency - don't repeat them
# The procedure will select the best local candidate (although it has worse fitness than the sBest) in order to escape the local optimal.
# his process continues until the user specified stopping criterion is met, at which point, the best solution seen during the search process is returned
# keep a separate ultimate best but still allow to escape local optima
def tabu_search(training_data, validation_data):
    

    n_x = 30 #100

    # define starting point - bitstring of 0's and 1's of length n_x
    x_current = np.random.randint(0, 2, n_x) # choose randomnly from a distribution (all poss starting points)
    print("Starting position: {} \n".format(x_current))
    x_best = x_current # absolute best

    t = 0  # initialize number of iterations 
    t_max = 3 # max iterations

    tabu_list = [] # initialize tabu list
    tabu_list.append(str(x_current).replace(" ", "")) # add curent solution to tabu list
    print(tabu_list)
    max_tabu_size = 4 # hyperparameter

    while (t < t_max): # while max number of iterations has not been reached

        print("\n-------Iteration {}--------".format(t))
        
        # calculate fitness for all elements in neighbourhood
        neighbours = find_neighbourhood(x_current)
        fitnesses = get_fitnesses(neighbours, training_data, validation_data, tabu_list) # includes current x or not?
        #x_candidate = argmax(fitnesses) # candidate solutions must be mapped to fitnesses properly

        best_fitness = objective_function(x_best, training_data, validation_data)
        print("\nBest fitness: {}".format(best_fitness))
        x_candidate_string = max(fitnesses, key=fitnesses.get) # candidate solution with best fitness
        print("\nCandidate fitness: {}".format(fitnesses[x_candidate_string]))

        # always update x_current with x_candidate?? (has tabu list already been checked)
        x_current = np.array(list(x_candidate_string[1:-1]), dtype=int)
        print("New current x: {}".format(x_current))

        # update absolute best if fitness is greater
        if (fitnesses[x_candidate_string] > best_fitness):
            # need to covert back to array from string??
            if(x_candidate_string not in tabu_list): # is this check redundant?
                print(np.array(list(x_candidate_string[1:-1]), dtype=int))
                # change back to numpy array
                x_best = np.array(list(x_candidate_string[1:-1]), dtype=int)
                print("New best x: {}".format(x_best))

        tabu_list.append(x_candidate_string)
        if (len(tabu_list) > max_tabu_size):
            tabu_list = tabu_list[1:] # or use pop?
        print(tabu_list)
        t +=1 



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

