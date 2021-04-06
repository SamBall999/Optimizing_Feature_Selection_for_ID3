# main class

import numpy as np
import pandas as pd
from id3 import ID3_Decision_Tree # better way to do this?
from testing import confusion_matrix
#from tabu_hill_search import tabu_search
#from genetic_algorithm import genetic_algorithm
#from statistical_tests import hypothesis_test

# Design decisions
# 1. How to compare algorithms properly using statistical tests
# - from lectures: often Mann-Whitney 
# - (usually a two-sample situation since comparing algs A and B, usually performance results are not normally distributed but must check for this first, usually independent samples since starting positions random)
# - choices: Chi-sq, Mann-Whitney, Median, Kolmogorov
# - from paper: Wilcoxon test - implies samples are paired??
# 2. How many times to run algorithms


# how best to store these features and target?
def read_text_file(filename):
    data_file = open(filename,"r")
    data = []
    for line in data_file.readlines():
        data.append(line)
    print(len(data)) # 5000 training examples
    print(len(data[0])) # 108 -> 100 features, letters in True, spaces and \n -> encode true and false and 1 and 0?
    print(data[0])
    data_file.close()
    return data


# alternative - use pandas??
def read_text_file_alt(filename):
    
    data = pd.read_csv(filename, header = None) 
    # first separate features and targets
    features_and_target = data[0].str.split(pat=" ", expand=True)
    data["Features"] = features_and_target[0]
    data["Target"] = features_and_target[1]
    # split features into their own columns
    split_features = data["Features"].str.split(pat="", expand=True)
    for i in range(split_features.shape[1]-1):
        data[i] = split_features[i] # does this index the column??

    data.drop(columns =["Features"], inplace = True) # drop old features column
    print(data.head())
    return data




def main():
    print("Feature Optimization for ID3")

    print("\nReading in training dataset...")
    training_data = read_text_file_alt("./Data/Training_Data.txt")

    print(len(training_data[training_data["Target"]=="True"]))
    print(len(training_data[training_data["Target"]=="False"]))

    tree = ID3_Decision_Tree() # intialize decision tree
    tree.id_3(training_data.iloc[:, : 40]) # test with a smaller subset to begin with, builds tree
    # how do we test or print this tree?
    #tree.print_tree()
   
    predictions = tree.predict_batch(training_data)
    print(predictions)
    targets = training_data["Target"].values
    print(targets)
    #print("\nPredicted value: {}".format(prediction))
    #print("Actual value: {}".format(target))
    confusion_matrix(targets, predictions)


    print("Reading in Validation Data") # or should it be test data??
    validation_data = read_text_file_alt("./Data/Validation_Data.txt")

    validation_predictions = tree.predict_batch(validation_data)
    #print(validation_predictions)
    validation_targets = validation_data["Target"].values
    #print(validation_targets)
    confusion_matrix(validation_targets, validation_predictions)


    

if __name__ == "__main__":
    main()