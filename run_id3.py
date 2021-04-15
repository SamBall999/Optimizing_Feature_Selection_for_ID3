import numpy as np
import pandas as pd
from custom_id3 import ID3_Decision_Tree
from testing import calculate_rates, accuracy






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



def read_text_file_alt(filename):

    """
    Reads in the dataset from a text file and stores in a dataframe.

    Arguments:
    - Filename of text file containing dataset

    Returns:
    - Dataframe containing features and target values split into columns.
    """
    
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
    print("Feature Optimization for ID3")

    print("\nReading in training dataset...")
    training_data = read_text_file_alt("./Data/Training_Data.txt")


    print("\nFeature list:")
    feature_list =  training_data.columns.values[:-1]
    print(feature_list)
  
    # convert x and y data to numpy arrays
    x = np.array(training_data.drop("Target", axis=1).copy())
    y = np.array(training_data["Target"])

    # create decision tree with given subset
    tree = ID3_Decision_Tree(x, y) # intialize decision tree
    tree.id_3(feature_list) # build tree

   
    # verify if training accuracy is 100% (overfitting is occurring)
    predictions = tree.predict_batch(x)
    targets = training_data["Target"].values
    TP, FP, TN, FN = calculate_rates(targets, predictions)
    training_accuracy = accuracy(TP, FP, TN, FN)
    print("\nTraining Accuracy: {}".format(training_accuracy))

    print("\nReading in Test Data") 
    test_data = read_text_file_alt("./Data/Test_Data.txt")

    # calculate accuracy on the test set
    X_test = np.array(test_data.drop("Target", axis=1).copy())
    test_predictions = tree.predict_batch(X_test)
    test_targets = test_data["Target"].values
    TP, FP, TN, FN = calculate_rates(test_targets, test_predictions) # calculate number of true positives, false positives etc.
    test_accuracy = accuracy(TP, FP, TN, FN) 
    print("Test Accuracy: {}\n".format(test_accuracy))




    

if __name__ == "__main__":
    main()