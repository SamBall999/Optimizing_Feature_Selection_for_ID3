import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
import pandas as pd
import numpy as np
from custom_id3 import ID3_Decision_Tree
from testing import calculate_rates, accuracy, confusion_matrix, one_sided_mann_whitney, two_sided_mann_whitney



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



def plot_performance_dist(data, c, alg_label, fig_name):
    
    #ax = sns.distplot(tabu_data, color = "cornflowerblue", label = "Tabu Search")
    ax = sns.distplot(data, color = c, label = alg_label)
    plt.xlabel("Test Accuracy")
    plt.ylabel("Density")
    plt.savefig(fig_name, dpi = 300)
    plt.show()


def plot_all(data_1, data_2, c_1, c_2, alg_label_1,  alg_label_2, fig_name):
    
    ax = sns.distplot(data_1, color = c_1, label = alg_label_1)
    plt.xlabel("Test Accuracy")
    plt.ylabel("Density")
    ax = sns.distplot(data_2, color = c_2, label = alg_label_2)
    plt.legend()
    plt.savefig(fig_name, dpi = 300)
    plt.show()


def plot_confusion_matrix(array, colourmap):

    df_cm = pd.DataFrame(array, index=["True", "False"],  columns=["True", "False"])
    sns.set(font_scale=1.2) # for label size
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    sns.heatmap(df_cm, annot=True, cmap = colourmap, annot_kws={"size": 18}, fmt="d") # font size #light=1
    plt.show()



def classification_performance(candidate_x, training_data, test_data):

    feature_indices = [(i+1) for i in range(len(candidate_x)) if candidate_x[i]==1]
    feature_list =  training_data.columns.values[feature_indices] 
  
    # convert x and y data to numpy arrays
    x = np.array(training_data.drop("Target", axis=1).copy())
    y = np.array(training_data["Target"])

    # create decision tree with given subset
    tree = ID3_Decision_Tree(x, y) # intialize decision tree
    tree.id_3(feature_list) # build tree


    # calculate accuracy on the test set
    X_test = np.array(test_data.drop("Target", axis=1).copy())
    test_predictions = tree.predict_batch(X_test)
    test_targets = test_data["Target"].values
    TP, FP, TN, FN = calculate_rates(test_targets, test_predictions) # calculate number of true positives, false positives etc.
    test_accuracy = accuracy(TP, FP, TN, FN) 
    print("Test Accuracy: {}".format(test_accuracy))
    confusion_matrix(test_targets, test_predictions)

    return TP, FP, TN, FN 



def main():

    # Insert performance data - set of test accuracies obtained by each algorithm
    tabu_data = [0.919, 0.9095, 0.8755, 0.904, 0.9265, 0.858, 0.908, 0.87, 0.875, 0.914, 0.921, 0.907, 0.9155, 0.8755, 0.923, 0.923, 0.9145, 0.9155, 0.882, 0.926, 0.919, 0.901, 0.9265, 0.9095, 0.9165, 0.883, 0.903, 0.9165, 0.897, 0.91]
    #tabu_data = [0.919, 0.9095, 0.8755, 0.904, 0.9265, 0.858, 0.908, 0.87, 0.875, 0.914, 0.921, 0.907, 0.9155, 0.8755, 0.923]
    ga_data = [0.871, 0.7915, 0.906, 0.877, 0.8965, 0.8865, 0.8585, 0.8605, 0.845, 0.8755, 0.908, 0.879, 0.8935, 0.839, 0.864, 0.8885, 0.8385, 0.8755, 0.8995, 0.8155, 0.858, 0.8425, 0.8815, 0.885, 0.889, 0.888, 0.8765, 0.831, 0.917, 0.862]
    #ga_data = [0.871, 0.7915, 0.906, 0.877, 0.8965, 0.8865, 0.8585, 0.8605, 0.845, 0.8755, 0.908, 0.879, 0.8935, 0.839, 0.864] # note this does not include 0.917


    # Plot performance distributions
    plot_performance_dist(tabu_data, "cornflowerblue", "Tabu Search", "tabu_performance_distribution.png")
    plot_performance_dist(ga_data, "lightcoral", "GA", "ga_performance_distribution.png")
    plot_all(tabu_data, ga_data, "cornflowerblue", "lightcoral", "Tabu Search", "GA", "performance_distributions.png" )



    # Calculate U statistic for one sided Mann Whitney U Test       
    print("\nOne sided Mann Whitney U Test")
    mann_whitney_result, p = one_sided_mann_whitney(ga_data, tabu_data)

    # Calculate U statistic for two sided Mann Whitney U Test
    print("\nTwo sided Mann Whitney U Test")
    mann_whitney_result, p = two_sided_mann_whitney(ga_data, tabu_data)



    # Calculate confusion matrix for best subset obtained from Tabu Search 

    print("\nReading in training dataset...")
    training_data = read_text_file_alt("./Data/Training_Data.txt") # read in the training data

    print("Reading in validation dataset...") 
    test_data = read_text_file_alt("./Data/Test_Data.txt")  # read in the validation data


    tabu_best_string = "[0101101010111110101010001001110011001110001110001111011011011010111101000000100101011100001010000010]" # best subset
    candidate_x_tabu = np.array(list(tabu_best_string[1:-1]), dtype=int)
    TP, FP, TN, FN = classification_performance(candidate_x_tabu, training_data, test_data)

    #tabu_array = np.array([[672, 69],
         #[78, 1181]])

    tabu_array = np.array([[TP, FN],
         [FP, TN]])

    plot_confusion_matrix(tabu_array, 'Blues')




    # Calculate confusion matrix for best subset obtained from GA

    ga_best_string = "[1001111111100011111100011011100000111011110110101111000010001011011111010101001010001000010001000110]"
    #ga_second_best_string = "[1101101111111011111110111101110101000010010010001010111011111001010000010111010100100011101111010010]"
    candidate_x_ga = np.array(list(ga_best_string[1:-1]), dtype=int)

    TP, FP, TN, FN = classification_performance(candidate_x_ga, training_data, test_data)

    ga_array = np.array([[TP, FN],
         [FP, TN]])

    plot_confusion_matrix(ga_array, sns.cubehelix_palette(light=0.95, as_cmap=True))



if __name__ == "__main__":
    main()


