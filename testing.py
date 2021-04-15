# Hypothesis testing and metrics
from scipy.stats import mannwhitneyu


def one_sided_mann_whitney(tabu_data, ga_data):

    """
    Performs statistical hypothesis test for two sets of independent samples.

    Arguments:
    - Test accuracies obtained from feature subset selected using tabu search
    - Test accuracies obtained from feature subset selected using genetic algorithm

    Returns:
    - Statistic and p value from one-sided Mann-Whitney test.
    """

    stat, p = mannwhitneyu(tabu_data, ga_data)
    print("U Statistic: {}".format(stat))
    print("p-value: {}".format(p))

    return stat, p




def two_sided_mann_whitney(tabu_data, ga_data):

    """
    Performs statistical hypothesis test for two sets of independent samples.

    Arguments:
    - Test accuracies obtained from feature subset selected using tabu search
    - Test accuracies obtained from feature subset selected using genetic algorithm

    Returns:
    - Statistic and p value from Mann-Whitney test.
    """

    stat, p = mannwhitneyu(tabu_data, ga_data, alternative='two-sided')
    print("U Statistic: {}".format(stat))
    print("p-value: {}".format(p))

    return stat, p




def calculate_rates(targets, predictions):

    """
    Calculates number of true positives, false positives, true negatives and false negatives for the given predictions and targets.

    Arguments:
    - True labels for the dataset
    - Predicted labels for the dataset

    Returns:
    - Number of true positives, false positives, true negatives and false negatives 
    """

    TP = 0
    FP = 0
    TN = 0
    FN = 0

    # quicker way?

    for i in range(len(predictions)): 
        if (targets[i]==predictions[i]=="True"):
           TP += 1
        if (predictions[i]=="True" and targets[i]!=predictions[i]):
           FP += 1
        if (targets[i]==predictions[i]=="False"):
           TN += 1
        if (predictions[i]=="False" and targets[i]!=predictions[i]):
           FN += 1

    return(TP, FP, TN, FN)




def confusion_matrix(targets, predictions):

    """
    Prints a confusion matrix for the given predictions and targets.

    Arguments:
    - True labels for the dataset
    - Predicted labels for the dataset

    Returns:
    - Number of true positives, false positives, true negatives and false negatives 
    """


    TP, FP, TN, FN = calculate_rates(targets, predictions)

    print("\n")
    print("-----------------------------------")
    print("|                 Predicted       |")
    print("|---------------------------------|")
    print("|        |        True     False  |")
    #print("-------------------------")
    print("| Actual | True    {}      {}    |".format(TP, FN))
    print("|        | False   {}      {}    |".format(FP, TN))
    print("-----------------------------------")
    print("\n")
    
    
    print("Accuracy: {}".format(accuracy(TP, FP, TN, FN)))
    print("Recall: {}".format(recall(TP, FP, TN, FN)))
    print("Specificity: {}".format(true_negative_rate(TP, FP, TN, FN)))
    print("Precision: {}".format(precision(TP, FP, TN, FN)))
    print("\n")

    return TP, FP, TN, FN


# Accuracy - measures overall accuracy of the model classification
def accuracy(TP, FP, TN, FN):

    """
    Calculates the accuracy of the classifier from the number of true positives, false positives, true negatives and false negatives .

    Arguments:
    - Number of true positives (TP) 
    - Number of false positives (FP) 
    - Number of true negatives (TN) 
    - Number of false negatives (FN) 

    Returns:
    - Accuracy of the predictions calculated as (no. of correct predictions/total samples).
    """

    if ((TN + FP + FN + TP) ==0):
        return -1

    return (TN + TP)/(TN + FP + FN + TP)



# True Positive Rate (Sensitivity)
def recall(TP, FP, TN, FN):

    """
    Calculates the sensitivity or True Positive Rate (TPR).

    Arguments:
    - Number of true positives (TP) 
    - Number of false positives (FP) 
    - Number of true negatives (TN) 
    - Number of false negatives (FN) 

    Returns:
    - Sensitivity of the classifier calculated as (no. of correctly classified positives/total number of positives)
    """

    if ((TP+FN) ==0):
        return -1

    return TP/(TP+FN)


# True Negative Rate (Specificity)
def true_negative_rate(TP, FP, TN, FN):

    """
    Calculates the specificity or True Negative Rate (TNR).

    Arguments:
    - Number of true positives (TP) 
    - Number of false positives (FP) 
    - Number of true negatives (TN) 
    - Number of false negatives (FN) 

    Returns:
    - Specificity of the classifier calculated as (no. of correctly classified negatives/total number of negatives)
    """

    if ((TN+FP) ==0):
        return -1

    return TN/(TN+FP)



# Precision
def precision(TP, FP, TN, FN):

    """
    Calculates the precision of the classifier.

    Arguments:
    - Number of true positives (TP) 
    - Number of false positives (FP) 
    - Number of true negatives (TN) 
    - Number of false negatives (FN) 

    Returns:
    - Precision of the classifier calculated as (no. of correctly classified positives/total number of positive predictions)
    """

    if ((TP+FP) ==0):
        return -1

    return TP/(TP+FP)