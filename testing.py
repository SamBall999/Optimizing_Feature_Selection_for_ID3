# Hypothesis testing and metrics



def calculate_rates(targets, predictions):
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

    return(TP, FP, TN, FN)


# Accuracy - measures overall accuracy of the model classification
def accuracy(TP, FP, TN, FN):


    if ((TN + FP + FN + TP) ==0):
        return -1

    return (TN + TP)/(TN + FP + FN + TP)



# True Positive Rate (Sensitivity)
def recall(TP, FP, TN, FN):

    if ((TP+FN) ==0):
        return -1

    return TP/(TP+FN)


# True Negative Rate (Specificity)
def true_negative_rate(TP, FP, TN, FN):

    if ((TN+FP) ==0):
        return -1

    return TN/(TN+FP)



# Precision
def precision(TP, FP, TN, FN):

    if ((TP+FP) ==0):
        return -1

    return TP/(TP+FP)