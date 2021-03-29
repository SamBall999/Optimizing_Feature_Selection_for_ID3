import numpy as np

#--- Implementing ID3 Decision Tree ---#

# Iterative Dichotomiser 3 - iteratively dichotomizes(divides) features into two or more groups at each step

# Top-down greedy approach to build a decision tree 
# - top-down approach means that we start building the tree from the top 
# - greedy approach means that at each iteration we select the best feature at the present moment to create a node

# Most generally ID3 is only used for classification problems with nominal features only.

# -- Information Gain --#
# ID3 uses Information Gain or just Gain to find the best feature.
# Information Gain calculates the reduction in the entropy and measures how well a given feature separates or classifies the target classes.
# The feature with the highest Information Gain is selected as the best one.
# Entropy is the measure of disorder and the Entropy of a dataset is the measure of disorder in the target feature of the dataset.

# In the case of binary classification (where the target column has only two types of classes) entropy is 0 if all values in the target column are homogenous(similar) and will be 1 if the target column has equal number values for both the classes.
# We denote our dataset as S, entropy is calculated as:
# Entropy(S) = - ∑ pᵢ * log₂(pᵢ) ; i = 1 to n



# test this function in isolation
# our classes are A and B
# NB it is entropy of target -> True and False
def entropy(data):
    sum = 0
    target = data["Target"] # target us last column - how to store our data? array of columns? but then how do we index rows?
    classes = np.unique(target)
    #print(classes)
    # iterate over classes (will be 2 in our case)
    for i in classes:
        #print(len(data[data["Target"]==i]))
        #print(len(target))
        prob_i = len(data[data["Target"]==i])/len(target) # probability of class ‘i’ = “number of rows with class i in the target column” to the “total number of rows” in the dataset
        log_2_p_i  = np.log(prob_i) / np.log(2) # check this
        sum += prob_i*log_2_p_i

    #print(sum)
    entropy = sum*-1
    #print(entropy)
    return entropy



# information gain for specified feature column
# IG(S, A) = Entropy(S) - ∑((|Sᵥ| / |S|) * Entropy(Sᵥ))
# where Sᵥ is the set of rows in S for which the feature column A has value v, |Sᵥ| is the number of rows in Sᵥ and likewise |S| is the number of rows in S.
# how does this differ for each feature??
def get_information_gain(data, feature):
    total_entropy = entropy(data) # entropy of target
    classes = np.unique(data[feature])
    #print(classes)
    summation = 0
    for v in classes:
        set_v = data[data[feature]==v]# set of rows in S for which the feature column A has value v
        entropy_set_v = entropy(set_v)
        scaling_factor =  len(set_v)/len(data[feature])
        #print(len(set_v))
        #print(len(data[feature]))
        #print(scaling_factor)
        summation += scaling_factor*entropy_set_v
    
    #print(summation)
    info_gain = total_entropy - summation
    return info_gain



# how to check if correct info_gain is being calculated??




# 1. Calculate the Information Gain of each feature.
# 2. Considering that all rows don’t belong to the same class, split the dataset S into subsets using the feature for which the Information Gain is maximum.
# 3. Make a decision tree node using the feature with the maximum Information gain.
# 4. If all rows belong to the same class, make the current node as a leaf node with the class as its label.
# 5. Repeat for the remaining features until we run out of all features, or the decision tree has all leaf nodes.
def id_3(data):
    information_gains = [] # use dictionary instead?
    # iterate over features
    no_features = data.shape[1] - 2 # subtract target row and index row
    print(no_features)
    for i in range(1, no_features+1):
        info_gain = get_information_gain(data, i) # or should i be the feature column?? how best to index everything?
        information_gains.append(info_gain)
    
    # split dataset into subsets using the feature for which info gain is max
    # info gain has to do with how it is splitting the target True and Falses! NB to understand for theory part of report
    max_gain =  np.max(information_gains)
    print(max_gain)
    max_gain_feature = np.argmax(information_gains)
    print(max_gain_feature)
    print(len(data[data[4]=='A']))


    min_gain =  np.min(information_gains)
    print(min_gain)
    min_gain_feature = np.argmin(information_gains)
    print(min_gain_feature)
    print(len(data[data[9]=='A']))

    classes = np.unique(data[max_gain_feature])
    
    subset_a = data[data[max_gain_feature] == classes[0]] # all rows where feature =  A
    subset_b =  data[data[max_gain_feature] == classes[1]]  # all rows where feature =  B
    print(subset_a.head())

    # make a decision tree node using the feature with the maximum Information gain


    # if all rows belong to the same class, make the current node as a leaf node with the class as its label


    # repeat for the remaining features until we run out of all features, or the decision tree has all leaf nodes



