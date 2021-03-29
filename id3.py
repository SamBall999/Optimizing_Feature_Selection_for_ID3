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


# NB are we calculating for one feature or all features- can we efficiently calculate for all features at once? - vectorized implementation?
# test this function in isolation
# our classes are A and B
def entropy(data, index):
    sum = 0
    feature = data[index] # NB try to do for all features
    classes = np.unique(feature)
    print(classes)
    # iterate over classes (will be 2 in our case)
    for i in range(len(classes)):
        prob_i = len(data[feature==i])/len(feature) # probability of class ‘i’ = “number of rows with class i in the target column” to the “total number of rows” in the dataset
        log_2_p_i  = np.log(prob_i) / np.log(2) # check this
        sum += prob_i*log_2_p_i

    entropy = sum*-1
    return entropy
