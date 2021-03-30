import numpy as np
import pandas as pd
from collections import deque

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




class Node:
    """Contains the feature value,... of each node in the tree"""

    def __init__(self):
        self.value = None
        self.next = None
        self.children = None



class ID3_Decision_Tree:
    """Decision Tree Classifier using ID3 algorithm"""


    def __init__(self):
        self.node = None  # nodes




    # test this function in isolation
    # our classes are A and B
    # NB it is entropy of target -> True and False
    def entropy(self, data):
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


        entropy = sum*-1
        #print(entropy)
        return entropy



    # information gain for specified feature column
    # IG(S, A) = Entropy(S) - ∑((|Sᵥ| / |S|) * Entropy(Sᵥ))
    # where Sᵥ is the set of rows in S for which the feature column A has value v, |Sᵥ| is the number of rows in Sᵥ and likewise |S| is the number of rows in S.
    # how does this differ for each feature??
    def get_information_gain(self, data, feature):
        total_entropy = self.entropy(data) # entropy of target
        classes = np.unique(data[feature])
        #print(classes)
        summation = 0
        for c in classes:
            set_c = data[data[feature]==c]# set of rows in S for which the feature column A has value v
            entropy_set_c = self.entropy(set_c)
            scaling_factor =  len(set_c)/len(data[feature])
            #print(len(set_v))
            #print(len(data[feature]))
            #print(scaling_factor)
            summation += scaling_factor*entropy_set_c
    
        #print(summation)
        info_gain = total_entropy - summation
        return info_gain



    # how to check if correct info_gain is being calculated??

 



    def find_max_info_gain(self, data, feature_list):

        information_gains = {} # use dictionary instead?
        # iterate over features
        #no_features = data.shape[1] - 2 # subtract target row and index row
        #for i in range(1, no_features+1):
        for feature_index in feature_list:
            info_gain = self.get_information_gain(data, feature_index) 
            information_gains[feature_index] = info_gain
    
        # info gain has to do with how it is splitting the target True and Falses! NB to understand for theory part of report
        # find max info gain
        max_gain_feature  = max(information_gains, key=information_gains.get)
        print(max_gain_feature)
        print(information_gains[max_gain_feature])
        return max_gain_feature


    def split_dataset(data, max_gain_feature):
        subsets = []
        classes = np.unique(data[max_gain_feature])
        for c in classes:
            subset_i = data[data[max_gain_feature] == c] # all rows where feature =  A
            subsets.append(subset_i)
        return subsets  


   

    # 1. Calculate the Information Gain of each feature.
    # 2. Considering that all rows don’t belong to the same class, split the dataset S into subsets using the feature for which the Information Gain is maximum.
    # 3. Make a decision tree node using the feature with the maximum Information gain.
    # 4. If all rows belong to the same class, make the current node as a leaf node with the class as its label.
    # 5. Repeat for the remaining features until we run out of all features, or the decision tree has all leaf nodes.
    def build_tree(self, data, feature_list, node):

        # initialise nodes
        if not node:
            print("Initialising node")
            node = Node() 


        # check for purity (all examples have the same class)
        output_classes = np.unique(data["Target"])
        #print(len(output_classes))
        if (len(output_classes)==1): 
            print("Pure node") # we are never reaching pure node - too few features for now??
            node.value = output_classes[0]
            return node


        # if there are no more features available to split the data, choose most common/probable output
        if (len(feature_list) == 0):
            print("No more features")
            print(data['Target'].value_counts().idxmax())
            node.value = data['Target'].value_counts().idxmax()
            return node
    
        # calculate information gain for each feature and find maximum
        max_info_gain_feature = self.find_max_info_gain(data, feature_list)
        # split dataset into subsets using the feature for which info gain is max
    

        # new node
        node.value = max_info_gain_feature
        node.children = []
        classes = np.unique(data[max_info_gain_feature])
        # create a branch for each value of the selected feature
        for c in classes:
            child = Node()
            child.value = c # e.g. A or B
            node.children.append(child) # add child node to tree
            subset_i = data[data[max_info_gain_feature] == c] # find subset where feature = c
            if subset_i.empty: # if subset is empty
                print(max(set(data["Target"]), key=data["Target"].count))
                child.next = max(set(data["Target"]), key=data["Target"].count) # leaf node with label being the most common/probable output
                print('')
            else:
                # remove most recently used feature from feature list
                #feature_list.remove(max_info_gain_feature)
                feature_list = feature_list[feature_list != max_info_gain_feature]
                print(feature_list)
                # recursively call the algorithm to build subtree
                child.next = self.build_tree(subset_i, feature_list, child.next) # pass subset of the data to the next iteration/node
        return node



    def id_3(self, data):

        feature_list =  data.columns.values[2:] # ignore the index and target columns
        print(feature_list)
        self.node = self.build_tree(data, feature_list, self.node) # root
        print(self.node) # node is now returned




    def print_tree(self):
        print("Printing tree")
        if not self.node:
            print("No node?")
            return
        nodes = deque()
        nodes.append(self.node)
        while len(nodes) > 0:
            node = nodes.popleft()
            print("Feature {}".format(node.value))
            if node.children:
                for child in node.children:
                    print('Child value ({})'.format(child.value))
                    nodes.append(child.next)
                    #print(nodes)
            elif node.next:
                print(node.next)



    # need to be able to use the tree that has been built
    """def predict(self, sample):
        
	    node = self

	    while len(node.children) > 0:
		    # Get the current splitting condition.
		    split_con = node.children[0].value
			
		    for child in node.children:
		    	data = child.data
				
			    # Get the first sample in this split
			    first_sample = list(data.index)[0]
				
			    # Check if this child has the right value of the splitting
			    # condition. If not, try another child.
			    if sample[split_con][0] == node.data.loc[first_sample,:][split_con]:
				    node = child
				    break

        return list(node.data[self.y_attr])[0]"""

    


    

    # make a decision tree node using the feature with the maximum Information gain


    # if all rows belong to the same class, make the current node as a leaf node with the class as its label


    # repeat for the remaining features until we run out of all features, or the decision tree has all leaf nodes



