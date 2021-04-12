import math
import numpy as np
import pandas as pd
from collections import deque
#import profile

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


    def __init__(self, data, targets):
        self.node = None  # nodes
        self.data = data
        #self.feature_list = feature_list
        self.targets = targets
        self.target_classes = np.unique(targets)
        self.target_counts = [list(targets).count(c) for c in self.target_classes]
        self.entropy = self.calc_entropy([i for i in range(len(self.targets))])  # calculates the initial entropy



    # for efficiency - pass the index rather than the whole dataset
    # data indices indicate the rows that are in play
    def calc_entropy(self, data_indices):
      
        targets = [self.targets[i] for i in data_indices] # get list of target values for each row in current data subset
        classes = np.unique(targets)
        target_counts = [targets.count(c) for c in classes]
        entropy = sum([- (count / len(targets)) * math.log(count / len(targets), 2) if count else 0 for count in target_counts])
       
        return entropy



    # information gain for specified feature column
    # IG(S, A) = Entropy(S) - ∑((|Sᵥ| / |S|) * Entropy(Sᵥ))
    # where Sᵥ is the set of rows in S for which the feature column A has value v, |Sᵥ| is the number of rows in Sᵥ and likewise |S| is the number of rows in S.
    def get_information_gain(self, data_indices, feature_name):

        total_entropy = self.calc_entropy(data_indices) # total entropy of targets/labels for whole dataset

        # store in a list all the values of the chosen feature
        feature_data = [self.data[i][feature_name] for i in data_indices]

        # all unique values the feature can take on
        feature_values = np.unique(feature_data)

        # calculate count of each feature value
        feature_values_count = [feature_data.count(v) for v in feature_values]

        # get the rows for which the data equals a given feature value
        subset_indices = [
            [data_indices[i]
            for i, x in enumerate(feature_data)
            if x == v]
            for v in feature_values
        ]

        # compute the information gain for the subset with feature == v
        information_gain = total_entropy - sum([count_v / len(data_indices) * self.calc_entropy(indices_v)
                                     for count_v, indices_v in zip(feature_values_count, subset_indices)])

        return information_gain

 



    def find_max_info_gain(self, data_indices, feature_list):

        information_gains = [self.get_information_gain(data_indices, feature_name)  for feature_name in feature_list]
        max_gain_feature = feature_list[information_gains.index(max(information_gains))]
    
        return max_gain_feature



   

    # 1. Calculate the Information Gain of each feature.
    # 2. Considering that all rows don’t belong to the same class, split the dataset S into subsets using the feature for which the Information Gain is maximum.
    # 3. Make a decision tree node using the feature with the maximum Information gain.
    # 4. If all rows belong to the same class, make the current node as a leaf node with the class as its label.
    # 5. Repeat for the remaining features until we run out of all features, or the decision tree has all leaf nodes.
    def build_tree(self, data_indices, feature_list, node):

        # initialise nodes
        if not node:
            #print("Initialising node")
            node = Node() 


        # check for purity (all examples have the same class)
        # if all rows belong to the same class, make the current node as a leaf node with the class as its label
        targets = [self.targets[i] for i in data_indices]
        output_classes = np.unique(targets)
        #print(len(output_classes))
        if (len(output_classes)==1): 
            #print("Pure node") # we are never reaching pure node - too few features for now??
            node.value = output_classes[0] # leaf node with label being the most common/probable output
            return node


        # if there are no more features available to split the data, choose most common/probable output
        if (len(feature_list) == 0):
            #print("No more features")
            node.value = max(set(targets), key=targets.count) 
            return node
    
        # calculate information gain for each feature and find maximum
        max_info_gain_feature = self.find_max_info_gain(data_indices, feature_list)
   
        # split dataset into subsets using the feature for which info gain is max
    

        # new node
        node.value = max_info_gain_feature
        node.children = []
        values = ['A', 'B'] 

        # create a branch for each value of the selected feature
        for v in values:
            child = Node()
            child.value = v # e.g. A or B
            node.children.append(child) # add child node to tree
            subset_indices = [i for i in data_indices if self.data[i][max_info_gain_feature] == v] # find subset where feature = v
            if  (len(subset_indices)==0): # if subset is empty
            
                # create new leaf node
                new_node = Node() 
                new_node.value = max(set(targets), key=targets.count) 
                new_node.children = []
                child.next = new_node  # leaf node with label being the most common/probable output
               
            else:
                # remove most recently used feature from feature list
                feature_list = feature_list[feature_list != max_info_gain_feature]
    
                # recursively call the algorithm to build subtree
                child.next = self.build_tree(subset_indices, feature_list, child.next) # pass subset of the data to the next iteration/node
        return node



    def id_3(self, feature_list):

        #print(data)
        data_indices = [i for i in range(len(self.data))]
        #feature_list =  self.data.columns.values[:-1] # ignore the index and target columns
        #print("Features: {}".format(feature_list))
        #print("Building tree...")
        self.node = self.build_tree(data_indices, feature_list, self.node) # root
        #print(self.node) # node is now returned




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
    
     

    

    

    def predict(self, sample):

        node = self.node

        #print(sample)


        while(len(node.children) > 0): # while node we are on is not a leaf node
            #print("Splitting Feature: {}".format(node.value))
            splitting_feature = node.value
            #print("Number of children {}".format(len(node.children)))

      
            for child in node.children:
                #feature_value = sample[splitting_feature].values[0]
                feature_value = sample[splitting_feature]
                #print("Feature Value: {}".format(feature_value)) 
                #print("Child Value: {}".format(child.value))
                if (feature_value==child.value):
                    node = child.next # move to this child or move to next??
                    #print("Moving") # on validation set -> not moving
                    break

            # check this condition
            if(node.children==None):
                break
            #else:
                #print(len(node.children))
        
        return node.value


    
    def predict_batch(self, samples):

        #print("Predicting...")
        #print(samples)

        #predictions = [self.predict(samples.iloc[[i]]) for i in range(samples.shape[0])]
        predictions = []
        for i in range(len(samples)):
            #print(samples.iloc[[i]])
            predictions.append(self.predict(samples[i]))
        return predictions
        



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



