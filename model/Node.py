import numpy as np
from pandas.api.types import is_numeric_dtype
import matplotlib.pyplot as plt

# Contains the information of the node and another nodes of the Decision Tree.
class Node: 

    def __init__(self, max_depth=10, current_depth=0): 

        # Set depth values
        self.max_depth = max_depth
        self.current_depth = current_depth 
        
        # Declaring variables specific to this node
        self.children = {} 
        self.decision = None 
        self.feature_name_split = None  # Splitting features 
        self.continious_feature = True

        # Counts
        self.target_values_count = None 
        self.count = 0 # Attribute count

        # Attribute calculations
        self.entropy = 0
        
       
    def calculate_entropy(self,y):
        """
        :param y: The data samples of a discrete distribution
        """
        if len(y) < 2: #  a trivial case
            return 0
        freq = np.array( y.value_counts(normalize=True) )
        # the small eps for safe numerical computation 
        return -(freq * np.log2(freq + 1e-16)).sum()  
        

    def information_gain_calc(self, samples, attr, target):
        values = samples[attr].value_counts(normalize=True)
        split_ent = 0
        for v, fr in values.items():
            index = samples[attr]==v
            sub_ent = self.calculate_entropy(target[index])
            split_ent += fr * sub_ent
        ent = self.calculate_entropy(target)
        return ent - split_ent

    def compute_info_gain_continuous(self,samples, attr, thres_value, target):
            """
            Compute the info gain based on feature and the split value thres_value

            """
            split_ent = 0
            # we split the samples into two groups:
            # feature with value >= thres_value
            # and feature with value < thres_value
            # compute the info gain for the two subsets
            index = samples[attr]<thres_value
            values = index.value_counts(normalize=True) # values can be only True or False
            for v, fr in values.items():
                sub_ent = self.calculate_entropy(target[index if v else ~index ])
                split_ent += fr * sub_ent

            ent = self.calculate_entropy(target)
            return ent - split_ent

    def print_tree(self, prefix='', direction=''):
        
        if self.feature_name_split is not None:

            if self.continous_feature == True:
                print(f"{prefix}|{direction} {self.feature_name_split} < {self.split_feat_value} ( Ent {self.entropy:0.4f}, values={self.target_values_count} )")
                self.children['left'].print_tree(f"{prefix}|    ","---")
                print(f"{prefix}|{direction} {self.feature_name_split} >= {self.split_feat_value}")
                self.children['right'].print_tree(f"{prefix}|    ","---")
            else:
                print(f"{prefix}|{direction} {self.feature_name_split}{self.split_feat_value} ( Ent {self.entropy:0.4f}, values={self.target_values_count} )")
                for v in self.split_feat_value:
                    print(f"{prefix}|    {v}")
                    self.children[v].print_tree(f"{prefix}|    ","---")
        else:
            print(f"{prefix}|{direction} class :{self.decision}, count: {self.count}, values={self.target_values_count}")


    def predict(self, sample):
        if self.decision is not None:
            # Leaf node with a decision
            return self.decision 
        else: 
            # this node is an internal one, further queries about an attribute 
            # of the data is needed.
            attr_val = sample[self.feature_name_split]
            if self.continous_feature == True:
                if (attr_val< self.split_feat_value):
                    child = self.children['left']
                else:
                    child = self.children['right']
            else:
                # Discrete feature, select based on the feature value
                # Check first if there is a node associated to the current value
                if (attr_val in self.children):
                    child = self.children[attr_val]
                else:
                    # There is no child node associated with the current value
                    # => Select the child with the lowest entropy.
                    child_entropy_min = 1e10
                    child_feat_value = None
                    for v in self.children:
                        if self.children[v].entropy < child_entropy_min:
                            child_entropy_min = self.children[v].entropy
                            child_feat_value = v
                            child = self.children[v]
                    print(f"Feature Value not found ({attr_val}) using {child_feat_value} instead.")
            return child.predict(sample)

    def fit(self, X, y):
            """
            The function accepts a training dataset, from which it builds the tree 
            structure to make decisions or to make children nodes (tree branches) 
            to do further inquiries
            :param X: [n * p] Pandas DataFrame n observed data samples of p attributes 
            :param y: [n] target values
            """
            if len(X) == 0:
                # If the data is empty when this node is arrived, 
                # we just make an arbitrary decision
                self.decision = 0
                self.count=-1
                return
            else: 
                unique_values = y.unique()
                self.count=len(y)
                if len(unique_values) == 1:
                    # There is only one target class, we are a leaf node.
                    self.decision = unique_values[0]
                    return
                # exit if we reach the max depth of the tree
                elif self.current_depth >= self.max_depth:   
                    self.target_values_count = y.value_counts().to_dict()
                    self.entropy = self.calculate_entropy(y)
                    self.decision = y.mode()[0] # sets the decision to the value that appears most often.
                    return
                else:
                    self.entropy = self.calculate_entropy(y)
                    self.target_values_count = y.value_counts().to_dict()
                    info_gain_max = -1
                    for attribute in X.keys():    # check each feature to split
                        idx = X[attribute].first_valid_index()
                        if is_numeric_dtype(type(X[attribute][idx])):
                            # apply a threshold for the split by getting the best value
                            # Loop through every possible value.
                            _feat_values = X[attribute].sort_values(ascending=True).unique()
                            for value in _feat_values:
                                aig = self.compute_info_gain_continuous(X,attribute,value,y)
                                if aig > info_gain_max:
                                    info_gain_max = aig
                                    self.feature_name_split = attribute
                                    self.split_feat_value = value
                                    self.continous_feature = True
                        else:
                            # discrete attribute, apply
                            aig = self.compute_info_gain(X, attribute, y)
                            if aig > info_gain_max:
                                self.feature_name_split = attribute
                                info_gain_max = aig
                                self.continous_feature = False               

                    if self.continous_feature == True:
                        index = X.loc[:,self.feature_name_split] < self.split_feat_value
                        self.children['left'] = Node(self.max_depth, self.current_depth+1)
                        self.children['left'].fit(X[index],y[index])
                        self.children['right'] = Node(self.max_depth, self.current_depth+1)
                        self.children['right'].fit(X[~index],y[~index])
                    else:
                        # if the feature is discrete, we will have as much branch as values seen here
                        self.split_feat_value = X[self.feature_name_split].unique()
                        for v in self.split_feat_value:
                            index = X[self.feature_name_split] == v
                            self.children[v] = Node(self.max_depth, self.current_depth+1)
                            self.children[v].fit(X[index],y[index])
