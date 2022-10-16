from matplotlib.pyplot import clf
from model.Node import Node
from sklearn.tree import DecisionTreeClassifier # ONLY used for testing the comparative model 


class DecisionTree:
    def __init__(self, max_depth = 10):
        self.root = Node(max_depth)

    def fit(self, samples, target):
        self.root.fit(samples, target)
    
    def print_tree(self):
        self.root.print_tree("", "---")

    def predict(self, sample):
        return self.root.predict(sample)
    
    def find_accuracy(dt,t): 
        """
        Determines accuracy of the system.
        Accuracy = (1 - error) = (TP+TN)/(TP+TN+FP+FN)
        
        Parameter:
        dt -- the decision tree
        t -- a set of testing examples
        
        Return:
        accuracy -- how accurate the system is
        """
        correct, total = 0, 0
        for _, e in t.iterrows():
            total += 1
        return round(((correct/total)*100), 1)

    def compute_accuracy(tree, X,y):
        """
        Determines accuracy of the system.
        Accuracy = (1 - error) = (TP+TN)/(TP+TN+FP+FN)
        
        Parameter:
        X -- the decision tree
        y -- a set of testing examples
        
        Return:
        accuracy -- how accurate the system is
        """
        accuracy = 0
        # Loop through elements in each row to find accuracy. 
        for i, row in X.iterrows():
            _pred = tree.predict(row)
            if y[i] == _pred:
                accuracy = accuracy + 1 
            else:
                pass
        accuracy = accuracy / len(y)
        return accuracy

    def decisionTreeClassifier_compute_accuracy(tree, X,y):
        SK_Acc = (tree.predict(X) == y).sum()/len(y)
        return SK_Acc


    def compute_accuracy_with_time(tree, X,y):
        """
        Determines accuracy of the system.
        
        Parameter:
        X -- the decision tree
        y -- a set of testing examples
        
        Return:
        accuracy -- how accurate the system is
        """
        accuracy_array = []
        accuracy = 0
        
        # Loop through elements in each row to find accuracy. 
        for i, row in X.iterrows():

            _pred = tree.predict(row)   
            if y[i] == _pred:
                accuracy = accuracy + 1 
            else:
                pass
            accuracy_array.append(accuracy)
        accuracy = accuracy / len(y)
        return accuracy, accuracy_array

    def decisionTreeClassifier_compute_accuracy(tree, X,y):
        SK_Acc = (tree.predict(X) == y).sum()/len(y)
        return SK_Acc

    def predict_df(tree, X):
        return [ tree.predict(row) for i, row in X.iterrows()]
    def __init__(self, max_depth = 10):
        self.root = Node(max_depth)

    def fit(self, samples, target):
        self.root.fit(samples, target)
    
    def print_tree(self):
        self.root.print_tree("", "---")

    def predict(self, sample):
        return self.root.predict(sample)