from model import run
from matplotlib import pyplot as plt
from model.ID3 import DecisionTree


def testing(): 

    print ("_________________________________________________ ")
    print ("                                                  ")
    print ("       Testing Palmer Penguins with sklearn"       )
    print ("_________________________________________________ ")
    
    penguin_tree = run.palmer_penguins_sklearn()

    print ("_________________________________________________ ")
    print ("                                                  ")
    print ("          Testing Iris with sklearn"               )
    print ("_________________________________________________ ")
    skClassifier = run.iris_sklearn_decisionTree()



if __name__ == "__main__":

    print ("_________________________________________________ ")
    print ("                                                  ")
    print ("          DATASET 1: PALMER PENGUINS"              )
    print ("_________________________________________________ ")
   
    penguin_tree = run.palmer_penguins_dataset()
    print ("------------------------------------------")
    print ("                tree                      ")
    print ("------------------------------------------")
    penguin_tree.print_tree()
    
    print ("_________________________________________________ ")
    print ("                                                  ")
    print ("          DATASET 2: IRIS DATASET"                 )
    print ("_________________________________________________ ")

    iris_tree = run.iris()
    print ("------------------------------------------")
    print ("                tree                      ")
    print ("------------------------------------------")
    iris_tree.print_tree()

    # run tests against skClassifier 
    testing()
