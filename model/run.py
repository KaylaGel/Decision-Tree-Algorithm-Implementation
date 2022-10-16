from data import data_preparation 
from model import visualisations
from model.ID3 import DecisionTree
from model.statisitcs import print_statistics
from model.training import build_train_test_split, extract_df_values, build_new_test_split, extract_df_values_new
from sklearn import datasets # used to import the iris dataset
import pandas as pd # used to creare dataframes
import time # gets the current time
import os

# The following imports are only used for evaluation 
from sklearn.tree import DecisionTreeClassifier, export_graphviz # For usecase of the sklearn decision tree classification
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

def palmer_penguins_dataset():
    # load data
    penguin_data = pd.read_csv('data\penguins.csv')
    
    # start timer 
    start_time = time.time()

    # prepare data
    data_preparation.prep(penguin_data)

    # define key parameters
    parameters = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex']
    target = "species"

    # training the model  
    penguin_train_df, penguin_test_df = build_train_test_split(penguin_data, test_size = 0.2, random_state=1)
    penguin_X, penguin_test_X, penguin_y, penguin_test_y = extract_df_values(penguin_train_df, penguin_test_df )
    
    # run model 
    model = DecisionTree(max_depth=5)
    model.fit(penguin_X,penguin_y)

    end_time = time.time()
    total_time = end_time - start_time

    testing_data_accuracy, testing_accuracy_array = DecisionTree.compute_accuracy_with_time(model,penguin_test_df,penguin_test_df.species)
    training_data_accuracy, training_accuracy_array  =  DecisionTree.compute_accuracy_with_time(model, penguin_train_df,penguin_train_df.species)
    print ("Accuracy on validation data: ",  testing_data_accuracy)
    print ("Accuracy on training data: ", training_data_accuracy)

    print ("---------------------------------------------")
    print ("STATISTICS                                   ")
    print ("---------------------------------------------")
    print_statistics(model,total_time, training_data_accuracy, testing_data_accuracy, len(penguin_train_df), len(penguin_test_df) )

    visualisations.plot("PALMER PENGUINS", testing_accuracy_array, training_accuracy_array)

    return model


def iris(): 
    
    iris = datasets.load_iris()

    # start timer 
    start_time = time.time()
    
    iris_train_df, iris_test_df = build_new_test_split(iris, test_size = 0.2, random_state=1)
    iris_X, iris_test_X, iris_y, iris_test_y = extract_df_values_new(iris_train_df, iris_test_df)
    iris_Tree = DecisionTree(max_depth=5)
    iris_Tree.fit(iris_X, iris_y)

    testing_data_accuracy, testing_accuracy_array  = DecisionTree.compute_accuracy_with_time(iris_Tree, iris_test_X ,iris_test_df['target'])
    training_data_accuracy, training_accuracy_array = DecisionTree.compute_accuracy_with_time(iris_Tree, iris_X, iris_train_df['target'])

    end_time = time.time()
    total_time = end_time - start_time
    
    print("Accuracy on testing data:", training_data_accuracy)
    print ("Accuracy on training data: ", testing_data_accuracy)

    print ("------------------------------------------")
    print ("STATISTICS ")
    print ("------------------------------------------")
    print_statistics(iris_Tree,total_time, training_data_accuracy, testing_data_accuracy, len(iris_train_df), len(iris_test_df) )

    visualisations.plot("IRIS", testing_accuracy_array, training_accuracy_array)

    return iris_Tree
def palmer_penguins_sklearn(): 


    # load data
    X, y = make_blobs(n_samples=344)
    penguin_data = pd.read_csv('data\penguins.csv')
    
    # start timer 
    start_time = time.time()

    # prepare data
    data_preparation.prep(penguin_data)

    # define key parameters
    parameters = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex']
    target = "species"

    # training the model  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    
    skClassifier = DecisionTreeClassifier(criterion='entropy',max_depth=5,random_state=1)
    fitted = skClassifier.fit(penguin_data[parameters], penguin_data[target])

    accuracy = skClassifier.score(penguin_data[parameters], penguin_data[target])
    
    end_time = time.time()
    total_time = end_time - start_time

    print ("total_time" , total_time)
    print ("accuracy: ", accuracy)
    
    print ("------------------------------------------")
    print ("STATISTICS ")
    print ("------------------------------------------")
    print_statistics(skClassifier,total_time, 0.1, accuracy, len(X_train), len(y_test) )

    export_graphviz(
        fitted, 
        out_file= os.getcwd() + '\data\penguins_dt.dot',
        feature_names=parameters,
        rounded=True,
        filled=True)



def iris_sklearn_decisionTree(): 
    iris = datasets.load_iris()

    # start timer 
    start_time = time.time()

    
    iris_train_df, iris_test_df = build_new_test_split(iris, test_size = 0.2, random_state=1)
    iris_X, iris_test_X, iris_y, iris_test_y = extract_df_values_new(iris_train_df, iris_test_df)
    skClassifier = DecisionTreeClassifier(criterion='entropy',max_depth=5,random_state=1)
    skClassifier.fit(iris_X, iris_y)

    testing_data_accuracy   = DecisionTree.decisionTreeClassifier_compute_accuracy(skClassifier, iris_test_X ,iris_test_df['target'])
    training_data_accuracy = DecisionTree.decisionTreeClassifier_compute_accuracy(skClassifier, iris_X, iris_train_df['target'])

    end_time = time.time()
    total_time = end_time - start_time
    
    print("Accuracy on testing data:", training_data_accuracy)
    print ("Accuracy on training data: ", testing_data_accuracy)
    
    print ("------------------------------------------")
    print ("STATISTICS ")
    print ("------------------------------------------")
    print_statistics(skClassifier,total_time, training_data_accuracy, testing_data_accuracy, len(iris_train_df), len(iris_test_df) )

    return skClassifier