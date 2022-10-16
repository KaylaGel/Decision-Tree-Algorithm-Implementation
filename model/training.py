from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

parameters = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex']

def build_train_test_split(dataset, test_size=None , random_state=None): 
    X = dataset[parameters]
    y = dataset.species

    cols = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex', 'species']

    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=test_size, random_state=random_state) 
 
    train_df = pd.DataFrame(np.c_[X_train, y_train], columns= cols)
    test_df = pd.DataFrame(np.c_[X_test, y_test], columns= cols)

    return train_df , test_df

def build_new_test_split(dataset, test_size=None , random_state=None): 
  
  labels = dataset['feature_names']
  labels = labels + ['target']
  
  X_train, X_valid, y_train, y_valid = \
  train_test_split(dataset['data'], dataset['target'],test_size=test_size, random_state=random_state)

  train_df = pd.DataFrame(data= np.c_[X_train, y_train],columns= labels)
  valid_df = pd.DataFrame(data= np.c_[X_valid, y_valid],columns= labels)

  return train_df, valid_df


def extract_df_values_new(train_df, valid_df):
    X = train_df.drop( "target", axis = 1)
    valid_X = valid_df.drop( "target", axis = 1)
    y = train_df["target"]
    valid_y = valid_df["target"]

    return X, valid_X, y, valid_y

"""
Purpose: 
- Extract training and validation values from respective dataframes
"""
def extract_df_values(train_df, valid_df):
  X = train_df.drop( "species", axis = 1)
  valid_X = valid_df.drop( "species", axis = 1)
  y = train_df["species"]
  valid_y = valid_df["species"]

  return X, valid_X, y, valid_y