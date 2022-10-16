def binarize_sex(val):
    if val == 'Male':
        return 1
    else:
        return 0  

def prep(data): 
    # remove any null values
    data.isnull().any()
    
    # assign numerical value to the boolen value of sex column
    data['sex'] = data['sex'].apply(binarize_sex)
    
    # drop null values and update it in the dataframe
    data.dropna(inplace=True)
    
    return data
