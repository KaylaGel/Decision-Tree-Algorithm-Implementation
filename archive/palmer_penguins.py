# this code was removed from run.py file as it was only used for the purpose of testing

def binarize_sex(val):
    if val == 'Male':
        return 1
    else:
        return 0  

def new_test(): 
    parameters = ['species', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex', 'year']


    start_time = time.time()

    data = pd.read_csv('data\penguins.csv')
    data = data.dropna()
    #data = data.drop(['sex', 'island', 'flipper_length_mm', 'body_mass_g'], axis=1)
    data = data.drop(['island'], axis=1)
    data['sex'] = data['sex'].apply(binarize_sex)
    data = data[data['species'] != 'Chinstrap']

    X = data.drop(['species'], axis=1)
    X = X.values
    ss = StandardScaler()
    X = ss.fit_transform(X)
    
    y = data['species']
    spicies = {'Adelie': -1, 'Gentoo': 1}
    y = [spicies[item] for item in y]
    y = np.array(y) 
    
    # Remove sample that is too close
    X = np.delete(X, 182, axis=0)
    y = np.delete(y, 182, axis=0)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=33)
    
    skClassifier = DecisionTreeClassifier(criterion='entropy',max_depth=5,random_state=1)
    fitted = skClassifier.fit(X_train, y_train)

    
    data = data.drop(['Unnamed: 0'], axis=1)
    print(data)
    print(parameters)

    export_graphviz(
        fitted, 
        out_file= os.getcwd() + '\data\penguins_dt.dot',
        feature_names=parameters,
        rounded=True,
        filled=True)

