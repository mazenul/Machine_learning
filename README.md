# DATA CLEANING
>> Things between { } brakets are need to change according to the user need

  # import different important libraries:
    improt pandas as pd 
    improt numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

  # import dataset :
    df = pd.read_csv('{lcoation of the dataset}')

  # Check the dataframes different informations: 
    df.head()
    df.shape
    df.columns
    df.info()

  # create a copy to make sure your dataframe do not get changes or harmed 
    data = df.copy()

  # Check the copied datas different informations: 
    data.head()
    data.shape
    data.columns
    data.info()

  # check duplicates and remove them form the data
    data.duplicated()
    data.duplicated().sum()
    data = data.drop_duplicates()

  # remove irrelevant columns :
    data = data.drop(columns = [{columns name e.g. "student_id","age"}])
    >> you can use : df.drop([{name of the column}], axis= 1)

  # check the missing values : 
    data.count()
    data.isnull().sum()

  # replace different unnecessary symbles with NAN [ not a number ] 
    data = data.replace("{which symble want to change}", np.nan)

  # string formating 
    data[{feature name}] = data["{feature name}"].str.title()

  # checking impossible values & convert them in to nan[not a number]
    data.loc[ [condition] , column name ] = np.nan
    Example : >> data.loc[ data["bedroom"] < 0, "bedroom"] = np.nan

  # ckecking invalid label and remove them 
    data = data.dropna(subset=["{name of the label}"]) >> if the label value is nan then the entire row will be droped 
    data = data.query("{condition}")

  # reanalyze the data to find everything is ok or not :
    data.shape
    data

  # feature and label separation: 
    X = data.drop("{label name}" , axis =1)
    y = data["{label name}"]

  # spliting the dataset for training and testing [standard ration 80/20]
    X_train, X_test, y_train, y_test = train_test_split(X ,y , test_size =0.2 , random_status = 42, stratify =y, shuffle =False)



# FEATURE ENGINEERING

>> must remember : feature engineering is needed to perform before the data preporcessing start 
    1.Fetaure  Genearation -->> Feature Trasformation -->> Feature Selection -->> Feature Elemination ..
    2.No predefine step of coding do it according to your need  -->> it will help you to get a better predection model ..

# DATA PREPROCESSING

  # copy the splits to make sure the main split is not disturbed [industry standard]
      1. X_train_prep = X_train.copy()
      2. X_test_prep = X_test.copy()
  # handling missing values :
    Imputation :
      1. Mean imputation
      2. Mode imputation

   # mean imputation:
    # check the values :
         X_train_prep.isnull().sum()
         X_test_prep.isnull().sum()
      # imputation:
         {feature_mean} = X_train_prep["{feature_name}"].mean()
         X_train_prep["{feature_name}"] = X_train_prep["{feature_name}"].fillna({feature_mean})
         X_test_prep["{feature_name}"] = X_test_prep["{feature_name}"].fillna({feature_mean})
      # recheck that all values are ok :
         X_tarin_prep["{feature_name}"].isnull().sum()
         X_test_prep["{feature_name}"].isnull().sum()
      
# HANDLING OUTLIERS:

  1. KEEP : when the dataset value is teal or genune then we have to keep the value
  2. CAP  : when you know the highest range of the feature then the values greater then the range will be caped to the highest range 
  3. REMOVE : if the value is clearly invalied

  # check the max value of the feature:
    X_tarin_prep["{feature_name}"].max()
    >> for multiple feature do this : 
      Numbers = X_train_prep.select_dtypes(include=['int','float']).columns.to_numpy()
      print(Numbers)
      X_train_prep[Numbers].max()
    
  # find the upper limit :
    {feature_upper_limit} = X_tarin_prep["{feature_name}"].quantile(0.95)
    >>> for multiple features do this:
      {feature_upper_limit} = X_train_prep[Numbers].quantile(0.95)  >> it will provide a array of max value of different features
      print({feature_uppe_limit})
  
  # cliping the feature value: 
    X_tarin_prep["{feature_name}"] = X_tarin_prep["{feature_name}"].clip(upper = {feature_name}) 
    >>> for multi features:
      X_train_prep[Numbers] = X_train_prep[Numbers].clip(upper={feature_upper_limit},axis=1)   >> axis = 1 mean do it for each feature 
  
  # removing the feature values:
    X_train_prep = X_train_prep[X_train_prep["{feature_name}"] <= {feature_upper_limit}]

# feature scaling :
  scaler = StandardScaler()
  X_train_prep[Numbers] = scaler.fit_transform(X_train_prep[Numbers])
  X_test_prep[Numbers] = scaler.transform(X_test_prep[Numbers])

# catagolical encoding :

  # binary encoding for yes no answer:: 
    X_train_prep["{feature_name}"] = X_train_prep["{feature_name}"].replace("Yes":1, "No":0)
    >> using map : X_train_prep["{feature_name}"] = X_train_prep["{feature_name}"].map({"Yes":1, "No":0})
  
  # find the feature with string datatype 
    {variable_name} = X_train_prep.select_dtypes(include=['str']).columns.to_numpy()   >>if you find any error with str, you can use object
    print({variable_name})
 
  # one hot encoding: 
    X_train_prep = pd.get_dummies(X_train_prep,columns={variable_name},dtype=int)
    X_test_prep = pd.get_dummies(X_test_prep,columns= {variable_name},dtype=int)
    X_train_prep, X_test_prep = X_train_prep.align(X_test_prep,join='left',axis=1,fill_value=0) >> this is importent because if the string type feature is only available in test then it will not be in the train split 


# MODEL TRAINING

>> use X_train_prep, X_test_prep (the preprocessed data), NOT the raw X_train / X_test
>> first decide your problem type:
     label is a number (price, salary ...)        -->> REGRESSION
     label is a category (yes/no, spam/not spam)  -->> CLASSIFICATION



  # PART A : REGRESSION  (10 models)
    from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
    from sklearn.svm import SVR
    from sklearn.neural_network import MLPRegressor
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    import numpy as np

  # put all models in one dictionary :
    reg_models = {
        "Linear Regression"      : LinearRegression(),
        "Ridge Regression"       : Ridge(alpha=1.0),
        "Lasso Regression"       : Lasso(alpha=0.1),
        "ElasticNet"             : ElasticNet(alpha=0.1, l1_ratio=0.5),
        "KNN Regressor"          : KNeighborsRegressor(n_neighbors=5),
        "Decision Tree"          : DecisionTreeRegressor(max_depth=10, random_state=42),
        "Random Forest"          : RandomForestRegressor(n_estimators=200, random_state=42),
        "Gradient Boosting"      : GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=42),
        "AdaBoost"               : AdaBoostRegressor(n_estimators=100, random_state=42),
        "Support Vector (SVR)"   : SVR(kernel="rbf", C=1.0),
        "Neural Network (MLP)"   : MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42),
    }

  # train + validate every model in a loop :
    reg_results = []

    for name, model in reg_models.items():
        model.fit(X_train_prep, y_train)

        y_tr_pre = model.predict(X_train_prep)
        y_te_pre = model.predict(X_test_prep)

        reg_results.append({
            "Model"      : name,
            "R2 Train"   : r2_score(y_train, y_tr_pre),
            "R2 Test"    : r2_score(y_test, y_te_pre),
            "MSE Test"   : mean_squared_error(y_test, y_te_pre),
            "RMSE Test"  : np.sqrt(mean_squared_error(y_test, y_te_pre)),
            "MAE Test"   : mean_absolute_error(y_test, y_te_pre),
        })

    reg_results = pd.DataFrame(reg_results).sort_values("R2 Test", ascending=False)
    print(reg_results)

    >> how to read the result :
         R2 Train high  + R2 Test low   -->> OVERFITTING
         R2 Train low   + R2 Test low   -->> UNDERFITTING
         R2 Train ~ R2 Test (both good) -->> GOOD MODEL

  # optional : XGBoost  ( pip install xgboost )
    from xgboost import XGBRegressor
    xgb = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42)
    xgb.fit(X_train_prep, y_train)
    print(r2_score(y_test, xgb.predict(X_test_prep)))




  # PART B : CLASSIFICATION  (10 models)
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

    clf_models = {
        "Logistic Regression"    : LogisticRegression(max_iter=1000),
        "KNN Classifier"         : KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes"            : GaussianNB(),
        "Decision Tree"          : DecisionTreeClassifier(max_depth=10, random_state=42),
        "Random Forest"          : RandomForestClassifier(n_estimators=200, random_state=42),
        "Extra Trees"            : ExtraTreesClassifier(n_estimators=200, random_state=42),
        "Gradient Boosting"      : GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=42),
        "AdaBoost"               : AdaBoostClassifier(n_estimators=100, random_state=42),
        "Support Vector (SVC)"   : SVC(kernel="rbf", C=1.0),
        "Neural Network (MLP)"   : MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42),
    }

    clf_results = []

    for name, model in clf_models.items():
        model.fit(X_train_prep, y_train)

        y_tr_pre = model.predict(X_train_prep)
        y_te_pre = model.predict(X_test_prep)

        clf_results.append({
            "Model"           : name,
            "Accuracy Train"  : accuracy_score(y_train, y_tr_pre),
            "Accuracy Test"   : accuracy_score(y_test, y_te_pre),
            "Precision Test"  : precision_score(y_test, y_te_pre, average="weighted", zero_division=0),
            "Recall Test"     : recall_score(y_test, y_te_pre, average="weighted", zero_division=0),
            "F1 Test"         : f1_score(y_test, y_te_pre, average="weighted", zero_division=0),
        })

    clf_results = pd.DataFrame(clf_results).sort_values("F1 Test", ascending=False)
    print(clf_results)

  # detail report of ONE model (confusion matrix + per-class scores) :
    best_model = clf_models["Random Forest"]
    y_te_pre = best_model.predict(X_test_prep)
    print(confusion_matrix(y_test, y_te_pre))
    print(classification_report(y_test, y_te_pre))

  # optional : XGBoost  ( pip install xgboost )
    >> XGBoost needs labels as 0,1,2... so encode the label first if it is text
    from xgboost import XGBClassifier
    xgb = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42)
    xgb.fit(X_train_prep, y_train)
    print(accuracy_score(y_test, xgb.predict(X_test_prep)))




  # PART C : cross validation (more reliable than one split)
    from sklearn.model_selection import cross_val_score

  # regression :
    scores = cross_val_score(reg_models["Random Forest"], X_train_prep, y_train, cv=5, scoring="r2")
    print(scores.mean(), scores.std())

  # classification :
    scores = cross_val_score(clf_models["Random Forest"], X_train_prep, y_train, cv=5, scoring="f1_weighted")
    print(scores.mean(), scores.std())




  # PART D : save the best model and use it  ( same as your note )
    import joblib
    joblib.dump(reg_models["Random Forest"], "{Model_Path}.pkl")      # change model name as you need

    model = joblib.load("{Model_Path}.pkl")
    prediction = model.predict({sample_data})      # sample_data must be preprocessed the same way (scaler + encoding + same columns)

    >> IMPORTANT : also save the scaler, otherwise new data can not be scaled the same way
    joblib.dump(scaler, "{scaler_path}.pkl")
