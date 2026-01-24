# Machine_learning just the format to remember using linear regression model 
information about machine learning related 

improt pandas as pd 
df = pd.read_csv('')

y = df['']
x = df.drop([''], axis= 0/1) [0 for row and 1 means colurm ]

from sklearn.model_selection import train_test_split as tts

x_train, x_test, y_train, y_test = tts(x ,y , test_size =0.2 , random_status =100, stratify =y, shuffle =False)


## Train model and predect result 

from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(x_train,y_train)

y_tr_pre = model.predict(x_train)
y_te_pre = model.predict(x_test)

== model velidation ==

from sklearn.metrics import r2_score, mean_squared_error 

r2_train = r2_score(y_train,y_tr_pre)

r2_test = r2_score(y_test,y_te_pre)


mse_train = mean_squared_error(y_train,y_tr_pre)

mse_test = mean_squared_error(y_test,y_te_pre)





== model save ==

import joblib

joblib.dump(model , Model_Path)




== use the saved model ==

import joblib 

model = joblib.load(Model_Path)

prediction = model.predict(Sample data)