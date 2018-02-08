import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from pathlib import Path
import lightgbm as lgb
from sklearn.model_selection import train_test_split


df = pd.read_pickle("Files/CleanedProcessedData.pickle")

# Sample down for faster development 
sampleRatio = .01
df = df.sample(frac=sampleRatio, replace=False) 


### Available Products ###
#products_FlatBill 	
#products_Paperless 	
#products_BudgetBill 	
#products_EFT 	
#products_WaterHeater 	
#products_HeatPump

targetFeature = "products_Paperless"

X = df.iloc[:,:-6]
y = df.loc[:,targetFeature]

test_size = .33
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=123)

print("Training on {0}".format(targetFeature))
print("Class balance:")
print(y.value_counts() / len(y))

print("Using {0} of the data, with a test ratio of {1}\n\n".format(sampleRatio,test_size))

params2 = {}
params2['learning_rate'] = .028704 # 0.011011
params2['boosting_type'] = 'gbdt'
params2['objective'] = 'binary'
params2['metric'] = 'binary_logloss'
params2['sub_feature'] = .514492
params2['num_leaves'] = 255
params2['max_depth'] = 7
params2['min_data'] = 32
params2['verbosity'] = 0
params2['bagging_fraction'] = 0.85
params2['lambda_l1'] = .018953
params2['lambda_l2'] = .05242
params2['bagging_freq'] = 5
params2['nthread'] = 16

# dont use crashes
gbm = lgb.LGBMClassifier(**params2)

# Fit
gbm.fit(X_train, y_train.values.ravel())

# Predict
predictions = gbm.predict(X_test)

print(accuracy_score(y_test.values.ravel(),predictions))
print("Finished")



