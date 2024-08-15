# A-Share-Listed-Company-Patent-Research-Based-on-XGBoost-algorithm
import numpy as np
import pandas as pd
import xgboost as xgb
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import time
from matplotlib import pyplot
from xgboost import plot_importance

data=pd.read_csv('D:/学姐的数据/财务数据2.csv',encoding='UTF-8')
label=data.iloc[:,[7]]

mask=np.random.rand(len(data))<0.8

train=data[mask]
test=data[~mask]

params={
        'objective':'reg:linear',
        'booster':'gbtree',
        "eta":0.1,
        'min_child_weight':6,
        'max_depth':4,
        }
num_round=6000
t=time.time()

xgb_train=xgb.DMatrix(train.iloc[:,:7],label=train.iloc[:,[7]])
xgb_test=xgb.DMatrix(test.iloc[:,:7],label=test.iloc[:,[7]])

watchlist=[(xgb_train, 'train'),(xgb_test,'test')]
model=xgb.train(params,xgb_train,num_round,watchlist)
model.save_model('./model.xgb')

bst=xgb.Booster()
bst.load_model('./model.xgb')

print('时长%0.1fs'%(time.time()-t))

pred=bst.predict(xgb_test)

pred=pd.DataFrame(pred)

plot_importance(model, max_num_features=7) # top 10 most important features
plt.show()
