import pandas as pd
from sklearn import svm
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics


Train_Data = pd.read_csv('tcd-ml-1920-group-income-train.csv', sep = ",")
Test_Data = pd.read_csv('tcd-ml-1920-group-income-test.csv', sep = ",")

Data = pd.concat([Train_Data,Test_Data],ignore_index=True)
Data = Data.dropna(subset=['Size of City','Age','Year of Record'])



#filling most freq value for each col for all the NaNs
print(Data.fillna(Data.mode().iloc[0], inplace = True))


#Label Encoder for all object type data
for col in Train_Data.dtypes[Train_Data.dtypes == 'object'].index.tolist():
    le = LabelEncoder()
    le.fit(Data[col].unique().astype(str))
    Data[col] = le.transform(Data[col].astype(str))


def frequencyEncoding(data,col1,col2,normalize=True):
    for i,c1 in enumerate(col1):
        vc = data[c1].value_counts(dropna=False, normalize=normalize).to_dict()
        nm = c1 + '_FE_FULL'
        data[nm] = data[c1].map(vc)
        data[nm] = data[nm].astype('float32')
        for j,c2 in enumerate(col2):
            new_col = c1 +'_'+ c2
            print('frequency encoding:', new_col)
            data[new_col] = data[c1].astype(str)+'_'+data[c2].astype(str)
            temp_data = data[new_col]
            fq_encode = temp_data.value_counts(normalize=True).to_dict()
            data[new_col] = data[new_col].map(fq_encode)
            data[new_col] = data[new_col]/data[c1+'_FE_FULL']
    return data

col1 = [ 'Gender', 'Country',
        'Profession','Hair Color', 'University Degree','Satisfation with employer','Crime Level in the City of Employement','Housing Situation','Satisfation with employer','Work Experience in Current Job [years]','Yearly Income in addition to Salary (e.g. Rental Income)']
col2 = ['Body Height [cm]','Wears Glasses']

Data = frequencyEncoding(Data,col1,col2)


del_col = set(['Total Yearly Income [EUR]','Instance'])
using_col =  list(set(Data) - del_col)
using_col

#segreagting the variables into X and y and train and test

X,test_data = Data[using_col].iloc[:1048573],Data[using_col].iloc[1048574:]
X_test_id = Data['Instance'].iloc[1048574:]
y = Data['Total Yearly Income [EUR]'].iloc[:1048573]


#split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1234)

#LightGBM

import lightgbm as lgb
params = {
          'max_depth': 20,
          'learning_rate': 0.001,
          "boosting": "gbdt",
          "bagging_seed": 11,
          "metric": 'mae',
          "verbosity": -1,
         }
trn_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_test, label=y_test)

clf = lgb.train(params, trn_data, 500000, valid_sets = [trn_data, val_data], verbose_eval=1000, early_stopping_rounds=500)
pred_lgb = clf.predict(X_test)
#print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test,pred_lgb)))
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test,pred_lgb))
pred_final_lgb = clf.predict(test_data)
round_final_pred = np.around(pred_final_lgb, decimals = 2)
sub_df = pd.DataFrame({'Instance':X_test_id,'Total Yearly Income [EUR]':round_final_pred})
sub_df.to_csv('tcd-ml-1920-group-income-submission3.csv',index = None)