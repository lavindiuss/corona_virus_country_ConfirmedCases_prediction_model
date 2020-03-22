import numpy as np 
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

import pandas as pd 

df = pd.read_csv('train.csv')
df = df.fillna(0)
test_df = pd.read_csv('test.csv')
test_df = test_df.fillna(0)

df["Date"] = df["Date"].apply(lambda x: x.replace("-",""))
df["Date"].astype('int32').dtypes
test_df["Date"] = test_df["Date"].apply(lambda x: x.replace("-",""))
test_df["Date"].astype('int32').dtypes

x_test =test_df[['Lat','Long','Date']]

print(x_test.shape)
print(df.head())
print("___________________________________")

x = df[['Lat','Long','Date']]
y = df[['ConfirmedCases']]


clf1 = DecisionTreeClassifier()
clf1.fit(x,y) 

# clf2 = LinearRegression()
# clf2.fit(x,y) 

# clf3 = LogisticRegression(multi_class="auto")
# clf3.fit(x,y) 

#this model wins 
predictions1 = clf1.predict(x_test)

# predictions2 = clf2.predict(x_test)
# predictions3 = clf3.predict(x_test)

prediction1 = pd.DataFrame(predictions1, columns=['predictions']).to_csv('prediction1.csv')

# prediction2 = pd.DataFrame(predictions2, columns=['predictions']).to_csv('prediction2.csv')
# prediction3 = pd.DataFrame(predictions3, columns=['predictions']).to_csv('prediction3.csv')
