import pandas as pd
import numpy as np

#Load the training data

data=pd.read_csv("train.csv")

#Display the data

data.head()

# If you need to extract the specific place name in address

data.ADDRESS = data.ADDRESS.apply(lambda x: x.split(',')[-1])

#In case you need to drop some column

data.drop('LATITUDE', axis=1, inplace=True)


# In case you need to set some column as index
data.set_index('LONGITUDE', inplace=True)

#display the shape of the data
data.shape

#Let's see if there is any missing values

data.isnull().sum()

#If there are missing values, but not significant to the total number of observations, then drop it.

data.dropna(inplace=True)

#some summary statistics
data.describe()

#know the data types
data.dtypes

#extract categorical features

cat_features=[i for i in data.columns if data.dtypes[i]=='object']

# Display categorical features
cat_features


#plot numerical columns

list(set(data.dtypes.tolist()))
df_num = data.select_dtypes(include = ['float64', 'int64'])

# display histogram
df_num.hist(bins=50)



#Label encoding of  categorical variables

from sklearn.preprocessing import OrdinalEncoder


enc=OrdinalEncoder(handle_unknown='use_encoded_value',
                                 unknown_value=-1).fit(data[cat_features])


data[cat_features]=enc.transform(data[cat_features])


# Correlation heatmap to check which features are highly correlated

import seaborn as sns

sns.heatmap(data.corr(),annot=True)



#Visualization, modelling


sns.distplot(data['TARGET(PRICE_IN_LACS)'],bins=1)








#Loading Test data
test_data=pd.read_csv("C:\\Users\\Samyajoy\\hcr_sample\\test.csv")
test_data.head()

#Do the same process as in training data

test_data.ADDRESS = test_data.ADDRESS.apply(lambda x: x.split(',')[-1])

test_data.drop('LATITUDE', axis=1, inplace=True)

test_data.set_index('LONGITUDE', inplace=True)

test_data.shape

#Let's see if there is any missing values

test_data.isnull().sum()

#If there are missing values, but not significant to the total number of observations, then drop it.

test_data.dropna(inplace=True)

test_data[cat_features]=enc.transform(test_data[cat_features])


test_data.head()


# Create outcome and input DataFrames
y = data['TARGET(PRICE_IN_LACS)'] 
X = data.drop('TARGET(PRICE_IN_LACS)', axis=1)
y.head()

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import RandomForestRegressor

#Building random forest model

model = RandomForestRegressor()
# evaluate the model
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')

# report performance
print('MAE: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))


#### If Classification

model = RandomForestRegressor()
# evaluate the model
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')

# report performance
print('Accuracy: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))




# Fit and Pridict test instances using test dataframe (tdf)
model.fit(X, y)
y_test = model.predict(test_data)
y_test

#plot the results

import matplotlib.pyplot as plt


plt.hist(y,alpha=0.5, label = "Observed Salary")
plt.hist(y_test,alpha=0.5, label = "Predicted Salary")
plt.xlabel('IDs')
plt.ylabel('Salary')
plt.title('Observed and Predicted Salary ')
plt.legend()
plt.show()


# Extract feature importance determined by RF model
feature_imp = pd.Series(model.feature_importances_, index=X.columns)
feature_imp.sort_values(ascending=True, inplace=True)

# Creating a bar plot
feature_imp.plot(kind='barh')

# Create a submission_df
d = {'id': test_data.index, 'Salary': y_test}
submission_df = pd.DataFrame(data=d)
submission_df
