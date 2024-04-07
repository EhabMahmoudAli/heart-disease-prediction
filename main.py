import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Read the dataset
# %matplotlib inline
data = pd.read_csv('Heart_Disease.csv')

data.isnull().sum()

"""# Null Values"""

data['smoking_status'].fillna('never smoked', inplace=True)  # mode
data['work_type'].fillna('Private', inplace=True)  # mode
data['Gender'].fillna('Female', inplace=True)  # mode
data.fillna(value=data['Age'].mean(), inplace=True)

data.isnull().sum()

"""
# Convert categorical to numerical"""

# Convert categorical columns to numerical using LabelEncoder
cat_cols = ['Gender', 'work_type', 'smoking_status', 'Heart Disease']
for col in cat_cols:
    data[col] = LabelEncoder().fit_transform(data[col])

"""# outlier"""

sns.boxplot(data, palette="rainbow", orient='h')


def remove_outliers(data):
    # Loop through all columns in the dataset
    for column in data.columns:
        # Only apply outlier detection to columns with more than 2 unique values
        if data[column].nunique() > 2:
            Q1 = data[column].quantile(0.25)
            Q3 = data[column].quantile(0.75)
            IQR = Q3 - Q1
            lowerlimit = Q1 - (1.5 * IQR)
            upperlimit = Q3 + (1.5 * IQR)
            # Replace outlier valueswith NaN
            data.loc[(data[column] < lowerlimit) | (data[column] > upperlimit), column] = np.nan
    # Drop rows with NaN values
    data.dropna(inplace=True)
    return data


for i in data.columns:
    if data[i].nunique() > 2:
        remove_outliers(data)

sns.boxplot(data, palette="rainbow", orient='h')

"""# **normalization**"""

# Normalize the numeric columns using tandardScaler ****we donot normalize the categorical

# select a subset of numerical columns to normalize
num_cols = ['id', 'Age', 'Chest pain type', 'BP', 'Cholesterol', 'FBS over 120', 'EKG results', 'Max HR',
            'Exercise angina', 'ST depression', 'Slope of ST', 'Number of vessels fluro', 'Thallium']

# Normalize the numeric columns using standardScaler
scaler = StandardScaler()
data[num_cols] = scaler.fit_transform(data[num_cols])
print(data)

# Save the preprocessed dataset as a new file
data.to_csv('Heart_Disease.csv', index=False)

sns.boxplot(data, palette="rainbow", orient='h')

"""# Feature selection and Extraction

## Data correlation
"""

data_corr = data.corr()

# Heatmap
plt.figure(figsize=(20, 15))
plot = sns.heatmap(data_corr.round(3), annot=True)
plot.set_title("correlation")

# Pairplot
sns.set_style('darkgrid')
g = sns.pairplot(data)

"""## Feature Selection using filter mehtod

"""

# using filter method

# separate the features and target variable
X = data.drop(columns=['Heart Disease'])
y = data['Heart Disease']

# select the top 10 features using the f_classif test
selector = SelectKBest(f_classif, k=10)
X_new = selector.fit_transform(X, y)

# print the names of the selected features
feature_names = X.columns[selector.get_support()]
print(feature_names)

"""## LogisticRegression model"""

X = data.drop(columns=['Heart Disease'])
y = data['Heart Disease']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# create a logistic regression model
model = LogisticRegression()

# fit the model to the training data
model.fit(X_train, y_train)

# make predictions on the testing data
y_pred_test = model.predict(X_test)
# evaluate the accuracy of the test model
accuracy = accuracy_score(y_test, y_pred_test)
print(f"test Accuracy : {accuracy}")

y_pred_train = model.predict(X_train)
# evaluate the accuracy of the train model
accuracy = accuracy_score(y_train, y_pred_train)
print(f"train Accuracy : {accuracy}")

"""## SVM model"""

# create a support vector machine model
svc_model = SVC(kernel='linear')

# fit the model to the training data
svc_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_test = svc_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_test)
print(f"test Accuracy: {accuracy}")

# Make predictions on the train set
y_pred_train = svc_model.predict(X_train)
accuracy = accuracy_score(y_train, y_pred_train)
print(f"train Accuracy: {accuracy}")

"""## Decision Tree model"""

tree_model = DecisionTreeClassifier(max_depth=3)

# fit the model to the training data
tree_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_test = tree_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_test)
print(f"test Accuracy: {accuracy}")

# Make predictions on the train set
y_pred_train = tree_model.predict(X_train)
accuracy = accuracy_score(y_train, y_pred_train)
print(f"train Accuracy: {accuracy}")

"""# Make Classification Models

## Random Forest Model
"""

# separate the features and target variable
X = data.drop(columns=['Heart Disease'])
y = data['Heart Disease']

# split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=40)

# create a random forest classifier with hyperparameters
rf_model = RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_split=2)

# train the model on the training data
rf_model.fit(X_train, y_train)

y_pred_test = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_test)
print(f"test Accuracy: {accuracy}")

y_pred_train = rf_model.predict(X_train)
accuracy = accuracy_score(y_train, y_pred_train)
print(f"train Accuracy: {accuracy}")

"""## confusion matrix"""

# separate the features and target variable
X = data.drop(columns=['Heart Disease'])
y = data['Heart Disease']

# split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)

# create a logistic regression classifier
cm_model = LogisticRegression()

# train the model on the training data
cm_model.fit(X_train, y_train)

# make predictions on the testing data
y_pred = cm_model.predict(X_test)

# calculate the confusion matrix
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

# visualize the confusion matrix
conf_matrix = pd.DataFrame({"Predicted Negative": [tn, fn], "Predicted Positive": [fp, tp]},
                           index=["Actual Negative", "Actual Positive"])
sns.heatmap(conf_matrix, annot=True, cmap="Blues")

plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title("Confusion Matrix")
plt.show()

"""## classification report"""

# separate the features and target variable
X = data.drop(columns=['Heart Disease'])
y = data['Heart Disease']

# split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)

# create a logistic regression classifier
classification_model = LogisticRegression()

# train the model on the training data
classification_model.fit(X_train, y_train)

# make predictions on the testing data
y_pred_test = classification_model.predict(X_test)
report = classification_report(y_test, y_pred_test)
print("test", report)

# make predictions on the training data
y_pred_train = classification_model.predict(X_train)
report = classification_report(y_train, y_pred_train)
print("train", report)

"""## mean square error"""

# separate the features and target variable
X = data.drop(columns=['Heart Disease'])
y = data['Heart Disease']

# split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=9)

# create a linear regression model
mse_model = LinearRegression()

# train the model on the training data
mse_model.fit(X_train, y_train)

# make predictions on the testing data
y_pred_test = mse_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred_test)
print(f"test MSE: {mse}")

# make predictions on the training data
y_pred_train = mse_model.predict(X_train)
mse = mean_squared_error(y_train, y_pred_train)
print(f"train MSE: {mse}")
