import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer

# Load training and test data
train_data = pd.read_excel('/Users/yuanweiyun/Desktop/EE6227/dataset/TrainingData.xlsx', header = None)
test_data = pd.read_excel('/Users/yuanweiyun/Desktop/EE6227/dataset/TestData.xlsx', header=None)

train_data.replace('?', float('nan'), inplace=True)

imputer = SimpleImputer(strategy='mean')  
train_data_imputed = pd.DataFrame(imputer.fit_transform(train_data), columns=train_data.columns)

X_train = train_data_imputed.iloc[:, :-1]
y_train = train_data_imputed.iloc[:, -1]
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

y_pred = clf.predict(test_data)
#print("Number of rows in test data:", len(test_data))
#print("Number of predictions:", len(y_pred))
y_pred = y_pred.astype(int)
print("Predicted class labels for test data:", y_pred)

y_pred = pd.DataFrame({'Predicted_Labels': y_pred})
y_pred.to_csv('predictions_3.csv', index=False, header = None)