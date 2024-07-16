import pandas as pd
from sklearn.tree import DecisionTreeClassifier

train_data = pd.read_excel('/Users/yuanweiyun/Desktop/EE6227/dataset/TrainingData.xlsx', header = None)
test_data = pd.read_excel('/Users/yuanweiyun/Desktop/EE6227/dataset/TestData.xlsx', header = None)

train_data.replace('?', float('nan'), inplace=True)
train_data.dropna(inplace=True)

X_train = train_data.iloc[:, :-1] 
y_train = train_data.iloc[:, -1]
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

# Predict class labels of the test data
y_pred = classifier.predict(test_data)
output_df = pd.DataFrame({'Predicted_Labels': y_pred})
output_df.to_csv('predicted_labels.csv', index=False, header = None)

print("Predicted class labels for test data saved to predicted_labels.csv")
