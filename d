#4
import pandas as pd
cols=["PlayTennis","Outlook","Temperature","Humidity","Wind"]
df= pd.read_csv("tennis.csv",header=0,names=cols)
print(df)
features=cols[1:]
from sklearn.preprocessing import LabelEncoder
Le=LabelEncoder()
df['Outlook'] = Le.fit_transform(df['Outlook'])
df['Temperature'] = Le.fit_transform(df['Temperature'])
df['Humidity'] = Le.fit_transform(df['Humidity'])
df['Wind'] = Le.fit_transform(df['Wind'])
df['PlayTennis'] = Le.fit_transform(df['PlayTennis'])
x=df[features]
y=df.PlayTennis
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,
test_size=0.2,
random_state=100)
from sklearn.tree import DecisionTreeClassifier
id3 = DecisionTreeClassifier()
id3.fit(x_train, y_train)
y_pred = id3.predict(x_test)
from sklearn import tree
text = tree.export_text(id3)
print(text)
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(25,20))
tree.plot_tree(id3,feature_names=features,class_names=['Yes','No'],filled=True)
print("Actual output: ")
print(y_test)
print("Predicted output: ")
print(y_pred)
from sklearn.metrics import confusion_matrix, accuracy_score
print("Confusion Matrix: ")
print(confusion_matrix(y_test, y_pred))
print("Accuracy score: ")
print(accuracy_score(y_test, y_pred) * 100)
