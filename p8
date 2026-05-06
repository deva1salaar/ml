import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

print(cancer.feature_names)
print()
print(cancer.target_names)
print()
print(cancer.data.shape)

cancer = load_breast_cancer(as_frame=True)
cancer_data=cancer.frame
print(cancer_data)
print(cancer_data.isnull().sum())
x=cancer_data.iloc[:,:-1]
y=cancer_data.iloc[:,-1]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


dtree = DecisionTreeClassifier(max_depth=4, random_state=42)
dtree = dtree.fit(x_train, y_train)
y_pred = dtree.predict(x_test)
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_pred, y_test)
print("Accuracy:",acc)
plot_tree(dtree, filled=True, feature_names=cancer.feature_names,class_names=cancer.target_names)
plt.show()
