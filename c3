import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("C:/Users/gagan/Downloads/Iris.csv")
X = df.iloc[:, 1:-1]  # Features (exclude Id)
y = df.iloc[:, -1]    # Target (Species)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Base classifier
dt = DecisionTreeClassifier(max_depth=3, random_state=42)

# Bagging
bag = BaggingClassifier(
    estimator=dt, n_estimators=10, random_state=42
)
bag.fit(X_train, y_train)
pred1 = bag.predict(X_test)

# Boosting
boost = AdaBoostClassifier(
    estimator=dt, n_estimators=10, random_state=42
)
boost.fit(X_train, y_train)
pred2 = boost.predict(X_test)

# Compare accuracy
print("Bagging Accuracy:", accuracy_score(y_test, pred1))
print("Boosting Accuracy:", accuracy_score(y_test, pred2))
