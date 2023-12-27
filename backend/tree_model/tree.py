import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import precision_score, recall_score, fbeta_score

import joblib


if __name__ == "__main__":
    # DECISION TREE CLASSIFIER MODEL
    
    df = pd.read_csv('data.csv')

    df.dropna(inplace=True)


    # Extract features and target
    X = df[['BlackToWhiteRatio', 'Height', 'Width', 'WidthToHeight', "Class"]]
    y = df['IsGood']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the DecisionTreeClassifier
    dt_classifier = DecisionTreeClassifier(random_state=42)

    # param_grid_decision_tree = {
    #     'criterion': ['gini', 'entropy'],
    #     'max_depth': [None, 2, 3, 4, 5],
    #     'min_samples_split': [1, 3, 5, 7, 10],
    #     'min_samples_leaf': [1, 2, 3, 4]
    # }

    # # Create the GridSearchCV object
    # grid_search = GridSearchCV(estimator=dt_classifier, param_grid=param_grid_decision_tree, cv=5, scoring='accuracy')

    # # Fit the model to the data
    # grid_search.fit(X_train, y_train)

    # # Print the best parameters and the corresponding accuracy
    # print("Best Parameters: ", grid_search.best_params_)

    #classifier with best params
    model = DecisionTreeClassifier(random_state=42, max_depth=2,  min_samples_leaf=1, min_samples_split=3, criterion='gini')
    model.fit(X_train, y_train)

    # Predict the target output on the test set
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    fbeta = fbeta_score(y_test, y_pred, beta=1)

    print(f"{precision:.2f}, {recall:.2f}, {fbeta:.2f}")


    plt.figure(figsize=(12, 8))
    plot_tree(model, feature_names=['BlackToWhiteRatio', 'Height', 'Width', 'WidthToHeight', 'IsGood'], class_names=['0', '1'], filled=True, rounded=True)
    plt.savefig("tree.png")
    plt.show()
    

    joblib.dump(model, '../trained_models/decision_tree_model.joblib')


