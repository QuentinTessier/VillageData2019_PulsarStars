from sklearn.tree import DecisionTreeClassifier

def DecisionTree(X_Train, Y_Train, X_Test, Y_Test):
    dc_model = DecisionTreeClassifier(random_state=42)
    dc_model.fit(X_Train, Y_Train)
    dc_predicted = dc_model.predict(X_Test)
    return dc_model.score(X_Test, Y_Test)