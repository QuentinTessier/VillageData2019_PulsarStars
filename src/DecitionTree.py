import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns

def DecisionTree(X_Train, Y_Train, X_Test, Y_Test):
    dc_model = DecisionTreeClassifier(random_state=42)
    dc_model.fit(X_Train, Y_Train)
    dc_head = dc_model.predict(X_Test)
    dc_predict = dc_model.score(X_Test, Y_Test)

    # Display confusion matrix

    cm_dc = confusion_matrix(Y_Test, dc_head)
    plt.suptitle("Decision Tree Confusion Matrix", fontsize=24)
    ax = sns.heatmap(cm_dc, cbar=False, annot=True, cmap="CMRmap_r", fmt="d")
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.xaxis.set_ticklabels(['Non-Pulsar', 'Pulsar'])
    ax.yaxis.set_ticklabels(['Non-Pulsar', 'Pulsar'])
    plt.show()

    return dc_predict, dc_head