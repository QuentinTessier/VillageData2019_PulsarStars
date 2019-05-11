from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns

def RandomForest(X_Train, Y_Train, X_Test, Y_Test):
    rfc_model = RandomForestClassifier(n_estimators=37, random_state=42, max_leaf_nodes=200, criterion="entropy")
    rfc_model.fit(X_Train, Y_Train)
    rfc_head = rfc_model.predict(X_Test)
    rfc_score = rfc_model.score(X_Test, Y_Test)

    # Display confusion matrix

    cm_rfc = confusion_matrix(Y_Test, rfc_head)
    plt.suptitle("Random Forest's Confusion Matrix", fontsize=24)
    ax = sns.heatmap(cm_rfc, cbar=False, annot=True, cmap="CMRmap_r", fmt="d")
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.xaxis.set_ticklabels(['Non-Pulsar', 'Pulsar'])
    ax.yaxis.set_ticklabels(['Non-Pulsar', 'Pulsar'])
    plt.show()
    return rfc_score, rfc_head