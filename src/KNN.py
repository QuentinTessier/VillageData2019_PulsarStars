import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns

def KNN(X_Train, Y_Train, X_Test, Y_Test):
    knn_model = KNeighborsClassifier(n_neighbors=7, weights='distance')
    knn_model.fit(X_Train, Y_Train)
    knn_head = knn_model.predict(X_Test)
    knn_score = knn_model.score(X_Test, Y_Test)

    # Display confusion matrix

    cm_knn = confusion_matrix(Y_Test, knn_head)
    plt.suptitle("K Nearest Neighbors Confusion Matrix", fontsize=24)
    ax = sns.heatmap(cm_knn, cbar=False, annot=True, cmap="CMRmap_r", fmt="d")
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.xaxis.set_ticklabels(['Non-Pulsar', 'Pulsar'])
    ax.yaxis.set_ticklabels(['Non-Pulsar', 'Pulsar'])
    plt.show()
    return knn_head, knn_score