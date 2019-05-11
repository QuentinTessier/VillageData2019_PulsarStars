from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import seaborn as sns

def SupportVectorMachine(X_Train, Y_Train, X_Test, Y_Test):
    svm_model = SVC(random_state=42, C=250, gamma=1.6, kernel="poly", probability=True)
    svm_model.fit(X_Train, Y_Train)
    svm_head = svm_model.predict(X_Test)
    svm_score = svm_model.score(X_Test, Y_Test)

    # Display confusion matrix

    cm_svm = confusion_matrix(Y_Test, svm_head)
    plt.suptitle("Support Vector Machine's Confusion Matrix", fontsize=24)
    ax = sns.heatmap(cm_svm, cbar=False, annot=True, cmap="CMRmap_r", fmt="d")
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.xaxis.set_ticklabels(['Non-Pulsar', 'Pulsar'])
    ax.yaxis.set_ticklabels(['Non-Pulsar', 'Pulsar'])
    plt.show()
    return svm_score, svm_head