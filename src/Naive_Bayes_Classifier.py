import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
import seaborn as sns

def NaiveBayes(X_Train, Y_Train, X_Test, Y_Test):
    nb_model = GaussianNB()
    nb_model.fit(X_Train,Y_Train)
    y_head_nb = nb_model.predict(X_Test)
    nb_score = nb_model.score(X_Test,Y_Test)

    # Display confusion matrix

    cm_nb = confusion_matrix(Y_Test, y_head_nb)
    plt.suptitle("Naive Bayes Confusion Matrix", fontsize=24)
    ax = sns.heatmap(cm_nb, cbar=False, annot=True, cmap="CMRmap_r", fmt="d")
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.xaxis.set_ticklabels(['Non-Pulsar', 'Pulsar'])
    ax.yaxis.set_ticklabels(['Non-Pulsar', 'Pulsar'])
    plt.show()
    return nb_score, y_head_nb