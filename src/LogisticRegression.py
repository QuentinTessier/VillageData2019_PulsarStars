import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import seaborn as sns

def LogReg(X_Train, Y_Train, X_Test, Y_Test):
    lr_model = LogisticRegression(random_state=42,solver="liblinear",C=1.6,penalty="l1")
    lr_model.fit(X_Train,Y_Train)
    y_head_lr = lr_model.predict(X_Test)
    lr_score = lr_model.score(X_Test,Y_Test)

    # Display confusion matrix

    cm_lr = confusion_matrix(Y_Test, y_head_lr)
    plt.suptitle("Logistic Regression's Confusion Matrix", fontsize=24)
    ax = sns.heatmap(cm_lr, cbar=False, annot=True, cmap="CMRmap_r", fmt="d")
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.xaxis.set_ticklabels(['Non-Pulsar', 'Pulsar'])
    ax.yaxis.set_ticklabels(['Non-Pulsar', 'Pulsar'])
    plt.show()
    return lr_score, y_head_lr