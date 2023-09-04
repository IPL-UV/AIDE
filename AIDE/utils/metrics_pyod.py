import numpy as np
from numpy import percentile
from sklearn.utils import column_or_1d
from sklearn.utils import check_consistent_length
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import average_precision_score

def evaluate_print(clf_name, y, scores, threshold=0.5):
    """Utility function for evaluating and printing the results for examples.
    Default metrics include ROC, Accuracy, Precision, Recall and Average Precision. 
    """

    y = column_or_1d(y)
    scores = column_or_1d(scores)
    check_consistent_length(y, scores)

    y_pred = (scores > threshold).astype('int')

    print('{clf_name} ROC:{roc}, accuracy:{acc},  precision:{prn}, recall:{rc}, AUC-PR:{avg_pr}'.format(
        clf_name=clf_name,
        roc=np.round(roc_auc_score(y, scores), decimals=4),
        acc=np.round(accuracy_score(y, y_pred), decimals=4),
        rc=np.round(recall_score(y, y_pred), decimals=4),
        avg_pr = np.round(average_precision_score(y, y_pred), decimals=4),
        prn=np.round(precision_score(y, y_pred), decimals=4)))
