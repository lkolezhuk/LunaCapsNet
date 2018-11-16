from sklearn.metrics import roc_auc_score


def get_roc(y, y_pred):
        roc = roc_auc_score(y, y_pred)
        return roc


