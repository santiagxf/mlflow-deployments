from typing import Dict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def compute_classification_metrics(pred: Dict) -> Dict[str, float]:
    """
    Computes the metrics given predictions of a torch model

    Parameters
    ----------
    pred: Dict
        Predictions returned from the model.

    Returns
    -------
    Dict[str, float]:
        The metrics computed, inclusing `accuracy`, `f1`, `precision`, `recall` and `support`.
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, support = precision_recall_fscore_support(labels, preds, average='weighted')
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'support': support
    }
