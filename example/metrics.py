
def calculate_metrics(model_outputs, gold_labels=None):
    model_outputs = (model_outputs > 0.5).float()
    correct = (model_outputs == gold_labels).sum()
    acc = correct / gold_labels.shape[0]
    return {'acc': acc.item()}
