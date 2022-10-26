import torch
from torch import nn
import torchmetrics

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

def accuracy(y_hat, y):  #y_hat is a matrix; 2nd dimension stores prediction scores for each class.
    """Compute the number of correct predictions."""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1) # Predicted class is the index of max score     

    #print(y_hat.shape)
    #print(y.shape)

    cmp = (y_hat.type(y.dtype) == y)  # because`==` is sensitive to data types
    return float(cmp.type(y.dtype).sum()) # Taking the sum yields the number of correct predictions.

# From the d2l textbook

class Accumulator:
    """For accumulating sums over `n` variables."""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def evaluate_accuracy(net, data_iter, top_k=1): 
    net.eval()
    """Compute the accuracy for a model on a dataset."""
    metric = Accumulator(1)  # No. of correct predictions, no. of predictions
    with torch.no_grad():
        for features, labels in data_iter:
            features = features.to(device)
            labels = labels.to(device)
            preds = net(features)
            metric.add(torchmetrics.functional.accuracy(preds, labels, top_k=top_k))
    return metric[0] / len(data_iter)

def train_epoch(net, train_iter, loss_fn, optimizer):
    # Set the model to training mode
    net.train()
    # Sum of training loss, sum of training accuracy, no. of examples
    metric = Accumulator(3)

    for features, labels in train_iter:
        features = features.to(device)
        labels = labels.to(device)

        preds = net(features)
        loss = loss_fn(preds, labels)
        for param in net.parameters():
            param.grad = None
        loss.mean().backward()
        nn.utils.clip_grad_norm_(net.parameters(), max_norm=2.0, norm_type=2)
        optimizer.step()

        metric.add(float(loss.sum()), accuracy(preds, labels), labels.numel())

    return metric[0] / metric[2], metric[1] / metric[2]

def train(net, train_epoch_fn, train_iter, test_iter, loss_fn, num_epochs, optimizer, wandb_log=False): 
    if wandb_log:
        import wandb

    # Arrays for logging model history
    history_train_accuracy = []
    history_train_loss = []
    history_test_accuracy = []

    for epoch in range(num_epochs):
        train_metrics = train_epoch_fn(net, train_iter, loss_fn, optimizer)
        test_acc = evaluate_accuracy(net, test_iter)

        # Log the model history
        history_train_loss.append(train_metrics[0])
        history_train_accuracy.append(train_metrics[1])
        history_test_accuracy.append(test_acc)

        # Print epoch, training accuracy and loss and validation accuracy
        print('Epoch:\t\t ', epoch)
        print('train_metrics:\t ', train_metrics)
        print('test_accuracy:\t ', test_acc)

        if wandb_log:
            wandb.log({"loss": train_metrics[0],
                        "train_accuracy": train_metrics[1],
                        "val_accuracy": test_acc})
        
    return history_train_accuracy, history_train_loss, history_test_accuracy