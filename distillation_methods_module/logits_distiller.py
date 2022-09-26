import torch
from torch import nn
import training_utils as utils
from distillation_methods_module.distiller import Distiller


if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


class Logits_Distiller(Distiller):
    def __init__(self, temp, **kwargs):
        super().__init__(**kwargs)
        self.temp = temp

        self.softmax_op = nn.Softmax(dim=1)
        self.mseloss_fn = nn.MSELoss()   
    
    def my_loss(self, preds, targets, temperature = 5):
        soft_pred = self.softmax_op(preds / temperature)
        soft_targets = self.softmax_op(targets / temperature)
        loss = self.mseloss_fn(soft_pred, soft_targets)
        return loss 

    def train_epoch(self, train_set, loss_fn):
        # Set the model to training mode
        self.student.train()
        # Sum of training loss, sum of training accuracy, no. of examples
        metric = utils.Accumulator(3)

        for features, labels in train_set:
            features = features.to(device)
            labels = labels.to(device)

            student_preds = self.student(features)
            teacher_preds = self.teacher(features)

            loss = loss_fn(student_preds, teacher_preds, temperature = self.temp)
            for param in self.student.parameters():
                param.grad = None
            loss.mean().backward()
            self.optimizer.step()

            metric.add(float(loss.sum()), utils.accuracy(student_preds, labels), labels.numel())

        return metric[0] / metric[2], metric[1] / metric[2]

    def train(self, train_set, test_set, num_epochs): 
        # Arrays for logging model history
        history_train_accuracy = []
        history_train_loss = []
        history_test_accuracy = []

        loss_fn = self.my_loss

        for epoch in range(num_epochs):
            train_metrics = self.train_epoch(train_set, loss_fn)
            test_acc = utils.evaluate_accuracy(self.student, test_set)

            # Log the model history
            history_train_loss.append(train_metrics[0])
            history_train_accuracy.append(train_metrics[1])
            history_test_accuracy.append(test_acc)

            # Print epoch, training accuracy and loss and validation accuracy
            print('Epoch:\t\t ', epoch)
            print('train_metrics:\t ', train_metrics)
            print('test_accuracy:\t ', test_acc)
            
        return history_train_accuracy, history_train_loss, history_test_accuracy