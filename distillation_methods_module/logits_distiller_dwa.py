""""------------------------------"""
"""" LOGITS-BASED MODEL DISTILLER """
""""------------------------------"""

"""
    Implements the logits-based approach to model distillation proposed by Hinton et al. (2014),
    adding the dynamic weight average technique proposed by Liu et al. (2019).

    Dynamic weight average implementation modified from Liu et al.'s repository (2022).
    
    Some code for training procedures modified from the Dive into Deep Learning textbook (Zhang et al., 2021).

    REFERENCES:
    Hinton, G., Vinyals, O. and Dean, J. (2014)
    ‘Distilling the Knowledge in a Neural Network’. arXiv. Available at: http://arxiv.org/abs/1503.02531 (Accessed: 5 July 2022).
    
    Liu, S., Johns, E. and Davison, A.J. (2019)
    ‘End-to-End Multi-Task Learning with Attention’. arXiv. Available at: http://arxiv.org/abs/1803.10704 (Accessed: 31 October 2022).

    Liu, S., Johns, E. and Davison, A.J. (2022)
    ‘mtan/utils.py at master · lorenmt/mtan’. Imperial College London. Available at: https://github.com/lorenmt/mtan (Accessed: 20 November 2022).

    Zhang, A., Lipton, Z.C., Li, M. and Smola, A.J. (2021)
    Dive into Deep Learning. Available at: https://d2l.ai/ (Accessed: 20 November 2022).
"""

import torch
from torch import nn
import training_utils as utils
from distillation_methods_module.distiller import Distiller
import numpy as np


if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


# temp:             Temperature hyperparameter, controls the 'softness' of both the student and teacher logits---higher values create
#                   a softer distribution across the logits, a value of 1 creates normal 'hard' logits.
# weight_temp:      Temperature hyperparameter for dynamic weight average
class Logits_Distiller_DWA(Distiller):
    def __init__(self, temp, weight_temp, **kwargs):
        super().__init__(**kwargs)
        self.temp = temp
        self.weight_temp = weight_temp

        self.softmax_op = nn.Softmax(dim=1)
        self.ce_loss = nn.CrossEntropyLoss(reduction='none').to(device)
    
    # Custom loss function softens the distribution of the student and teacher logits and
    # calculates the L2 distance between them
    def soft_targets_loss(self, preds, targets, temperature = 5):
        soft_pred = self.softmax_op(preds / temperature)
        soft_targets = self.softmax_op(targets / temperature)
        loss = self.ce_loss(soft_pred, soft_targets)
        return loss 

    """-----------------------------"""
    """ INTERFACES FOR DISTILLATION """
    """-----------------------------"""

    ''' This method has a single stage, so these interfaces perform the entirety of knowledge distiillation '''

    # Some code for training procedure modified from the Dive into Deep Learning textbook (Zhang et al., 2021)
    # Code for calculating dynamic weight average modified from Liu et al.'s implementation (2022)
    def train_distillation(self, net, train_epoch_fn, train_iter, test_iter, loss_fn, num_epochs, optimizer, wandb_log=False, calc_val_accuracy=True): 
        if wandb_log:
            import wandb

        # Arrays for logging model history
        history_train_accuracy = []
        history_train_loss = []
        history_test_accuracy = []
        
        T = self.weight_temp
        
        # Stores information for dynamic weight average
        self.avg_cost = np.zeros([num_epochs, 2], dtype=np.float32)
        self.lambda_weight = np.ones([2, num_epochs])

        # Calculate dynamic weight average
        for epoch in range(num_epochs):
            if epoch == 0 or epoch == 1:
                self.lambda_weight[0, epoch] = 1.0
                self.lambda_weight[1, epoch] = 0.2
            else:
                w_1 = self.avg_cost[epoch - 1, 0] / self.avg_cost[epoch - 2, 0]
                w_2 = self.avg_cost[epoch - 1, 1] / self.avg_cost[epoch - 2, 1]
                self.lambda_weight[0, epoch] = 2 * np.exp(w_1 / T) / (np.exp(w_1 / T) + np.exp(w_2 / T))
                self.lambda_weight[1, epoch] = 2 * np.exp(w_2 / T) / (np.exp(w_1 / T) + np.exp(w_2 / T))

            train_metrics = train_epoch_fn(net, train_iter, loss_fn, optimizer, epoch, self.lambda_weight)
            if calc_val_accuracy:
                test_acc = utils.evaluate_accuracy(net, test_iter)
            else:
                test_acc = 0

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

    # Some code for training procedure modified from the Dive into Deep Learning textbook (Zhang et al., 2021)
    def train_epoch(self, net, train_set, loss_fn, optimizer, epoch, lambda_weight):
        # Set the model to training mode
        net.train()
        # Sum of training loss, sum of training accuracy, no. of examples
        metric = utils.Accumulator(3)

        # Iterate over the current batch
        for features, labels in train_set:
            features = features.to(device)
            labels = labels.to(device)

            # Student and teacher models make predictions
            student_preds = net(features)
            teacher_preds = self.teacher(features)
            
            # Calculation modified from Hu's implementation (2022)
            soft_loss = nn.functional.kl_div(nn.functional.log_softmax(student_preds/self.temp, dim=1),
						nn.functional.softmax(teacher_preds/self.temp, dim=1),
						reduction='batchmean') * self.temp * self.temp
            hard_loss = self.ce_loss(student_preds, labels)
            hard_loss = torch.mean(hard_loss)

            self.avg_cost[epoch, 0] += soft_loss.item() / len(train_set)
            self.avg_cost[epoch, 1] += hard_loss.item() / len(train_set)

            loss = (lambda_weight[0, epoch] * soft_loss) + (lambda_weight[1, epoch] * hard_loss) / 2
            
            for param in net.parameters():
                param.grad = None
                
            # Perform backprop
            loss.mean().backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_norm=2.0, norm_type=2)
            optimizer.step()

            # Add metrics to the accumulator
            metric.add(float(loss.sum()), utils.accuracy(student_preds, labels), labels.numel())

        # Return the metrics for this epoch
        return metric[0] / metric[2], metric[1] / metric[2]

    def train(self, train_set, test_set, num_epochs, wandb_log=False): 
        # Perform knowledge distillation, using 'train_epoch' fn to train the student model each epoch
        return self.train_distillation(self.student, self.train_epoch, train_set, test_set, self.soft_targets_loss, num_epochs, self.optimizer, wandb_log)