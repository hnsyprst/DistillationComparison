""""------------------------------"""
"""" LOGITS-BASED MODEL DISTILLER """
""""------------------------------"""

"""
    Implements the logits-based approach to model distillation proposed by Hinton et al. (2014).

    Hinton, G., Vinyals, O. and Dean, J. (2014)
    ‘Distilling the Knowledge in a Neural Network’. arXiv. Available at: http://arxiv.org/abs/1503.02531 (Accessed: 5 July 2022).
"""

import torch
from torch import nn
import training_utils as utils
from distillation_methods_module.distiller import Distiller


if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


# temp:     controls the 'softness' of both the student and teacher logits---higher values create
#           a softer distribution across the logits, a value of 1 creates normal 'hard' logits.
class Logits_Distiller(Distiller):
    def __init__(self, temp, hard_loss_weight, **kwargs):
        super().__init__(**kwargs)
        self.temp = temp
        self.hard_loss_weight = hard_loss_weight

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

    def train_epoch(self, net, train_set, loss_fn, optimizer):
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

            # The loss between the student and teacher soft logits is calculated
            hard_loss_weight = self.hard_loss_weight
            soft_loss_weight = 1 - self.hard_loss_weight

            loss = ((loss_fn(student_preds, teacher_preds, temperature = self.temp) * (soft_loss_weight)) + (self.ce_loss(features, labels) * hard_loss_weight)) / 2

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
        return utils.train(self.student, self.train_epoch, train_set, test_set, self.soft_targets_loss, num_epochs, self.optimizer, wandb_log)