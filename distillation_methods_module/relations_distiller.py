import torch
from torch import nn
import training_utils as utils
from distillation_methods_module.distiller import Distiller
import numpy as np


if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


class Relations_Distiller(Distiller):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.feature_map = {}
        
    # This helper function defines a hook for collecting feature maps from a given layer in a model
    def get_feature_map(self, name):
        def hook(model, input, output):
            self.feature_map[name] = output.detach()
        return hook

    def calculate_FSP_matrix(self, feature_map_1, feature_map_2):
        return torch.matmul(torch.flatten(feature_map_1, start_dim=1), torch.flatten(feature_map_2, start_dim=1).T)

    def get_feature_maps(self, feature_map_list, feature_map_dict, name):
        feature_map_list.append(feature_map_dict[name].cpu().numpy())
        feature_map_list = np.concatenate(feature_map_list)
        return torch.tensor(feature_map_list, requires_grad=True).to('cuda')

    def train_epoch_distillation_stage1(self, student_net, teacher_net, train_iter, optimizer):
        # Set the model to training mode
        student_net.train()
        # Sum of training loss, sum of training accuracy, no. of examples
        metric = utils.Accumulator(3)

        loss_fn = nn.MSELoss().to(device)

        for features, labels in train_iter:
            # initialise a list to store the outputs of the hint layer this batch
            teacher_feature_map_1_list = []
            teacher_feature_map_2_list = []
            student_feature_map_1_list = []
            student_feature_map_2_list = []

            features = features.to('cuda')
            labels = labels.to('cuda')

            preds = student_net(features)
            teacher_preds = teacher_net(features)

            # add the output of the hint layer to the list
            # this bit could definitely use some cleaning up
            teacher_feature_maps_1 = self.get_feature_maps(teacher_feature_map_1_list, self.feature_map, 'teacher_linear_2')
            teacher_feature_maps_2 = self.get_feature_maps(teacher_feature_map_1_list, self.feature_map, 'teacher_linear_3')
            teacher_FSPs = self.calculate_FSP_matrix(teacher_feature_maps_1, teacher_feature_maps_2)

            student_feature_maps_1 = self.get_feature_maps(student_feature_map_1_list, self.feature_map, 'student_linear_2')
            student_feature_maps_2 = self.get_feature_maps(student_feature_map_1_list, self.feature_map, 'student_linear_3')
            student_FSPs = self.calculate_FSP_matrix(student_feature_maps_1, student_feature_maps_2)

            # calculate the loss between the outputs of the regressor for the guided layer and the hint layer
            loss = loss_fn(student_FSPs, teacher_FSPs)

            for param in student_net.parameters():
                param.grad = None
                
            loss.mean().backward()
            optimizer.step()

            metric.add(float(loss.sum()), utils.accuracy(preds, labels), labels.numel())

        return metric[0] / metric[2], metric[1] / metric[2]

    def train_distillation_stage1(self, student_net, teacher_net, train_iter, test_iter, num_epochs): 
        # The helper function is used to attach a forward hook to the layer linear_2
        # this means that the outputs of the layer linear_2 will be stored in the dictionary 'feature_map'
        # on every forward pass the network takes
        # in this case, the outputs will be stored in the dictionary with the key 'linear_2'
        teacher_net.linear_2.register_forward_hook(self.get_feature_map('teacher_linear_2'))
        teacher_net.linear_3.register_forward_hook(self.get_feature_map('teacher_linear_3'))
        student_net.linear_2.register_forward_hook(self.get_feature_map('student_linear_2'))
        student_net.linear_3.register_forward_hook(self.get_feature_map('student_linear_3'))

        # Arrays for logging model history
        history_train_accuracy = []
        history_train_loss = []
        history_test_accuracy = []

        for epoch in range(num_epochs):
            train_metrics = self.train_epoch_distillation_stage1(student_net, teacher_net, train_iter, self.optimizer)
            test_acc = utils.evaluate_accuracy(student_net, test_iter)

            # Log the model history
            history_train_loss.append(train_metrics[0])
            history_train_accuracy.append(train_metrics[1])
            history_test_accuracy.append(test_acc)

            # Print epoch, training accuracy and loss and validation accuracy
            print('Epoch:\t\t ', epoch)
            print('train_metrics:\t ', train_metrics)
            print('test_accuracy:\t ', test_acc)
            
        return history_train_accuracy, history_train_loss, history_test_accuracy

    def train_distillation_stage2(self, student_net, teacher_net, train_iter, test_iter, num_epochs):
        loss_fn = nn.CrossEntropyLoss(reduction='none').to(device)

        # Arrays for logging model history
        history_train_accuracy = []
        history_train_loss = []
        history_test_accuracy = []

        for epoch in range(num_epochs):
            train_metrics = self.train_epoch(train_iter, loss_fn)
            test_acc = utils.evaluate_accuracy(student_net, train_iter)

            # Log the model history
            history_train_loss.append(train_metrics[0])
            history_train_accuracy.append(train_metrics[1])
            history_test_accuracy.append(test_acc)

            # Print epoch, training accuracy and loss and validation accuracy
            print('Epoch:\t\t ', epoch)
            print('train_metrics:\t ', train_metrics)
            print('test_accuracy:\t ', test_acc)
            
        return history_train_accuracy, history_train_loss, history_test_accuracy

    def train_epoch(self, train_set, loss_fn):
        # Set the model to training mode
        self.student.train()
        # Sum of training loss, sum of training accuracy, no. of examples
        metric = utils.Accumulator(3)

        for features, labels in train_set:
            features = features.to('cuda')
            labels = labels.to('cuda')

            preds = self.student(features)
            loss = loss_fn(preds, labels)
            for param in self.student.parameters():
                param.grad = None
            loss.mean().backward()
            self.optimizer.step()

            metric.add(float(loss.sum()), utils.accuracy(preds, labels), labels.numel())

        return metric[0] / metric[2], metric[1] / metric[2]

    def train(self, train_set, test_set, num_epochs):
        self.train_distillation_stage1(self.student, self.teacher, train_set, test_set, num_epochs)
        self.train_distillation_stage2(self.student, self.teacher, train_set, test_set, num_epochs)