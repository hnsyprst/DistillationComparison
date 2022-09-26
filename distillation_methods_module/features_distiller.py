import torch
from torch import nn
import training_utils as utils
import network_utils as nutils
from distillation_methods_module.distiller import Distiller
import numpy as np


if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    

class Features_Distiller(Distiller):
    def __init__(self, hint_layer, hinted_layer, **kwargs):
        super().__init__(**kwargs)
        self.feature_map = {}
        self.hint_layer = hint_layer
        self.hinted_layer = hinted_layer

        hinted_layer_index = nutils.get_index_from_layer(self.student, hinted_layer)

        # Split the student network into layers before the hint layer and after
        # and add the regressor to the first half of the network
        modules = list(self.student.children())[:hinted_layer_index+1]
        self.prehint_student = nn.Sequential(*modules)
        modules = list(self.student.children())[hinted_layer_index+1:]
        self.posthint_student = nn.Sequential(*modules)


    # This helper function defines a hook for collecting feature maps from a given layer in a model
    def get_feature_map(self, name):
        def hook(model, input, output):
            self.feature_map[name] = output.detach()
        return hook


    def train_stage_1(self, train_set, test_set, num_epochs):
        class Stage_1_Student_Net(nn.Module):
            def __init__(self, prehint, hint_layer, hinted_layer):
                super(Stage_1_Student_Net, self).__init__()
                self.prehint = prehint
                self.regressor = nn.Linear(hinted_layer.out_features, hint_layer.out_features)
            
            def forward(self, input):
                input = torch.flatten(input, start_dim=1)
                out = self.prehint(input)
                out = self.regressor(out)

                return out
        
        self.stage_1_student = Stage_1_Student_Net(self.prehint_student, self.hint_layer, self.hinted_layer).to(device)
        self.hint_layer.register_forward_hook(self.get_feature_map('hint_layer'))

        loss_fn = nn.MSELoss().to(device)

        # Arrays for logging model history
        history_train_accuracy = []
        history_train_loss = []
        history_test_accuracy = []

        for epoch in range(num_epochs):
            train_metrics = self.train_epoch_stage_1(self.stage_1_student, self.teacher, train_set, loss_fn, self.optimizer)
            test_acc = utils.evaluate_accuracy(self.stage_1_student, test_set)

            # Log the model history
            history_train_loss.append(train_metrics[0])
            history_train_accuracy.append(train_metrics[1])
            history_test_accuracy.append(test_acc)

            # Print epoch, training accuracy and loss and validation accuracy
            print('Epoch:\t\t ', epoch)
            print('train_metrics:\t ', train_metrics)
            print('test_accuracy:\t ', test_acc)
            
        return history_train_accuracy, history_train_loss, history_test_accuracy

    def train_epoch_stage_1(self, student_net, teacher_net, train_set, loss_fn, optimizer):
        # Set the model to training mode
        student_net.train()
        # Sum of training loss, sum of training accuracy, no. of examples
        metric = utils.Accumulator(3)

        for features, labels in train_set:
            # initialise a list to store the outputs of the hint layer this batch
            feature_map_list = []

            features = features.to('cuda')
            labels = labels.to('cuda')

            preds = student_net(features)
            teacher_preds = teacher_net(features)

            # add the output of the hint layer to the list
            # this bit could definitely use some cleaning up
            feature_map_list.append(self.feature_map['hint_layer'].cpu().numpy())
            feature_map_list = np.concatenate(feature_map_list)
            teacher_hints = torch.from_numpy(feature_map_list).to('cuda')

            # calculate the loss between the outputs of the regressor for the guided layer and the hint layer
            loss = loss_fn(preds, teacher_hints)

            for param in student_net.parameters():
                param.grad = None
                
            loss.mean().backward()
            optimizer.step()

            metric.add(float(loss.sum()), utils.accuracy(preds, labels), labels.numel())

        return metric[0] / metric[2], metric[1] / metric[2]

    def train_stage_2(self, train_set, test_set, num_epochs):
        class Stage_2_Student_Net(nn.Module):
            def __init__(self, stage_1_student, posthint):
                super(Stage_2_Student_Net, self).__init__()
                self.stage_1_student = stage_1_student
                self.posthint = posthint
            
            def forward(self, input):
                input = torch.flatten(input, start_dim=1)
                out = self.stage_1_student(input)
                out = self.posthint(out)

                return out
        
        class Identity(nn.Module):
            def __init__(self):
                super(Identity, self).__init__()
                
            def forward(self, x):
                return x

        self.stage_1_student.regressor = Identity()

        self.stage_2_student = Stage_2_Student_Net(self.stage_1_student, self.posthint_student).to(device)

        loss_fn = nn.CrossEntropyLoss(reduction='none').to(device)
        
        # Arrays for logging model history
        history_train_accuracy = []
        history_train_loss = []
        history_test_accuracy = []

        for epoch in range(num_epochs):
            train_metrics = self.train_epoch(train_set, loss_fn)
            test_acc = utils.evaluate_accuracy(self.stage_2_student, test_set)

            # Log the model history
            history_train_loss.append(train_metrics[0])
            history_train_accuracy.append(train_metrics[1])
            history_test_accuracy.append(test_acc)

            # Print epoch, training accuracy and loss and validation accuracy
            print('Epoch:\t\t ', epoch)
            print('train_metrics:\t ', train_metrics)
            print('test_accuracy:\t ', test_acc)
            
        return history_train_accuracy, history_train_loss, history_test_accuracy

    def train(self, train_set, test_set, num_epochs): 
        self.train_stage_1(train_set, test_set, num_epochs)
        self.train_stage_2(train_set, test_set, num_epochs)

    def train_epoch(self, train_set, loss_fn):
        # Set the model to training mode
        self.stage_2_student.train()
        # Sum of training loss, sum of training accuracy, no. of examples
        metric = utils.Accumulator(3)

        for features, labels in train_set:
            features = features.to('cuda')
            labels = labels.to('cuda')

            preds = self.stage_2_student(features)
            loss = loss_fn(preds, labels)
            for param in self.stage_2_student.parameters():
                param.grad = None
            loss.mean().backward()
            self.optimizer.step()

            metric.add(float(loss.sum()), utils.accuracy(preds, labels), labels.numel())

        return metric[0] / metric[2], metric[1] / metric[2]