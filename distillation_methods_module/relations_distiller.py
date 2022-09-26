import torch
from torch import nn
import training_utils as utils
from distillation_methods_module.distiller import Distiller


if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


class Relations_Distiller(Distiller):
    def __init__(self, hint_layers, guided_layers, **kwargs):
        super().__init__(**kwargs)
        self.hint_layers = hint_layers
        self.guided_layers = guided_layers
        self.feature_map = {}
        
    # This helper function defines a hook for collecting feature maps from a given layer in a model
    def get_feature_map(self, name):
        def hook(model, input, output):
            self.feature_map[name] = output
        return hook

    def calculate_FSP_matrix(self, feature_map_1, feature_map_2):
        return torch.matmul(torch.flatten(feature_map_1, start_dim=1), torch.flatten(feature_map_2, start_dim=1).T)

    def train_epoch_distillation_stage1(self, student_net, teacher_net, train_iter, optimizer):
        # Set the model to training mode
        student_net.train()
        # Sum of training loss, sum of training accuracy, no. of examples
        metric = utils.Accumulator(3)

        loss_fn = nn.MSELoss().to(device)

        for features, labels in train_iter:
            features = features.to(device)
            labels = labels.to(device)

            preds = student_net(features)
            teacher_preds = teacher_net(features)

            # add the output of the hint layer to the list
            hint_start_layer_feature_map = self.feature_map['hint_start_layer']
            hint_end_layer_feature_map = self.feature_map['hint_end_layer']
            teacher_FSPs = self.calculate_FSP_matrix(hint_start_layer_feature_map, hint_end_layer_feature_map)

            guided_start_layer_feature_map = self.feature_map['guided_start_layer']
            guided_end_layer_feature_map = self.feature_map['guided_end_layer']
            student_FSPs = self.calculate_FSP_matrix(guided_start_layer_feature_map, guided_end_layer_feature_map)

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
        self.hint_layers[0].register_forward_hook(self.get_feature_map('hint_start_layer'))
        self.hint_layers[1].register_forward_hook(self.get_feature_map('hint_end_layer'))
        self.guided_layers[0].register_forward_hook(self.get_feature_map('guided_start_layer'))
        self.guided_layers[1].register_forward_hook(self.get_feature_map('guided_end_layer'))

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
            features = features.to(device)
            labels = labels.to(device)

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
        return self.train_distillation_stage2(self.student, self.teacher, train_set, test_set, num_epochs)