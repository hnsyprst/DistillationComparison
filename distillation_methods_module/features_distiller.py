""""-------------------------------"""
"""" FEATURE-BASED MODEL DISTILLER """
""""-------------------------------"""

"""
    Implements the feature-based approach to model distillation proposed by Romero et al. (2015).
    
    Romero, A., Ballas, N., Kahou, S.E., Chassang, A., Gatta, C. and Bengio, Y. (2015) 
    ‘FitNets: Hints for Thin Deep Nets’. arXiv. Available at: http://arxiv.org/abs/1412.6550 (Accessed: 31 August 2022).
"""

import torch
from torch import nn
import training_utils as utils
import network_utils as nutils
from distillation_methods_module.distiller import Distiller


if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    
# hint_layer:      a layer from the teacher model, the representation contained within which will be distilled into the student
# guided_layer:    a layer from the student model to distill a representation from the teacher into
class Features_Distiller(Distiller):
    def __init__(self, hint_layer, guided_layer, is_2D, **kwargs):
        super().__init__(**kwargs)
        self.feature_map = {}
        self.hint_layer = hint_layer
        self.guided_layer = guided_layer
        self.is_2D = is_2D

        # Split the student network into layers up to and including the guided layer and after it
        guided_layer_index = nutils.get_index_from_layer(self.student, guided_layer)
        # Layers after the guided layer
        self.freeze_list = list(self.student.modules())[guided_layer_index+1:]

        # Freeze all layers after the guided layer
        for item in self.freeze_list:
            for param in item.parameters():
                param.requires_grad = False

    # This helper function defines a hook for collecting feature maps from a given layer in a model
    def get_feature_map(self, name):
        def hook(model, input, output):
            self.feature_map[name] = output
        return hook


    """------------------------"""
    """    STAGE 1 TRAINING    """
    """ KNOWLEDGE DISTILLATION """
    """------------------------"""

    ''' Stage 1 distills the knowledge in the teacher model into the student model
        by training the student to match the output of its guided layer to the output of the hint layer '''

    # Train the student model to match its guided layer to the teacher's hint layer over one minibatch
    def train_epoch_stage_1(self, net, train_set, loss_fn, optimizer):
        # Set the model to training mode
        net.train()
        # Sum of training loss, sum of training accuracy, no. of examples
        metric = utils.Accumulator(3)

        # Iterate over the current batch
        for features, labels in train_set:
            features = features.to(device)
            labels = labels.to(device)

            # Student and teacher models make predictions
            preds = net(features)
            teacher_preds = self.teacher(features)

            # The feature maps generated by the teacher model at the hint layer for this predicton are extracted
            teacher_hints = self.feature_map['hint_layer']

            # The distance between the outputs of the regressor for the guided layer and the hint layer is calculated
            loss = loss_fn(preds, teacher_hints)

            for param in net.parameters():
                param.grad = None
                
            # Perform backprop
            loss.mean().backward()
            optimizer.step()

            # Add metrics to the accumulator
            metric.add(float(loss.sum()), utils.accuracy(torch.flatten(preds, start_dim=1), labels), labels.numel())

        # Return the metrics for this epoch
        return metric[0] / metric[2], metric[1] / metric[2]

    # Train the student model to match its guided layer to the teacher's hint layer over the given number of epochs
    def train_stage_1(self, train_set, test_set, num_epochs, wandb_log=False):
        # Define a new model using the layers of the student model up to and including the guided layer
        # and attach a regressor that will allow the hint layer to be larger than the guided layer
        class Stage_1_Student_Net(nn.Module):
            def __init__(self, student, hint_layer, guided_layer, feature_map_dict, is_2D):
                super(Stage_1_Student_Net, self).__init__()
                self.student = student
                self.feature_map_dict = feature_map_dict
                self.is_2D = is_2D

                # Create a regressor to allow the hint layer to be larger than the guided layer
                # The regressor has an input size equal to the guided layer's output size
                # and an output size equal to the hint layer's output size
                if self.is_2D:
                    self.regressor = nn.Conv2d(guided_layer.out_channels, hint_layer.out_channels, (3, 3), padding=1)
                else:
                    self.regressor = nn.Linear(guided_layer.out_features, hint_layer.out_features)
            
            def forward(self, input):
                if not self.is_2D:
                    input = torch.flatten(input, start_dim=1)
                student_out = self.student(input)
                
                guided_layer_out = self.feature_map_dict['guided_layer']
                out = self.regressor(guided_layer_out)

                return out
        
        # Instantiate a new model containing the layers of the student model up to and including the guided layer
        # attached to the regressor
        self.stage_1_student = Stage_1_Student_Net(self.student, self.hint_layer, self.guided_layer, self.feature_map, self.is_2D).to(device)
        
        # Use the helper function to attach a forward hook to the hint layer
        # this means that the outputs of this layer will be stored in the dictionary 'feature_map'
        # on every forward pass the network takes
        # the string passed to the helper function defines the feature map's key in the dict
        self.hint_layer.register_forward_hook(self.get_feature_map('hint_layer'))
        self.guided_layer.register_forward_hook(self.get_feature_map('guided_layer'))

        # Perform the first stage of model training, using 'train_epoch_stage_1' fn to train the model each epoch
        loss_fn = nn.MSELoss().to(device)
        return utils.train(self.stage_1_student, self.train_epoch_stage_1, train_set, test_set, loss_fn, num_epochs, self.optimizer, wandb_log)


    """-------------------"""
    """ STAGE 2 TRAINING  """
    """ STANDARD TRAINIING """
    """-------------------"""

    ''' Stage 2 involves standard training of the student model, now that it has recieved a 'good initialisation' from Stage 1 '''

    def train_stage_2(self, train_set, test_set, num_epochs, wandb_log=False):
        # Define a new model using the now trained layers of the student model up to and including the regressor
        # and reattach the remaining layers of the student model (those after the guided layer)
        class Stage_2_Student_Net(nn.Module):
            def __init__(self, stage_1_student):
                super(Stage_2_Student_Net, self).__init__()
                self.stage_1_student = stage_1_student
                self.is_2D = self.stage_1_student.is_2D
            
            def forward(self, input):
                if not self.is_2D:
                    input = torch.flatten(input, start_dim=1)
                out = self.stage_1_student(input)
                out = self.post_guided_layer(out)

                return out

        # Change the regressor layer from a nn.Linear to an Identity layer - a layer that performs no action
        # (this is done to essentially 'delete' the regressor layer)
        self.stage_1_student.regressor = nutils.Identity()
        # Unfreeze all layers of the student model
        for item in self.freeze_list:
            for param in item.parameters():
                param.requires_grad = True
        # Now we just train the student model as normal, with no regressor attached
        self.stage_2_student = self.student

        # Perform the second stage of model training, using 'train_epoch' fn to train the student model each epoch
        loss_fn = nn.CrossEntropyLoss(reduction='none').to(device)
        return utils.train(self.stage_2_student, self.train_epoch, train_set, test_set, loss_fn, num_epochs, self.optimizer, wandb_log)

    def train_epoch(self, student, train_set, loss_fn, optimizer):
        return utils.train_epoch(student, train_set, loss_fn, optimizer)


    """-----------------------------"""
    """ INTERFACES FOR DISTILLATION """
    """-----------------------------"""

    ''' These functions are used to begin knowledge distillation '''
    
    def train(self, train_set, test_set, num_epochs, wandb_log=False): 
        self.train_stage_1(train_set, test_set, num_epochs, wandb_log=False)
        return self.train_stage_2(train_set, test_set, num_epochs, wandb_log)