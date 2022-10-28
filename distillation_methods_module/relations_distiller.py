""""--------------------------------"""
"""" RELATION-BASED MODEL DISTILLER """
""""--------------------------------"""

"""
    Implements the relations-based approach to model distillation proposed by Yim et al. (2017).

    Yim, J., Joo, D., Bae, J. and Kim, J. (2017)
    ‘A Gift from Knowledge Distillation: Fast Optimization, Network Minimization and Transfer Learning’,
    in 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Honolulu, HI: IEEE, pp. 7130–7138.
    Available at: https://doi.org/10.1109/CVPR.2017.754  (Accessed: 27 September 2022). 
"""

import torch
from torch import nn
from torchvision.models.feature_extraction import create_feature_extractor
import training_utils as utils
from distillation_methods_module.distiller import Distiller


if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


# hint_layers:      a list of 2-tuples, each containing the start layer and end layer of a hint module (the block of layers that
#                   encapsulate the relations the student should learn)
# guided_layers:    a list of 2-tuples, each containing the start layer and end layer of the module to be guided by the hint module
class Relations_Distiller(Distiller):
    def __init__(self, hint_layers, guided_layers, **kwargs):
        super().__init__(**kwargs)
        self.hint_layers = hint_layers
        self.guided_layers = guided_layers
        self.hint_feature_extractor = {}
        self.guided_feature_extractor = {}
        
    # This helper function defines a hook for collecting feature maps from a given layer in a model
    def get_feature_map(self, name):
        def hook(model, input, output):
            self.feature_map[name] = output
        return hook

    # Calculate the flow of solution procedure (FSP) matrix of the given feature maps
    # (same calculation as for a Gram matrix but between two feature maps instead of one and its transpose)
    # code modified from https://github.com/AberHu/Knowledge-Distillation-Zoo
    def calculate_FSP_matrix(self, feature_map_1, feature_map_2):
        if feature_map_1.size(2) > feature_map_2.size(2):
            feature_map_1 = F.adaptive_max_pool2d(feature_map_1, (feature_map_2.size(2), feature_map_2.size(3)))

        feature_map_1 = feature_map_1.view(feature_map_1.size(0), feature_map_1.size(1), -1)
        feature_map_2 = feature_map_2.view(feature_map_2.size(0), feature_map_2.size(1), -1).transpose(1,2)
        return torch.bmm(feature_map_1, feature_map_2) / feature_map_1.size(2)


    """------------------------"""
    """    STAGE 1 TRAINING    """
    """ KNOWLEDGE DISTILLATION """
    """------------------------"""

    ''' Stage 1 distills the knowledge in the teacher model into the student model
        by training the student to match its FSP matrices to the teacher's FSP matrices '''

    # Train the student model to match its FSPs to the teacher's FSPs over one minibatch
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

            losses = []
            hint_feature_maps = self.hint_feature_extractor(features)
            guided_feature_maps = self.guided_feature_extractor(features)

            for count in range(len(self.hint_layers)):
                # The feature maps generated by the teacher model at this hint module for this predicton are extracted
                # and the FSP is calculated between them
                hint_start_layer_feature_map = hint_feature_maps['hint_start_%d' %count]
                hint_end_layer_feature_map = hint_feature_maps['hint_end_%d' %count]
                teacher_FSPs = self.calculate_FSP_matrix(hint_start_layer_feature_map, hint_end_layer_feature_map)

                # The feature maps generated by the student model at this guided module for this predicton are extracted
                # and the FSP is calculated between them
                guided_start_layer_feature_map = guided_feature_maps['guided_start_%d' %count]
                guided_end_layer_feature_map = guided_feature_maps['guided_end_%d' %count]
                student_FSPs = self.calculate_FSP_matrix(guided_start_layer_feature_map, guided_end_layer_feature_map)

                # The loss between the student and teacher FSPs is calculated
                losses.append(loss_fn(student_FSPs, teacher_FSPs))
                
            loss = sum(losses) / len(losses)

            for param in net.parameters():
                param.grad = None
                
            # Perform backprop
            loss.mean().backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_norm=2.0, norm_type=2)
            optimizer.step()

            # Add metrics to the accumulator
            metric.add(float(loss.sum()), utils.accuracy(preds, labels), labels.numel())

        # Return the metrics for this epoch
        return metric[0] / metric[2], metric[1] / metric[2]

    # Train the student model to match its FSPs to the teacher's FSPs over the given number of epochs
    def train_stage_1(self, student_net, teacher_net, train_set, test_set, num_epochs, wandb_log=False): 
        # Use the helper function to attach a forward hook to the hint and guided layers
        # this means that the outputs of these layers will be stored in the dictionary 'feature_map'
        # on every forward pass the network takes
        # the string passed to the helper function defines each feature map's key in the dict
        #for count, hint_pair in enumerate(self.hint_layers):
        #    print("registering hint pair %d" %count)
        #    hint_pair[0].register_forward_hook(self.get_feature_map('hint_start_%d' %count))
        #    hint_pair[1].register_forward_hook(self.get_feature_map('hint_end_%d' %count))
        
        #for count, guided_pair in enumerate(self.guided_layers):
        #    print("registering guided pair %d" %count)
        #    guided_pair[0].register_forward_hook(self.get_feature_map('guided_start_%d' %count))
        #    guided_pair[1].register_forward_hook(self.get_feature_map('guided_end_%d' %count))
        
        hint_dict = {}
        for count, hint_pair in enumerate(self.hint_layers):
            hint_dict[hint_pair[0]] = 'hint_start_%d' %count
            hint_dict[hint_pair[1]] = 'hint_end_%d' %count
        self.hint_feature_extractor = create_feature_extractor(teacher_net, return_nodes=hint_dict)
        
        guided_dict = {}
        for count, guided_pair in enumerate(self.guided_layers):
            guided_dict[guided_pair[0]] = 'guided_start_%d' %count
            guided_dict[guided_pair[1]] = 'guided_end_%d' %count
        self.guided_feature_extractor = create_feature_extractor(student_net, return_nodes=guided_dict)

        # Perform the first stage of model training, using 'train_epoch_stage_1' fn to train the model each epoch
        loss_fn = nn.MSELoss().to(device)
        return utils.train(student_net, self.train_epoch_stage_1, train_set, test_set, loss_fn, num_epochs, self.optimizer, wandb_log, calc_val_accuracy=False)


    """-------------------"""
    """ STAGE 2 TRAINING  """
    """ STANDARD TRAINIING """
    """-------------------"""

    ''' Stage 2 involves standard training of the student model, now that it has recieved a 'good initialisation' from Stage 1 '''

    def train_stage_2(self, student_net, train_set, test_set, num_epochs, wandb_log=False):
        # Perform the second stage of model training, using 'train_epoch' fn to train the student model each epoch
        loss_fn = nn.CrossEntropyLoss(reduction='none').to(device)
        return utils.train(student_net, self.train_epoch, train_set, test_set, loss_fn, num_epochs, self.optimizer, wandb_log)
        
    def train_epoch(self, student, train_set, loss_fn, optimizer):
        return utils.train_epoch(student, train_set, loss_fn, optimizer)


    """-----------------------------"""
    """ INTERFACES FOR DISTILLATION """
    """-----------------------------"""

    ''' These functions are used to begin knowledge distillation '''

    def train(self, train_set, test_set, num_epochs, wandb_log=False):
        self.train_stage_1(self.student, self.teacher, train_set, test_set, num_epochs, wandb_log=False)
        return self.train_stage_2(self.student, train_set, test_set, num_epochs, wandb_log)