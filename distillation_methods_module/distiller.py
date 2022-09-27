# A base class for knolwedge distillation

# teacher:      the model containing knowledge to be distilled
# student:      the model to distill knowledge into
# optimizer:    an optimizer for training the student network during distillation 
class Distiller():
    def __init__(self, teacher, student, optimizer):
        super().__init__()

        self.teacher = teacher
        self.student = student
        self.optimizer = optimizer

    def train_epoch(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError