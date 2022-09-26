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