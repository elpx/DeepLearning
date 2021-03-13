class Perceptron(object):
    def __init__(self, input_num, activator) -> None:
        """
        activator -> f
        weights   -> w
        bias      -> b
        """
        self.activator = activator
        self.weights = [0.0 for _ in range(input_num)]
        self.bias = 0.0

    def __str__(self) -> str:
        return 
    
    