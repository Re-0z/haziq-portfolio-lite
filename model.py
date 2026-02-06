import math

class Neuron:
    def __init__(self, weights):
        self.weights = weights

    def forward(self, inputs):
        output = 0
        for i in range(len(inputs)):
            output += inputs [i] * self.weights[i]
        activation = 1 / (1 + math.exp(-output))
        return activation
    
    def train(self, inputs, error):
        learning_rate = 0.01
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * error * inputs[i]

class Neural_Network:
    def __init__(self):
        self.h1 = Neuron([0.1, 0.2, 0.3])
        self.h2 = Neuron([0.4, 0.5, 0.6])
        self.o1 = Neuron([0.7, 0.8, 0.9])
    
    def inspect(self, inputs):
        out_h1 = self.h1.forward(inputs)
        out_h2 = self.h2.forward(inputs)
        out_o1 = self.o1.forward([out_h1, out_h2])
        return {
            "output": out_o1,
            "hidden_layer": [out_h1, out_h2],
            "input_used": inputs
        }
    
    def train(self, inputs, target):
        learning_rate = 0.5

        out_h1 = self.h1.forward(inputs)
        out_h2 = self.h2.forward(inputs)
        out_o1 = self.o1.forward([out_h1, out_h2])
        
        error_o1 = out_o1 - target
        delta_o1 = error_o1 * (out_o1 * (1 - out_o1))

        error_h1 = delta_o1 * self.o1.weights[0]
        delta_h1 = error_h1 * (out_h1 * (1 - out_h1))
        error_h2 = delta_o1 * self.o1.weights[1]
        delta_h2 = error_h2 * (out_h2 * (1 - out_h2))

        self.o1.weights[0] -= learning_rate * delta_o1 * out_h1
        self.o1.weights[1] -= learning_rate * delta_o1 * out_h2

        for i in range(len(self.h1.weights)):
            self.h1.weights[i] -= learning_rate * delta_h1 * inputs [i]
            self.h2.weights[i] -= learning_rate * delta_h2 * inputs [i]

def init_brain():
    net = Neural_Network()
    dataset = [[1.5, 2.3, 3.1], [4.1, 0.1, 6.1], [7.2, 8.3, 1.4]]
    targets = [0, 1, 0]

    print("Training AI Model in background...")
    for epoch in range(5000):
        for i in range(len(dataset)):
            net.train(dataset[i], targets[i])
    
    print("AI Training Complete. System Ready.")
    return net