import numpy as np
from commen import*


class SoftmaxLayer:
    def __init__(self, nodes):
        self.nodes = nodes

    def forward(self, input):
        input = input.flatten()
        input_len = input.size()
        weights = np.random.randn(input_len, self.nodes) / input_len
        biases = np.zeros(self.nodes)
        totals = np.dot(input, weights) + biases
        exp = np.exp(totals)
        return exp/np.sum(exp, axis=0)


class HideLayer:
    def __init__(self, nodes):
        self.nodes = nodes

    def forward(self, input):
        input = input.flatten()
        input_len = input.size()
        weights = np.random.randn(input_len, self.nodes) / input_len
        biases = np.zeros(self.nodes)
        totals = np.dot(input, weights) + biases
        return relu(totals)


