"""
An object-oriented approach to neural networks. Free from any matrix math. It doesn't even use numpy!
Extremely unoptimized. This is a learning exercise only.
"""
import random
import math
import operator


def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


def softmax(values):
    denom = math.fsum(math.exp(i) for i in values)
    return [math.exp(i) / denom for i in values]


def one_hot(index, dimensions):
    return [i == index for i in range(dimensions)]


class Neuron(object):
    def __init__(self, type):
        if type not in ('input', 'hidden', 'output'):
            raise ValueError(f'Invalid Neuron type: {type}')
        self.type = type
        self.inputs = []
        self.outputs = []
        self.bias = 0
        if type == 'hidden':
            self.bias = random.uniform(-1, 1)

    def connect(self, other):
        connection = Connection(self, other)
        self.outputs.append(connection)
        other.inputs.append(connection)
        return connection

    @property
    def z_value(self):
        return math.fsum(c.input.activation * c.weight for c in self.inputs) + self.bias

    @property
    def activation(self):
        if self.type == 'input':
            return self.value
        else:
            return sigmoid(self.z_value)

    def cost(self, y):
        return (1 / 2) * ((y - self.activation) ** 2)

    def cost_prime(self, y):
        return y - self.activation

    def backprop(self, y=None, learning_rate=None):
        if self.type == 'output':
            self._delta = learning_rate * sigmoid_prime(self.z_value) * self.cost_prime(y)
        if self.type == 'hidden':
            self._delta = math.fsum(c.weight * c.output._delta for c in self.outputs) * sigmoid_prime(self.z_value)
            self.bias += self._delta
        for connection in self.inputs:
            connection._weight_delta = self._delta * connection.input.activation
        return self._delta


class Connection(object):
    def __init__(self, input=None, output=None):
        self.input = input
        self.output = output
        self.weight = random.uniform(-1, 1)

    def update_weight(self):
        self.weight += self._weight_delta
        self._weight_delta = None


class Layer(object):
    def __init__(self, size, type):
        self.neurons = [Neuron(type=type) for _ in range(size)]

    def connect(self, other):
        connections = []
        for left in self.neurons:
            for right in other.neurons:
                connection = left.connect(right)
                connections.append(connection)
        return connections

    @property
    def inputs(self):
        return [n.inputs for n in self.neurons]

    @property
    def outputs(self):
        return [n.outputs for n in self.neurons]

    @property
    def biases(self):
        return [n.bias for n in self.neurons]

    @property
    def output(self):
        z_values = [n.z_value for n in self.neurons]
        return softmax(z_values)

    def cost(self, y):
        costs = (n.cost(target) for n, target in zip(self.neurons, y))
        return math.fsum(costs)

    def backprop(self, y_hot=None, learning_rate=None):
        deltas = []
        if y_hot and learning_rate:
            for n, y in zip(self.neurons, y_hot):
                delta = n.backprop(y, learning_rate)
                deltas.append(delta)
        else:
            for n in self.neurons:
                delta = n.backprop()
                deltas.append(delta)
        return deltas


class NN(object):
    def __init__(self, shape, learning_rate=1.0):
        self.input_layer = Layer(shape[0], 'input')
        self.output_layer = Layer(shape[-1], 'output')
        hidden_layers = [Layer(size, 'hidden') for size in shape[1:-1]]
        self.layers = [self.input_layer] + hidden_layers + [self.output_layer]
        self.connections = []
        for left, right in zip(self.layers[:-1], self.layers[1:]):
            layer_connections = left.connect(right)
            self.connections.extend(layer_connections)
        self.learning_rate = learning_rate

    def input(self, x):
        for neuron, value in zip(self.input_layer.neurons, x):
            neuron.value = value

    def feedforward(self, x):
        self.input(x)
        max_index, max_value = max(enumerate(self.output_layer.output), key=operator.itemgetter(1))
        return max_index, max_value

    def backprop(self, x, y):
        y_hot = one_hot(y, len(self.output_layer.neurons))
        self.input(x)
        for layer in reversed(self.layers[1:]):
            if layer == self.output_layer:
                layer.backprop(y_hot, self.learning_rate)
            else:
                layer.backprop()
        self.update_weights()

    def update_weights(self):
        for c in self.connections:
            c.update_weight()

    def fit(self, X, Y):
        for x, y in zip(X, Y):
            self.backprop(x, y)

    def evaluate(self, X, Y):
        recall = 0
        for x, y in zip(X, Y):
            prediction, confidence = self.feedforward(x)
            if prediction == y:
                recall += 1
        accuracy = recall / len(Y)
        return accuracy
