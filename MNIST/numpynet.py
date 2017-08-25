import json
import random
import sys

import numpy as np

class CrossEntropyCost(object):

    @staticmethod
    def fn(a, y):
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a))) \
                        +(self.lmbda/(2*len(training_data)))*sum(
                        [w**2 for w in self.weights])

    @staticmethod
    def delta(z, a, y):
        return (a-y)

class Sigmoid(object):

    @staticmethod
    def fn(x):
        return sigmoid(x)

    @staticmethod
    def prime(x):
        return sigmoid(x)*(1-sigmoid(x))

class Tanh(object):

    @staticmethod
    def fn(x): #I might consider making these static methods because I'll want to call both tanh_f and tanh_f_prime
        return tanh_f(x)

    @staticmethod
    def prime(x):
        return 1 - (tanh_f(x))**2

class Network(object):

    def __init__(self, sizes, activation_function):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.activation_function = activation_function
        self.cost = CrossEntropyCost
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = self.activation_function.fn(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            lmbda = 0.0,
            evaluation_data=None):
        if evaluation_data: n_data = len(evaluation_data)
        n = len(training_data)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(
                    mini_batch, eta, lmbda, len(training_data))
            print "Epoch %s training complete" % j
            self.evaluate_accuracy(evaluation_data)
        return evaluation_cost, evaluation_accuracy, \
            training_cost, training_accuracy

    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [(1-eta*(lmbda/n))*w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = self.activation_function.fn(z)
            activations.append(activation)

        delta = (self.cost).delta(zs[-1], activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = self.activation_function.prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate_accuracy(self, evaluation_data):
        np.random.shuffle(evaluation_data)
        results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in evaluation_data]
        count = 0
        for (x, y) in results:
            if int(x == y): count += 1
        print "Accuracy on evaluation data: {} / {}".format(
            count, len(evaluation_data))
        return

def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def sigmoid(x):
    return np.exp(x)/(1.0+np.exp(x))

def sigmoid_prime(x):
    return sigmoid(x)*(1-sigmoid(x))

def tanh_f(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = Network([784, 30, 10], Sigmoid)
net.SGD(training_data, 30, 40, 0.5, lmbda=100 , evaluation_data=test_data)
