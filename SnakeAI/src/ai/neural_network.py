import numpy as np
import random
from typing import List, Tuple


class NeuralNetwork:
    def __init__(self, layer_sizes: List[int]):
        self.layer_sizes = layer_sizes
        self.weights = []
        self.biases = []
        
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * 0.5
            b = np.random.randn(layer_sizes[i + 1]) * 0.5
            self.weights.append(w)
            self.biases.append(b)
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        activation = inputs
        
        for i in range(len(self.weights) - 1):
            z = np.dot(activation, self.weights[i]) + self.biases[i]
            activation = self.relu(z)
        
        z = np.dot(activation, self.weights[-1]) + self.biases[-1]
        output = self.softmax(z)
        
        return output
    
    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()
    
    def get_weights_flat(self) -> np.ndarray:
        flat = []
        for w, b in zip(self.weights, self.biases):
            flat.extend(w.flatten())
            flat.extend(b.flatten())
        return np.array(flat)
    
    def set_weights_flat(self, flat_weights: np.ndarray):
        idx = 0
        for i in range(len(self.weights)):
            w_shape = self.weights[i].shape
            w_size = w_shape[0] * w_shape[1]
            self.weights[i] = flat_weights[idx:idx + w_size].reshape(w_shape)
            idx += w_size
            
            b_shape = self.biases[i].shape
            b_size = b_shape[0]
            self.biases[i] = flat_weights[idx:idx + b_size]
            idx += b_size
    
    def copy(self):
        new_nn = NeuralNetwork(self.layer_sizes)
        new_nn.set_weights_flat(self.get_weights_flat().copy())
        return new_nn
    
    def mutate(self, mutation_rate: float = 0.1, mutation_strength: float = 0.5):
        flat = self.get_weights_flat()
        
        for i in range(len(flat)):
            if random.random() < mutation_rate:
                flat[i] += np.random.randn() * mutation_strength
        
        self.set_weights_flat(flat)
    
    @staticmethod
    def crossover(parent1, parent2):
        child = parent1.copy()
        
        flat1 = parent1.get_weights_flat()
        flat2 = parent2.get_weights_flat()
        child_flat = flat1.copy()
        
        crossover_point = random.randint(0, len(flat1))
        child_flat[crossover_point:] = flat2[crossover_point:]
        
        child.set_weights_flat(child_flat)
        return child
