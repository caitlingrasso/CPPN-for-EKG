import numpy as np
from activations import sigmoid, sin, cos, tanh, relu, identity, normalize

class CPPN:

    activations_list = [sigmoid, sin, cos, tanh, relu]

    def __init__(self, inputs=3, outputs=1, fixed=False, n_nodes=15, n_hl=3, n_h=6):
        self.n_in = inputs
        self.n_out = outputs

        if fixed:
            self.model = self.initialize_fixed(n_hl, n_h)
        else:
            self.model = self.initialize(n_nodes)
        self.edge_mask = self.set_mask()

    def initialize(self, n_nodes):
        """generates a random CPPN topology"""
        #n_nodes = total number of hidden nodes

        layers = []
        input_dim = self.n_in

        while n_nodes > 1:
            splt = np.random.choice(range(1,n_nodes))
            layers.append(Dense(input_dim, splt))
            n_nodes = n_nodes - splt
            input_dim = splt

        layers.append(Dense(layers[-1].W.shape[1], self.n_out, activation=sigmoid))

        return layers

    def initialize_fixed(self, n_hl, n_h):
        """generates random CPPN with fixed topology"""
        #n_hl = number of hidden layers
        #n_h = number of hidden nodes per layer

        layers = []
        layers.append(Dense(self.n_in, n_h)) # input layer
        for i in range(n_hl): # hidden layers
            layers.append(Dense(n_h, n_h))
        layers.append(Dense(n_h, self.n_out, activation=sigmoid)) # output layer
        return layers

    def set_mask(self):
        """sets binary connections mask for CPPN"""

        connections = []
        connections.append(np.ones(self.model[0].W.shape)) # fully-connected input layer
        for i in range(1, len(self.model)-1):
            connections.append(np.random.randint(0,2, size=self.model[i].W.shape))
        connections.append(np.ones(self.model[-1].W.shape)) # fully-connected output layer
        return connections

    def get_coordinates(self):
        """temporary x, y, z data - will be replaced by the coordinates of cells in simulated heart"""
        self.n_x = 10
        self.n_y = 10
        self.n_z = 10

        all_indices = np.indices((self.n_x, self.n_y, self.n_z))
        input_x = all_indices[0]
        input_y = all_indices[1]
        input_z = all_indices[2]

        x_dat = np.reshape(input_x.flatten(), (len(input_x.flatten()),1))
        y_dat = np.reshape(input_y.flatten(), (len(input_y.flatten()), 1))
        z_dat = np.reshape(input_z.flatten(), (len(input_z.flatten()), 1))

        x_dat = normalize(x_dat)
        y_dat = normalize(y_dat)
        z_dat = normalize(z_dat)

        return np.concatenate((x_dat, y_dat, z_dat), axis=1)

    def get_output(self):
        """executes forward pass of cppn"""

        x = self.get_coordinates()
        for i, layer in enumerate(self.model):
            x = layer.build(x, self.edge_mask[i])
        return np.reshape(x, (self.n_x, self.n_y, self.n_z))

    def mutate(self):
        """mutation types: add edge, remove edge, mutate weights, mutate activation function"""
        mut_type = np.random.choice(4)
        print(mut_type)
        #TODO: finish mutation functions

    def evaluate(self):
        result = self.get_output()
        print(result)

        # TODO: feed cppn output as input into EKG functions to get a fitness score

        # return fitness

    def print(self):
        print("CPPN SUMMARY:")
        for i, layer in enumerate(self.model):
            print("--LAYER", i, "--")
            print("input_dim:", layer.W.shape[0], ", output_dim:", layer.W.shape[1], ", activation(s):", end=" ")
            for j, a in enumerate(layer.activations):
                if j==len(layer.activations)-1:
                    print(a.__name__)
                else:
                    print(a.__name__, end=", ")

class Dense:

    def __init__(self, input_size, output_size, activation=None):
        """initializes weights/activations for single fully-connected layer of network"""

        self.W = np.random.standard_normal((input_size, output_size))
        if activation is not None:
            self.activations = [activation]
        else:
            self.activations = np.random.choice(CPPN.activations_list, output_size)

    def build(self, input, edge_mask):
        """builds single layer"""

        weights = self.W*edge_mask

        y = np.dot(input, weights)
        for i in range(len(self.activations)):
            y[:, i] = self.activations[i](y[:, i])
        return y





