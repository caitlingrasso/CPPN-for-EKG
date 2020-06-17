import numpy as np
import random
import copy
from activations import sigmoid, sin, cos, tanh, abs, normalize

class CPPN:

    activations_list = [sigmoid, sin, cos, tanh, abs]

    def __init__(self, inputs=3, outputs=1, novel=False, n_nodes=15):
        self.n_in = inputs
        self.n_out = outputs

        # model = list of layers
        if novel:
            self.model = self.novel_individual(n_nodes=n_nodes)
            self.edge_mask = self.set_mask()
        else:
            self.model = [Dense(self.n_in, self.n_out, activation=sigmoid)]
            self.edge_mask = [np.ones((self.n_in, self.n_out))]

    def novel_individual(self, n_nodes):
        """generates a random CPPN topology"""
        #n_nodes = total number of hidden nodes

        layers = []
        input_dim = self.n_in

        while n_nodes > 1:
            splt = np.random.choice(range(1,n_nodes))
            layers.append(Dense(input_dim, splt))
            n_nodes = n_nodes - splt
            input_dim = splt
            if input_dim==1 and n_nodes==2:
                n_nodes=0

        layers.append(Dense(layers[-1].W.shape[1], self.n_out, activation=sigmoid))

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
        self.n_x = 2
        self.n_y = 2
        self.n_z = 2

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
        """chooses a random mutation to execute"""

        mut_type = np.random.choice(6)
        if mut_type == 0:
            type = "mutate weight"
            done = self.mutate_weight()
        elif mut_type == 1:
            type = "mutate activation"
            done = self.mutate_activations()
        elif mut_type == 2:
            type = "add edge"
            done = self.add_edge()
        elif mut_type == 3:
            type = "remove edge"
            done = self.remove_edge()
        elif mut_type == 4:
            type = "add node"
            done = self.add_node()
        elif mut_type == 5:
            type = "remove node"
            done = self.remove_node()
        print(type, done)
        return done # true if a mutation was performed, false otherwise

    # -----------------------------------
    # MUTATION FUNCTIONS
    # -----------------------------------

    def mutate_weight(self, mut_std = 0.5):
        # choose a layer
        i = np.random.choice(len(self.model))
        l = self.model[i]

        # mutate random weight in chosen layer
        r = np.random.randint(l.W.shape[0])
        c = np.random.randint(l.W.shape[1])
        old_weight = l.W[r,c]
        l.W[r,c] = random.gauss(old_weight, mut_std)
        return True

    def mutate_activations(self):
        if len(self.model) == 1: # no hidden layers, don't change activation of output neuron
            return False
        i = np.random.choice(len(self.model) - 1)
        l = self.model[i]
        rand_index = np.random.randint(len(l.activations))
        new_act = np.random.choice(CPPN.activations_list)
        l.activations[rand_index] = new_act
        return True

    def add_edge(self):

        fully_connected = True
        for m in self.edge_mask:
            if np.all(m==1)==False:
                fully_connected=False
        if fully_connected:
            return False # no edges to add if the network is fully-connected

        i = np.random.choice(len(self.model))
        m = self.edge_mask[i]

        # choose a not fully-connected layer
        while np.all(m==1):
            i = np.random.choice(len(self.model))
            m = self.edge_mask[i]

        r = np.random.randint(m.shape[0])
        c = np.random.randint(m.shape[1])
        while m[r,c] == 1:
            r = np.random.randint(m.shape[0])
            c = np.random.randint(m.shape[1])
        m[r,c]=1
        return True

    def remove_edge(self):

        no_edges = True
        for m in self.edge_mask:
            if np.any(m==1):
                no_edges = False
        if no_edges:
            return False # no edges to be removed

        i = np.random.choice(len(self.model))
        m = self.edge_mask[i]

        # choose a layer with at least one connection
        while np.all(m==0):
            i = np.random.choice(len(self.model))
            m = self.edge_mask[i]

        r = np.random.randint(m.shape[0])
        c = np.random.randint(m.shape[1])

        # choose a edge that is set to 1
        while m[r, c] == 0:
            r = np.random.randint(m.shape[0])
            c = np.random.randint(m.shape[1])
        m[r, c] = 0

        return True

    def add_node(self):
        if len(self.model) == 1:  # no hidden layers
            return False

        # choose a hidden layer to add a node to
        i = np.random.choice(len(self.model) - 1)
        l = self.model[i]

        weights = copy.deepcopy(l.W)
        mask = copy.deepcopy(self.edge_mask[i])
        activations = copy.deepcopy(l.activations)
        nodes_in = weights.shape[0]
        nodes_out = weights.shape[1] + 1

        new_weights = np.zeros((nodes_in, nodes_out))
        new_weights[:,0:weights.shape[1]] = weights
        new_node_weights = np.random.standard_normal((nodes_in, 1))
        new_weights[:,nodes_out-1:nodes_out] = new_node_weights

        new_mask = np.zeros((nodes_in, nodes_out), dtype=int)
        new_mask[:,0:mask.shape[1]] = mask
        new_node_mask = np.random.randint(0,2, size=(nodes_in, 1))
        new_mask[:,nodes_out-1:nodes_out] = new_node_mask

        new_activations = [None] * nodes_out
        new_activations[nodes_out-1] = np.random.choice(CPPN.activations_list)
        new_activations[0:nodes_out-1] = activations

        self.model[i].W = new_weights
        self.edge_mask[i] = new_mask
        self.model[i].activations = new_activations

        # connect new node to the next layer
        next_layer = self.model[i+1]
        weights = copy.deepcopy(next_layer.W)
        mask = copy.deepcopy(self.edge_mask[i+1])
        nodes_in = weights.shape[0] + 1
        nodes_out = weights.shape[1]

        new_weights = np.zeros((nodes_in, nodes_out))
        new_weights[0:weights.shape[0], :] = weights
        new_weights[nodes_in-1, :] = np.random.standard_normal((1, nodes_out))

        new_mask = np.zeros((nodes_in, nodes_out), dtype=int)
        new_mask[0:weights.shape[0], :] = mask
        new_node_mask = np.random.randint(0, 2, size=(1, nodes_out))
        new_mask[nodes_in-1, :] = new_node_mask

        self.model[i+1].W = new_weights
        self.edge_mask[i+1] = new_mask

        return True

    def remove_node(self):
        if len(self.model) == 1:  # no hidden layers
            return False

        # choose a hidden layer to remove a node from
        i = np.random.choice(len(self.model) - 1)
        l = self.model[i]

        if l.W.shape[1] == 1: # hidden layer has one node - remove layer?
            return False

        weights = copy.deepcopy(l.W)
        mask = copy.deepcopy(self.edge_mask[i])
        activations = copy.deepcopy(l.activations)

        node_to_remove = np.random.randint(weights.shape[1]) # an output neuron

        slice1 = weights[:, 0:node_to_remove]
        slice2 = weights[:, node_to_remove+1:]
        new_weights = np.concatenate((slice1, slice2), axis=1)

        slice1 = mask[:, 0:node_to_remove]
        slice2 = mask[:, node_to_remove + 1:]
        new_mask = np.concatenate((slice1, slice2), axis=1)

        slice1 = activations[0:node_to_remove]
        slice2 = activations[node_to_remove + 1:]
        new_activations = np.concatenate((slice1, slice2))

        self.model[i].W = new_weights
        self.edge_mask[i] = new_mask
        self.model[i].activations = new_activations

        # remove the node as input to the next layer
        l = self.model[i+1]

        weights = copy.deepcopy(l.W)
        mask = copy.deepcopy(self.edge_mask[i+1])

        slice1 = weights[0:node_to_remove, :]
        slice2 = weights[node_to_remove + 1:, :]
        new_weights = np.concatenate((slice1, slice2), axis=0)

        slice1 = mask[0:node_to_remove, :]
        slice2 = mask[node_to_remove + 1:, :]
        new_mask = np.concatenate((slice1, slice2), axis=0)

        self.model[i+1].W = new_weights
        self.edge_mask[i+1] = new_mask

        return True

    def evaluate(self):
        result = self.get_output()
        print(result)

        # TODO: feed cppn output as input to EKG functions - return fitness score

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





