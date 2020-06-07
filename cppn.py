import numpy as np
from activations import sigmoid

class CPPN:

    activations = [sigmoid] # list of activation functions (sin, cos, tanh, relu)

    def __init__(self):
        self.input_nodes = ['x', 'y', 'z']
        self.output_nodes = ['t']
        self.nodes = []
        self.links = []

        self.initialize()

    def initialize(self):
        self.dense()
        # self.mutate()

    def dense(self):
        """minimal graph - all inputs connected to all outputs (dense)"""

        # creating nodes
        for in_name in self.input_nodes:
            self.nodes.append(Node(name=in_name, type="input", state=None, activation=None)) # input nodes have no activation
        for out_name in self.output_nodes:
            self.nodes.append(Node(name=out_name, type="output", state=None, activation=sigmoid)) # all output nodes have sigmoid activation

        # creating connections
        for input_node in self.nodes:
            if input_node.type == "input":
                for output_node in self.nodes:
                    if output_node.type == "output":
                        self.links.append(Connection(input_node, output_node, weight=0))  # initializing all weights to 0 for now...

    def set_input_node_states(self, grid_size_xyz=(10,10,10)):
        """ temporary - x, y, z data will be replaced by the coordinates of cells in the simulated heart"""

        all_indices = np.indices(grid_size_xyz)
        input_x = all_indices[0]
        input_y = all_indices[1]
        input_z = all_indices[2]

        # x, y, and z vectors should be the length of the number of points in a grid

    # TODO: MUTATION FUNCTIONS (add node, remove node, add edge, remove edge, mutate weights, mutate activation functions)

class Node:
    def __init__(self, name, type, state, activation):
        self.name = name
        self.type = type  # input, output, hidden
        self.state = state  # holds current state (values) at neuron
        self.activation = activation  # activation function

class Connection:
    def __init__(self, node_in, node_out, weight):
        self.node_in = node_in
        self.node_out = node_out
        self.weight = weight

def main():
    cppn = CPPN()
    cppn.create_grid()

if __name__=='__main__':
    main()

