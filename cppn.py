import sys
import numpy as np
from activations import sigmoid, sin, cos, tanh, relu

class CPPN:

    activations = [sigmoid, sin, cos, tanh, relu] # list of activation functions

    def __init__(self):
        self.input_nodes = ['x', 'y', 'z']
        self.output_nodes = ['t']
        self.nodes = []
        self.edges = []

        self.fitness = 0
        self.age = 0

        self.initialize()


    def initialize(self):
        self.dense()
        self.mutate()

    def dense(self):
        """minimal graph - all inputs connected to all outputs (dense)"""

        # creating nodes
        for in_name in self.input_nodes:
            self.nodes.append(Node(id=in_name, layer=0, type="input", activation=None)) # input nodes have no activation
        for out_name in self.output_nodes:
            self.nodes.append(Node(id=out_name, layer=sys.maxsize, type="output", activation=sigmoid)) # all output nodes have sigmoid activation

        # creating connections
        for input_node in self.nodes:
            if input_node.type == "input":
                for output_node in self.nodes:
                    if output_node.type == "output":
                        edge = Connection(input_node, output_node, weight=np.random.random()*2-1)  # initialized to random number in [-1,1)
                        self.edges.append(edge)
                        output_node.edges_in.append(edge)

    def set_input_node_states(self):
        """ temporary - x, y, z data will be replaced by the coordinates of cells in the simulated heart"""

        grid_size_xyz = (2, 2, 2)

        all_indices = np.indices(grid_size_xyz)
        input_x = all_indices[0]
        input_y = all_indices[1]
        input_z = all_indices[2]

        for input_node in self.nodes:
            if input_node.id == "x":
                input_node.state = input_x
                input_node.evaluated = True
            elif input_node.id == "y":
                input_node.state = input_y
                input_node.evaluated = True
            elif input_node.id == "z":
                input_node.state = input_z
                input_node.evaluated = True

        # x, y, and z vectors should be the length of the number of points in a grid

    def get_output(self):
        """runs pass through CPPN to get output -- calculates state at each node"""

        # clear all states
        for node in self.nodes:
            node.state = 0
            node.evaluated = False

        # reset input node states
        self.set_input_node_states()

        for i in range(len(self.nodes)):
            if self.nodes[i].type == "output":
                self.calculate_node_state(i)

        sorted_nodes = sorted(self.nodes, key=lambda x: x.layer)
        return sorted_nodes[-1].state

    def calculate_node_state(self, i):
        if self.nodes[i].evaluated==True:
            return self.nodes[i].state

        self.nodes[i].evaluated=True

        state = 0
        for edge in self.nodes[i].edges_in:
            state += self.calculate_node_state(self.nodes.index(edge.node_in)) * edge.weight

        self.nodes[i].state = self.nodes[i].activation(state)

    # TODO: MUTATION FUNCTIONS (add node, remove node, add edge, remove edge, mutate weights, mutate activation functions)
    def mutate(self):
        pass

    def evaluate(self):
        pass

class Node:
    def __init__(self, id, layer, type, activation):
        self.id = id
        self.layer = layer
        self.type = type  # input, output, hidden
        self.activation = activation  # activation function
        self.state = 0  # holds current state (values) at neuron
        self.evaluated = False
        self.edges_in = []

    def print(self):
        print("---NODE---")
        print("ID:", self.id)
        print("Layer:", self.layer)
        print("Type:", self.type)
        print("Activation: ", self.activation)
        print("State: ", self.state)

class Connection:
    def __init__(self, node_in, node_out, weight):
        self.node_in = node_in
        self.node_out = node_out
        self.weight = weight

    def print(self):
        print("---CONNECTION---")
        print("Node In:", self.node_in.type, self.node_in.id)
        print("Node Out:", self.node_out.type, self.node_out.id)
        print("Weight: ", self.weight)

def main():
    cppn = CPPN()
    # for edge in cppn.edges:
    #     edge.print()
    output = cppn.get_output()
    print(output)

if __name__=='__main__':
    main()

