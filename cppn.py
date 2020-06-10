import numpy as np
from activations import sigmoid, sin, cos, tanh, relu, identity

class CPPN:

    activations_list = [sigmoid, sin, cos, tanh, relu]

    def __init__(self, input_size=3, output_size=1, hidden_size=0):
        self.n_in = input_size
        self.n_out = output_size
        self.n_h = hidden_size

        self.n_x = 2
        self.n_y = 2
        self.n_z = 2

        self.age = 0
        self.fitness = 0

        self.dense(self.n_in, self.n_out)

    def get_coordinates(self):
        """ temporary x, y, z data - will be replaced by the coordinates of cells in simulated heart"""

        all_indices = np.indices((self.n_x, self.n_y, self.n_z))
        input_x = all_indices[0]
        input_y = all_indices[1]
        input_z = all_indices[2]

        x_dat = np.reshape(input_x.flatten(), (len(input_x.flatten()),1))
        y_dat = np.reshape(input_y.flatten(), (len(input_y.flatten()), 1))
        z_dat = np.reshape(input_z.flatten(), (len(input_z.flatten()), 1))

        return np.concatenate((x_dat, y_dat, z_dat), axis=1)

    def dense(self, input_size, output_size):
        self.W = np.random.standard_normal((input_size, output_size))
        self.activations = np.random.choice(CPPN.activations_list, output_size)

    def build(self):
        data = self.get_coordinates()
        y = np.dot(data, self.W)

        for i in range(len(self.activations)):
            y[:,i] = self.activations[i](y[:,i])

        return np.reshape(y, (self.n_x, self.n_y, self.n_z))

    #TODO: MUTATION FUNCTIONS (add node, remove node, add edge, remove edge, mutate weights, mutate activation functions)
    def mutate(self):
        pass

    def evaluate(self):
        pass

def main():
    np.random.seed()
    cppn = CPPN()
    output = cppn.build()
    print(output)

if __name__=='__main__':
    main()




