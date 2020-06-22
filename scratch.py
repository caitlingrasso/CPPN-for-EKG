import numpy as np
import pickle

# Converting lead coordinates to a numpy array and saving as a pickle file
f = open('activationMatrix.txt', 'rb')
activation_list = f.readlines()
f.close()

activations = np.zeros((len(activation_list), 1), dtype=int)

for i,line in enumerate(activation_list):
    activations[i] = int(line)

f = open('activation_matrix.p', 'wb')
pickle.dump(activations, f)
f.close()