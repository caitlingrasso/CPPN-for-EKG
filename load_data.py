import numpy as np
import pickle

# Converting lead coordinates to a numpy array and saving as a pickle file
f = open('leadCoords.txt', 'rb')
lead_list = f.readlines()
f.close()

lead_coords = np.zeros((len(lead_list),3), dtype=int)

for i,line in enumerate(lead_list):
    coord_list = line.split(',')
    lead_coords[i,0] = int(coord_list[0]) # x
    lead_coords[i, 1] = int(coord_list[1]) # y
    lead_coords[i, 2] = int(coord_list[2]) # z

f = open('lead_coords.p', 'wb')
pickle.dump(lead_coords, f)
f.close()

# exit()

# Converting heart coordinates to a numpy array and saving as a pickle file

f = open('HeartCoords.txt', 'rb')
cell_list = f.readlines()
f.close()

heart_coords = np.zeros((len(cell_list), 3), dtype=float)

for i, line in enumerate(cell_list):
    coord_list = line.split(',')
    heart_coords[i, 0] = float(coord_list[0]) # x
    heart_coords[i, 1] = float(coord_list[1]) # y
    heart_coords[i, 2] = float(coord_list[2]) # z

f = open('heart_coords.p', 'wb')
pickle.dump(heart_coords, f)
f.close()


