"""loading and saving input data as numpy arrays"""
import numpy as np
import pickle

# Calculated12Lead for testing
f = open('calculated12Lead.txt', 'rb')
lead12_list = f.readlines()
f.close()

calculated12lead = np.zeros((len(lead12_list), 12), dtype=float)

for i, line in enumerate(lead12_list):
    line_as_list = line.split(',')
    for j in range(len(line_as_list)):
        calculated12lead[i, j] = line_as_list[j]

f = open('calculated12Lead.p', 'wb')
pickle.dump(calculated12lead, f)
f.close()

# Lead distances

f = open('leadDistances.txt', 'rb')
lead_dist_list = f.readlines()
f.close()

lead_distances = np.zeros((len(lead_dist_list), 10), dtype=float)

for i,line in enumerate(lead_dist_list):
    line_as_list = line.split(',')
    for j in range(len(line_as_list)):
        lead_distances[i,j] = line_as_list[j]

f = open('lead_distances.p', 'wb')
pickle.dump(lead_distances, f)
f.close()

# Lead coordinates

f = open('leadCoords.txt', 'rb')
lead_coord_list = f.readlines()
f.close()

lead_coords = np.zeros((len(lead_coord_list),3), dtype=int)

for i,line in enumerate(lead_coord_list):
    coord_list = line.split(',')
    lead_coords[i,0] = int(coord_list[0]) # x
    lead_coords[i, 1] = int(coord_list[1]) # y
    lead_coords[i, 2] = int(coord_list[2]) # z

f = open('lead_coords.p', 'wb')
pickle.dump(lead_coords, f)
f.close()

# Heart coordinates

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


