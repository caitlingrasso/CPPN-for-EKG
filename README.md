# CPPN-for-EKG

Evolving CPPNs with AFPO to paint activation sequences on simulated heart cells. 

**CPPN.py, activations.py**  - contains code to create, execute, and mutate a CPPN.

**AFPO.py, genome.py** - contains code to run AFPO.
* To run an instance of AFPO use `python afpo.py --gens=10 --popsize=10`

**evaluate_EKG.py** - functions used to evaluate the output of CPPNs. Converts activation sequence to an EKG signal and compares to the 'truth' EKG. 

**util.py** - helper functions to load and write data, plot EKGs, normalize and smooth data. Before you start...
1. Call `load_txt_file(filename)` for each data file (lead_distances, heart_coords, answer_EKG) to convert the data to a numpy array that is saved in a pickle file.
This can be done in `scratch.py` (see example).
2. At the top of `CPPN.py` specify the pickle file names for each of the data files.  

**scratch.py** - (temp) examples to run/test different parts of the code. 