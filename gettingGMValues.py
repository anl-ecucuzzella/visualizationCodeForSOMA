import numpy as np
import pickle

with open("../2023-07-20-true_pred.pkl", "rb") as file:
	data = pickle.load(file)
	gm = np.array(data['gm'])
file.close()
gm = gm[:,0,16,0,50,50]
gm = np.reshape(gm, 10)
gm = np.unique(gm)
print(len(gm))
print(gm)
