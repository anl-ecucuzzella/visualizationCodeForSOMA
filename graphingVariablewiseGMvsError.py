import numpy as np 
import pickle
from sklearn import preprocessing as pre
import matplotlib.pyplot as plt 
import seaborn as sns

with open("../2023-07-25-true_pred.pkl", "rb") as file:
	data = pickle.load(file)
	actual = np.array(data['true'])
	predicted = np.array(data['pred'])
file.close()


with open("SOMA_mask.pkl", "rb") as file:
	maskdata = pickle.load(file)
	mask = maskdata['mask1']
	mask = np.reshape(mask[0,:,:,:,0], 1*60*100*100)
	mask = np.tile(mask, 29)
	mask = np.reshape(mask, 29*60*100*100)
file.close()

errors = []
for i in range(10):
	current_errors = []
	for k in range(16):
		curr_act = np.reshape(actual[i,:,k,:,:,:], 29*60*100*100)
		curr_pred = np.reshape(predicted[i,:,k,:,:,:], 29*60*100*100)

		curr_act[mask == True] = np.nan
		curr_pred[mask == True] = np.nan 

		error = np.nansum(np.subtract(curr_act, curr_pred)**2) / len(curr_act[curr_act != np.nan])
		current_errors.append(error)
	errors.append(current_errors)

gms = [0.395, 1.2466667, 1.6133333, 1.6883334, 1.8916667, 2.0033333, 2.1033332, 3.0666666, 3.1716666, 3.2033334]
gms = ['{0:.{1}f}'.format(p, 2) for p in gms]

variable_names = [str(i) for i in range(16)]

ax = sns.heatmap(errors, cmap = "Reds", yticklabels = gms, annot = False, xticklabels = variable_names, cbar = True)
ax.set(xlabel = "GM", ylabel = "Variable", title = "MSE for GM Values per Variable")
plt.savefig("visualizations/heatmapOfGMvsVariableError.png")

