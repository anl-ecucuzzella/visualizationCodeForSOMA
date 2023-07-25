import numpy as np 
import pickle 
from scipy.stats import f_oneway

with open("../2023-07-20-true_pred.pkl", "rb") as file:
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

all_predictions = []
for g in range(10):
	curr_pred = predicted[g, :, 12, :, :, :]
	curr_pred = np.reshape(curr_pred, 29*60*100*100)

	curr_pred[mask == True] = np.nan

	curr_pred_new = curr_pred[np.logical_not(np.isnan(curr_pred))]
	print(curr_pred_new)

	all_predictions.append(curr_pred_new)
	print(str(g))

r = f_oneway(all_predictions[0], all_predictions[1], all_predictions[2], all_predictions[3], all_predictions[4], all_predictions[5], all_predictions[6], all_predictions[7], all_predictions[8], all_predictions[9])
print(r)
