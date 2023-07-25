import numpy as np 
import h5py
import matplotlib.pyplot as plt
import pickle
from sklearn import preprocessing as pre 

with open("../2023-07-25-true_pred.pkl", "rb") as file:
	data = pickle.load(file)
	actual = data['true'][0]
	predicted = data['pred'][0]
file.close()

with open("SOMA_mask.pkl", "rb") as file:
	maskdata = pickle.load(file)
	mask = maskdata['mask1']
	mask = np.reshape(mask[0, 0, :, :, 1], (100, 100))
file.close()

plt.figure(figsize = (20, 20))
fig, axs = plt.subplots(4, 3)
starting = 12
for k in range(4):
	curr_act = actual[0, k+starting, 0, :, :]
	curr_act = np.reshape(curr_act, (100, 100))
	curr_pred = predicted[0, k+starting, 0, :, :]
	curr_pred = np.reshape(curr_pred, (100, 100))

	curr_pred[mask == True] = np.nan

	curr_act[mask == True] = np.nan

	temp_act = np.reshape(curr_act, 100*100)
	temp_pred = np.reshape(curr_pred, 100*100)
	maximum = np.nanmax(np.concatenate((temp_act, temp_pred)))
	minimum = np.nanmin(np.concatenate((temp_act, temp_pred)))

	diff = np.subtract(curr_act, curr_pred) ** 2

	im1 = axs[k, 0].imshow(curr_act, cmap = "Blues", interpolation = 'none', vmin = minimum, vmax = maximum)
	if k == 0:
		axs[k, 0].set_title("Actual")
	axs[k, 0].set_ylabel("Variable " + str(k+starting), rotation = 90)
	#cbar3 = axs[k, 0].figure.colorbar(im1, ax = axs[k, 0])
	axs[k, 0].tick_params(left = False, right = False, labelleft = False, labelbottom = False, bottom = False)

	im2 = axs[k, 1].imshow(curr_pred, cmap = "Blues", interpolation = 'none', vmin = minimum, vmax = maximum)
	if k == 0:
		axs[k, 1].set_title("Predicted")
	cbar2 = axs[k, 1].figure.colorbar(im2, ax = axs[k, 1])
	axs[k, 1].tick_params(left = False, right = False, labelleft = False, labelbottom = False, bottom = False)

	im3 = axs[k, 2].imshow(diff, cmap = "Reds", interpolation = 'none')# vmin = 0.0, vmax = 1.0)
	if k == 0:
		axs[k, 2].set_title("Squared Error")
	cbar = axs[k, 2].figure.colorbar(im3, ax = axs[k, 2])
	axs[k, 2].tick_params(left = False, right = False, labelleft = False, labelbottom = False, bottom = False)

for ax in axs.flat:
	ax.label_outer()

plt.savefig("visualizations/actualpredictedsquarederror12through16.png")
