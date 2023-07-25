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
	mask = np.tile(mask, 290)
	mask = np.reshape(mask, 290*60*100*100)
file.close()

mse_per_variable = []
total_sum = 0
for k in range(16):
	curr_act = actual[:,:,k,:,:,:]
	curr_act = np.reshape(curr_act, 290*60*100*100)
	curr_pred = predicted[:,:,k,:,:,:]
	curr_pred = np.reshape(curr_pred, 290*60*100*100)

	#combined = np.concatenate((curr_pred, curr_act))

	#combined = pre.MinMaxScaler().fit_transform(np.reshape(combined, (-1, 1)))

	#curr_pred = combined[:290*60*100*100]
	#curr_act = combined[290*60*100*100:]
	#curr_act = np.reshape(curr_act, 290*60*100*100)
	#curr_pred = np.reshape(curr_pred, 290*60*100*100)

	curr_act[mask == True] = np.nan 
	curr_pred[mask == True] = np.nan 

	diff = np.subtract(curr_act, curr_pred) ** 2
	mse = np.nansum(diff) / (len(curr_act[curr_act != np.nan]))
	mse_per_variable.append(mse)

	total_sum += np.nansum(diff)

total_sum = total_sum / (16*len(curr_act[curr_act != np.nan]))

plt.figure(figsize = (10, 7))
plots = sns.barplot(x = np.arange(0, 16), y = mse_per_variable, color = "lightblue")
for bar in plots.patches:
	plots.annotate(format(bar.get_height(), '.4f'), (bar.get_x() + bar.get_width() / 2, bar.get_height()), ha='center', va='center', size=8, xytext=(0, 8), textcoords='offset points')
plt.xlabel("Variable")
plt.ylabel("Mean Squared Error")
plt.title("MSE for each Variable")

plt.axhline(y = total_sum, color = "blue", linestyle = "dashed", label = "Total MSE")
plt.legend(loc = 'upper right')
plt.savefig("visualizations/msebarplot.png")

