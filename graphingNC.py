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

nc_per_variable = []
total_sum = 0
for k in range(16):
    curr_act = actual[:,:,k,:,:,:]
    curr_act = np.reshape(curr_act, 290*60*100*100)
    curr_pred = predicted[:,:,k,:,:,:]
    curr_pred = np.reshape(curr_pred, 290*60*100*100)

    curr_act[mask == True] = np.nan
    curr_pred[mask == True] = np.nan

    real_length = len(curr_act[curr_act != np.nan])

    act_mean = np.nansum(curr_act) / real_length
    pred_mean = np.nansum(curr_pred) / real_length
    act_stdev = np.nanstd(curr_act)
    pred_stdev = np.nanstd(curr_pred)

    inside_sum = (curr_act - act_mean)*(curr_pred - pred_mean)/(act_stdev * pred_stdev)
    nc = np.nansum(inside_sum) / (real_length - 1)
    nc_per_variable.append(nc)

#total_nc = total_nc / (16*len(curr_act[curr_act != np.nan]))

plt.figure(figsize = (10, 7))
plots = sns.barplot(x = np.arange(0, 16), y = nc_per_variable, color = "lightblue")
for bar in plots.patches:
    plots.annotate(format(bar.get_height(), '.4f'), (bar.get_x() + bar.get_width() / 2, bar.get_height()), ha='center', va='center', size=8, xytext=(0, 8), textcoords='offset points')
plt.xlabel("Variable")
plt.ylabel("NC")
plt.title("NC for each Variable")

#plt.axhline(y = total_sum, color = "blue", linestyle = "dashed", label = "Total MSE")
#plt.legend(loc = 'upper right')
plt.savefig("visualizations/ncbarplot.png")
