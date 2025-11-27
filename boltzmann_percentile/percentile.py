import numpy as np

def charge_norm(qs,delta_q):
    new_q = []
    add = delta_q/len(qs)
    for i in qs:
        new_q.append(i-add)
    return new_q


all_chgs = np.load(f'all_charges_mobley_36119.npy')
weights = np.load(f'all_weights_mobley_36119.npy')
weights=weights/np.sum(weights) #normalization
perc_estimate = []
for i in range(len(all_chgs[0])):
    chgs =all_chgs[:,i]
    sorted_indices = np.argsort(abs(chgs)) # Sorting by absolute value
    sorted_q = chgs[sorted_indices]
    sorted_w = weights[sorted_indices] # Sorting the weights accordingly
    cumulative_w = np.cumsum(sorted_w) # Computing cumulative sum 
    x = 0.90
    q_x = sorted_q[np.searchsorted(cumulative_w,x)] # Select the 90-th percentile
    perc_estimate.append(q_x)

perc_estimate=np.array(perc_estimate)
delta = sum(perc_estimate)
new_chgs = charge_norm(perc_estimate,delta)
new_chgs = np.array(new_chgs)
np.save(f'percentile_mobley_36119',new_chgs)