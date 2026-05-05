
import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt('./results.csv', delimiter=',', skiprows=0)
data = np.transpose(data)

plt.title("Normalized Emergence Curve Error")
counts, bins = np.histogram(data[1], bins=50)
plt.bar(bins[:-1], counts, width=4.0)
plt.savefig(f"result_cumulative_error.png")
plt.clf()

plt.title("Emergence Curve Onset Error")
counts, bins = np.histogram(data[2], bins=50)
plt.bar(bins[:-1], counts, width=4.0)
plt.savefig(f"results_onset_error.png")
plt.clf()

plt.title("Two Way Minmax Emergence Curve Error")
counts, bins = np.histogram(data[3], bins=50)
plt.bar(bins[:-1], counts, width=0.8)
plt.savefig(f"result_tw_minmax_error.png")
plt.clf()

plt.title("One Way Minmax Emergence Curve Error")
counts, bins = np.histogram(data[4], bins=50)
plt.bar(bins[:-1], counts, width=0.8)
plt.savefig(f"result_ow_minmax_error.png")
plt.clf()
