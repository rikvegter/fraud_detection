import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


n_estimators = np.linspace(50, 400, 8)
#ee, auroc, auprc
ee = [0.9546, 0.9522, 0.9522, 0.9521, 0.9519, 0.9519, 0.9516, 0.9506]
auroc = [0.9951, 0.996, 0.9964, 0.9964, 0.9972, 0.9971, 0.9975, 0.9973]
auprc = [0.9443, 0.9484, 0.9482, 0.9488, 0.9489, 0.9486, 0.9488, 0.9489]


# plot lines
plt.plot(n_estimators, ee, label = "EE")
plt.plot(n_estimators, auroc, label = "auroc")
plt.plot(n_estimators, auprc, label = "auprc")
plt.title('EE, AUROC, AUPRC of Ghost with RF for a various number of trees')
plt.xlabel('Number of trees')
plt.ylabel('Performance')
plt.legend()
plt.show()
