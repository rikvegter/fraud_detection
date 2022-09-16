ghost_eu_1 = ['ghost', 'eu', 0.997, 0.999, 0.992]
ghost_eu_2 = ['ghost', 'eu', 0.994, 0.999, 0.988]
ghost_eu_3 = ['ghost', 'eu', 0.988, 0.999, 0.985]
ghost_eu_4 = ['ghost', 'eu', 1.00, 0.999, 0.981]
ghost_eu_5 = ['ghost', 'eu', 0.780, 0.986, 0.798]

cat_eu_1 = ['cat', 'eu', 0.981, 0.999, 0.994]
cat_eu_2 = ['cat', 'eu', 0.989, 0.999, 0.989]
cat_eu_3 = ['cat', 'eu', 0.986, 0.999, 0.981]
cat_eu_4 = ['cat', 'eu', 0.989, 0.999, 0.986]
cat_eu_5 = ['cat', 'eu', 0.966, 0.972, 0.766]

oxgb_eu_1 = ['oxgb', 'eu', 1.00, 1.00, 1.00]
oxgb_eu_2 = ['oxgb', 'eu', 1.00, 1.00, 1.00]
oxgb_eu_3 = ['oxgb', 'eu', 1.00, 1.00, 1.00]
oxgb_eu_4 = ['oxgb', 'eu', 1.00, 1.00, 1.00]
oxgb_eu_5 = ['oxgb', 'eu', 0.785, 0.982, 0.805]

ghostoxgb_eu_1 = ['ghostoxgb', 'eu', 1.00, 1.00, 1.00]
ghostoxgb_eu_2 = ['ghostoxgb', 'eu', 1.00, 1.00, 1.00]
ghostoxgb_eu_3 = ['ghostoxgb', 'eu', 1.00, 1.00, 1.00]
ghostoxgb_eu_4 = ['ghostoxgb', 'eu', 1.00, 1.00, 1.00]
ghostoxgb_eu_5 = ['ghostoxgb', 'eu', 0.786, 0.982, 0.807]


#----------Pay Sim----------#
ghost_ps_1 = ['ghost', 'ps', 0.938, 0.999, 0.983]
ghost_ps_2 = ['ghost', 'ps', 0.991, 0.999, 0.989]
ghost_ps_3 = ['ghost', 'ps', 0.972, 0.999, 0.986]
ghost_ps_4 = ['ghost', 'ps', 0.992, 0.999, 0.990]
ghost_ps_5 = ['ghost', 'ps', 0.979, 0.999, 0.904]

cat_ps_1 = ['cat', 'ps', 0.947, 0.999, 0.982]
cat_ps_2 = ['cat', 'ps', 0.938, 0.999, 0.951]
cat_ps_3 = ['cat', 'ps', 0.929, 0.999, 0.978]
cat_ps_4 = ['cat', 'ps', 0.925, 0.999, 0.968]
cat_ps_5 = ['cat', 'ps', 0.925, 0.999, 0.925]

oxgb_ps_1 = ['oxgb', 'ps', 1.00, 1.00, 1.00]
oxgb_ps_2 = ['oxgb', 'ps', 1.00, 1.00, 1.00]
oxgb_ps_3 = ['oxgb', 'ps', 1.00, 1.00, 1.00]
oxgb_ps_4 = ['oxgb', 'ps', 1.00, 1.00, 1.00]
oxgb_ps_5 = ['oxgb', 'ps', 0.998, 0.999, 0.936]

ghostoxgb_ps_1 = ['ghostoxgb', 'ps', 1.00, 1.00, 1.00]
ghostoxgb_ps_2 = ['ghostoxgb', 'ps', 1.00, 1.00, 1.00]
ghostoxgb_ps_3 = ['ghostoxgb', 'ps', 1.00, 1.00, 1.00]
ghostoxgb_ps_4 = ['ghostoxgb', 'ps', 1.00, 1.00, 1.00]
ghostoxgb_ps_5 = ['ghostoxgb', 'ps', 0.982, 0.999, 0.932]

#-----------------------------------------------------#
import numpy as np
import pandas as pd
from statsmodels.multivariate.manova import MANOVA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
from scipy.stats import f_oneway


'''
https://www.reneshbedre.com/blog/manova-python.html
'''

data = [ghost_eu_1, ghost_eu_2, ghost_eu_3, ghost_eu_4, ghost_eu_5
        ,cat_eu_1, cat_eu_2, cat_eu_3, cat_eu_4, cat_eu_5
        ,oxgb_eu_1, oxgb_eu_2, oxgb_eu_3, oxgb_eu_4, oxgb_eu_5
        ,ghostoxgb_eu_1, ghostoxgb_eu_2, ghostoxgb_eu_3, ghostoxgb_eu_4, ghostoxgb_eu_5
        ,ghost_ps_1, ghost_ps_2, ghost_ps_3, ghost_ps_4, ghost_ps_5
        ,cat_ps_1, cat_ps_2, cat_ps_3, cat_ps_4, cat_ps_5
        ,oxgb_ps_1, oxgb_ps_2, oxgb_ps_3, oxgb_ps_4, oxgb_ps_5
        ,ghostoxgb_ps_1, ghostoxgb_ps_2, ghostoxgb_ps_3, ghostoxgb_ps_4, ghostoxgb_ps_5]
df = pd.DataFrame(data, columns = ['Method', 'Data', 'EE', 'AUROC', 'AUPRC'])

#----------Perform MANOVA-------------#
fit = MANOVA.from_formula('EE + AUPRC + AUROC ~ Method', data=df)
print(fit.mv_test())

X = df[['EE', 'AUROC', 'AUPRC']]

y = df['Method']

post_hoc = lda().fit(X=X, y=y)

#Get prior probabilities of groups
print('---------------------')
print(post_hoc.priors_)
