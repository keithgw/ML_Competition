"""Create null hypothesis test.csv"""

import pandas as pd
import numpy as np
import os
import re

# Get training label frequencies

train = pd.read_csv('../data/train.csv')
pdf_labels = list(np.mean(train.values[:, 1:], axis=0).astype(str))

# create test predictions
ids = [re.sub('\D', '', f) for f in os.listdir('../data/test/')]
yhat = np.zeros((len(ids), len(pdf_labels) + 1), dtype='|S32')
for i in range(len(ids)):
    yhat[i, :] = [ids[i]] + pdf_labels

# export as CSV
df = pd.DataFrame(data=yhat, columns=train.columns.values)
df.to_csv('../testY1.csv', index=False)