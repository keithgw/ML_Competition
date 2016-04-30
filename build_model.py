"""
Train model
"""

import numpy as np
import pandas as pd
from process_data import represent_image
from PIL import ImageFile
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import log_loss
import re
import dill
import os


# Allow for truncated images to load
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Declare constants
TRAIN_FILE = 'train_pca.csv'
VAL_FILE = 'val_pca.csv'
STD_SIZE = (350, 233)
IMG_DIR = '../data/test/'

# load training and validation data
train = pd.read_csv(TRAIN_FILE)
val = pd.read_csv(VAL_FILE)

# convert label matrix to label array for model fitting
train['y'] = '0'
val['y'] = '0'
for label in range(1, 9):
    cname = 'col' + str(label)
    train['y'][train[cname] == 1] = str(label)
    val['y'][val[cname] == 1] = str(label)

# get training and validation pca values
trainX = train.select(lambda x: re.match('pc', x), axis=1)
valX = val.select(lambda x: re.match('pc', x), axis=1)


def prepare_test_data(file_list, resize):
    # load RandomizedPCA class object
    with open('pca.pkl', 'rb') as f:
        pca = dill.load(f)
        
    # initialize
    n_samples, n_components = len(file_list), pca.get_params()['n_components']
    pca_out = np.zeros((n_samples, n_components))
    bad_indices = []
    for i in range(n_samples):
        try:
            img = represent_image(file_list[i], resize).reshape(1, -1)
        except IOError as e:
            print e, 'no prediction will be made'
            bad_indices.append(i)
        else:
            pca_out[i] = pca.transform(img)
    
    # Remove bad files
    mask = np.ones(n_samples).astype(bool)
    mask[bad_indices] = False
    id_s = [re.sub('\D', '', f) for f in np.array(file_list)[mask]]
    
    # Create data frame
    pc_col_names = ['pc' + str(i) for i in range(n_components)]
    df_id = pd.DataFrame({'id' : id_s})
    df_pc = pd.DataFrame(data=pca_out[mask], columns=pc_col_names)
    test_pca = pd.concat([df_id, df_pc], axis=1)
    if not os.path.isfile('test_pca.csv'):
        test_pca.to_csv('test_pca.csv', index=False)
    
    return test_pca            


def make_predictions():
    # Fit Logistic Regression Model
    logreg = LogisticRegressionCV(scoring='log_loss', n_jobs=-1, verbose=1, random_state=6156)
    logreg.fit(X=trainX, y=train['y'].values)
    
    # Validate
    pred_pr = logreg.predict_proba(valX)
    loss = log_loss(y_true=val['y'].values, y_pred=pred_pr)
    print "Validation log loss:", loss
    
    # Get Test predictions
    img_files = [os.path.join(IMG_DIR, f) for f in os.listdir(IMG_DIR)]
        
    if os.path.isfile('test_pca.csv'):
        test_pca = pd.read_csv('test_pca.csv', dtype={'id' : str})
    else:
        test_pca = prepare_test_data(img_files, STD_SIZE)
        
    test_predictions = logreg.predict_proba(test_pca.values[:, 1:])
    id_s = [re.sub('\D', '', f) for f in img_files]
    df_id = pd.DataFrame({'id' : id_s})
    col_names = ['col'+str(i) for i in range(1, 9)]
    df_yhat = pd.DataFrame(data=test_predictions, columns=col_names)
    df_id_yhat = pd.concat([test_pca['id'], df_yhat], axis=1)
    yhat = df_id.merge(df_id_yhat, on='id', how='left')
    yhat.fillna(1./8, inplace=True)
    yhat.to_csv('kaggle_430_2pm.csv', index=False)
    
    
if __name__ == '__main__':
    make_predictions()