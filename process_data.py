"""
Pre Process the training data

1. Find any corrupted or empty files to remove from the training labels
2. Resize all of the images to the median image size
3. Extract RGB values and convert to a vector
"""

import os
import re
import numpy as np
from PIL import Image, ImageFile
import pandas as pd
#from sklearn.cross_validation import train_test_split
from sklearn.decomposition import RandomizedPCA
from time import time
import dill

# Allow for truncated images to load
ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_med_size(img_files):
    """
    Parameters
    ----------
    img_files : list
        List of image files with path
        
    Returns
    -------
    tuple (size, list_files), where size is a (l, w) tuple of the median image
        size, and list_files is a list of empty or damaged files that could
        not be opened.
    """
    # Get size for each image, log corrupted or empty files
    sizes = []
    bad_files = []
    
    for img_file in img_files:
        # check if image file empty or corrupt
        try:
            img = Image.open(img_file)
        except IOError as e:
            print e, 'removing from training set'
            # log image ID, so training label can be discarded later
            m = re.search('\.\./data/train/[0123456789]*\.jpg', e[0])
            bad_files.append(m.group(0))
        else:
            sizes.append(img.size)
    
    # get median image size        
    median_size = np.median(np.array(sizes), axis=0)
    
    return (tuple(median_size.astype(int)), bad_files)
    
    
def represent_image(img_file, new_size):
    """
    Parameters
    ----------
    img_file : str
        path to image file to be resized and converted to feature vector
    size : tuple
        standard image size to which image is resized
        
    Returns
    -------
    numpy array of RGB value matrix flattend to 1-D array
    """
    # open and convert to RGB if not already
    img = Image.open(img_file)
    if img.mode != 'RGB':
        img = img.convert(mode='RGB')
    
    # Resize image
    img = img.resize(new_size)
    
    # get RGB pixel values
    img = np.array(img.getdata())
    
    # convert to 1-D numpy array
    return img.reshape(img.size)
    

def reduce_dim(file_list, resize, pca=None):
    """
    Parameters
    ----------
    file_list : array like
        List of image files to be transformed using principal components
    resize  : tuple
        Size to which images in file_list should be resized
    pca       : class RandomizedPCA
        Fit to training data.
        If None, pca will be created from file_list
    
    Returns
    -------
    tuple :
        pca class object,
        numpy array (n_files, n_pca_components) of reduced dimensions
    """
    # Represent images as flattened RGB Matrices of equal length
    m, n = len(file_list), np.prod(resize) * 3
    raw_data = np.zeros((m, n), dtype='uint8')
    print 'Converting {} images into RGB Vectors'.format(m)
    t0 = time()
    for i in range(m):
        raw_data[i] = represent_image(file_list[i], resize)
    print 'Finished in {sec:.{rd}f} seconds'.format(sec=time() - t0, rd=2)
    
    if pca is None:
        # Get principal components from half the training data
        components = 30 # determined by plotting PC
        print 'Calculating principal components'
        t0 = time()
        pca = RandomizedPCA(n_components = components)
        pca.fit(raw_data)
        print 'Finished in {sec:.{rd}f} seconds'.format(sec=time() - t0, rd=2)
        
        # Save pca object
        with open('pca.pkl', 'wb') as f:
            dill.dump(pca, f)

        
    # create pandas DataFrame attaching pca representation to img id
    n_pc = pca.get_params()['n_components']
    pc_col_names = ['pc' + str(i) for i in range(n_pc)]
    print 'Transforming Raw Data to {} Principal Components'.format(n_pc)
    t0 = time()
    pc_df = pd.DataFrame(data=pca.transform(raw_data), columns=pc_col_names)
    print 'Finished in {sec:.{rd}f} seconds'.format(sec=time() - t0, rd=2)
    id_lst = [re.sub('\D', '', f) for f in file_list]
    id_dict = {'id' : id_lst}
    id_df = pd.DataFrame(data=id_dict)
    principal_components = pd.concat([id_df, pc_df], axis=1)
            
    return (pca, principal_components)
    
    
def main():
    IMG_DIR = '../data/train/'
    STD_SIZE = (350, 233)   #Change to None to recalculate
    BAD_FILES = ['../data/train/11402.jpg', '../data/train/36911.jpg']

################################################################################
## Pre-Process the Data   
            
    # import training labels
    train = pd.read_csv('../data/train.csv', dtype={'id' : str})
    train.sort_values('id', inplace=True) # Same order as img_file names
    
    # get image file paths
    img_files = [os.path.join(IMG_DIR, f) for f in os.listdir(IMG_DIR)]
    img_files.sort() # ensure labels and examples are in same order
    
    # get standard size and empty or corrupted files
    if STD_SIZE is None:
        t0 = time()
        print 'Getting median image size.'
        std_size, bad_files = get_med_size(img_files)
        print 'finished in {sec:.{rd}f} seconds'.format(sec=time() - t0, rd=2)
    else:
        std_size, bad_files = (STD_SIZE, BAD_FILES)
                    
    # update image list and training labels to exclude empty or corrupted files
    #img_files = sorted(set(img_files).difference(bad_files))
    img_clean = np.array([f for f in img_files if f not in bad_files])
    bad_ids = [re.sub('\D', '', fpath) for fpath in bad_files]
    train_clean = train[~train.id.isin(bad_ids)]
    
    # Partition the training data into train and validation
    np.random.seed(6156)
    test_pct = 0.25
    in_train = np.random.uniform(size = len(img_clean)) > test_pct
    Xtrainf, Xtestf = img_clean[in_train], img_clean[~in_train]
    Ytrain, Yval = train_clean[in_train], train_clean[~in_train]
    
    # Sample the training data to find principal components
    pca_pct = 1./2 # Only half the data fits in RAM at one time
    in_pca = np.random.uniform(size = len(Xtrainf)) < pca_pct
    pca_files = Xtrainf[in_pca]
    
    # Use Randomized PCA to reduce the dimensions of the images
    if os.path.isfile('pca.pkl'):
        with open('pca.pkl', 'rb') as f:
            pca = dill.load(f)
    else:
        pca = None
    pca, pca_transformed = reduce_dim(file_list=pca_files, resize=std_size, pca=pca)
    pca, remaining_train = reduce_dim(file_list=Xtrainf[~in_pca], resize=std_size, pca=pca)   
        
    # Recombine training data and write to csv
    df_train_x = pd.concat([pca_transformed, remaining_train], axis=0)
    df_train = df_train_x.merge(Ytrain, on='id', how='inner')
    df_train.to_csv('./train_pca.csv', index=False)
    
    # Process validation data
    pca, Xval = reduce_dim(file_list=Xtestf, resize=std_size, pca=pca)
    df_val = Xval.merge(Yval, on='id', how='inner')
    df_val.to_csv('./val_pca.csv', index=False)
    
            
        
if __name__ == '__main__':
    main()