"""
Pre Process the training data

1. Find any corrupted or empty files to remove from the training labels
2. Resize all of the images to the median image size
3. Extract RGB values and convert to a vector
"""

import os
import re
import numpy as np
from PIL import Image
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import RandomizedPCA

# Allow for truncated images to load
from PIL import ImageFile
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
    
    
def max_std_size(n_samples, img_dim, max_bytes):
    """
    Parameters
    ----------
    n_samples : int
        number of training images
    img_dim   : tuple
        dimensions of equally sized images
        
    Returns
    -------
    tuple, new standard size that maintains dimension ratio 
        if memory of array of RGB pixels exceeds max_bytes
    """
    d1, d2 = img_dim
    ratio = float(d1) / d2
    
    bytes_needed = n_samples * d1 * d2 * 3 #3 RGB values per pixel
    if bytes_needed > max_bytes:
        scale = max_bytes / (d2 * ratio * d2)
        new_d1, new_d2 = int(d1 * scale), int(d2 * scale)
        new_dim = (new_d1, new_d2)
    else:
        new_dim = img_dim
        
    return new_dim        
    
    
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
    # open and resize
    img = Image.open(img_file).resize(new_size)
    
    # get RGB pixel values
    img = np.array(img.getdata())
    
    # convert to 1-D numpy array
    return img.reshape(img.size)
    
    
def main():
    IMG_DIR = '../data/train/'
    MAX_BYTES = 4e9  # 4 GB maximum for np.array representation of images

################################################################################
## Pre-Process the Data   
            
    # import training labels
    train = pd.read_csv('../data/train.csv')
    
    # get image file paths
    img_files = [os.path.join(IMG_DIR, f) for f in os.listdir(IMG_DIR)]
    
    ## FIGURE OUT SORTING sort this list as strings, sort pd.trainX by id as string
    
    # get standard size and empty or corrupted files
    med_size, bad_files = get_med_size(img_files)
    std_size = max_std_size(n_samples = len(img_files) - len(bad_files), 
        img_dim = med_size, max_bytes = MAX_BYTES)
    print "standard size: ", std_size
       
    # update image list and training labels to exclude empty or corrupted files
    #img_files = sorted(set(img_files).difference(bad_files))
    img_clean = [f for f in img_files if f not in bad_files]
    bad_ids = [int(re.sub('\D', '', fpath)) for fpath in bad_files]
    train_clean = train[~train.id.isin(bad_ids)]
    trainY = train_clean.values[:, 1:]
    
    # Represent images as flattened RGB Matrices of equal length
    m, n = len(img_clean), np.prod(std_size) * 3
    trainX = np.zeros((m, n), dtype='uint8')
    for i in range(m):
        print img_clean[i]
        trainX[i] = represent_image(img_clean[i], std_size)
    
################################################################################
## Partition the training data into train and validation

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(
        trainX, trainY, test_size=0.25, random_state=6156)
    
    
    
if __name__ == '__main__':
    main()