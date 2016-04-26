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


def get_std_size(img_files):
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
    # Create list of image files
    #img_files = [os.path.join(path, f) for f in os.listdir(path)]
    
    # Get size for each image, log corrupted or empty files
    sizes = []
    bad_files = []
    
    for img_file in img_files:
        # check if image file empty or corrupt
        try:
            img = Image.open(img_file)
        except IOError as e:
            print e
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
    # open and resize
    img = Image.open(img_file).resize(new_size)
    
    # get RGB pixel values
    img = np.array(img.getdata())
    
    # convert to 1-D numpy array
    return img.reshape(img.size)
    
    
def main():
    IMG_DIR = '../data/train/'
    
    # import training labels
    train = pd.read_csv('../data/train.csv')
    
    # get image file paths
    img_files = [os.path.join(IMG_DIR, f) for f in os.listdir(IMG_DIR)]
    
    # get standard size and empty or corrupted files
    std_size, bad_files = get_std_size(img_files)
    
    # update image list and training labels to exclude empty or corrupted files
    img_files = sorted(set(img_files).difference(bad_files))
    bad_ids = [int(re.sub('\D', '', fpath)) for fpath in bad_files]
    trainY = train[~train.id.isin(bad_ids)]
    
    # Represent images as flattened RGB Matrices of equal length
    m, n = len(img_files), np.prod(std_size) * 3
    trainX = np.zeros((m, n))
    for i in range(m):
        trainX[i] = represent_image(img_files[i], std_size)
    
    
    
if __name__ == '__main__':
    main()