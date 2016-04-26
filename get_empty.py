"""
Find empty files, and remove the files
and delete the corresponding label entry in train.csv
"""

import os

def get_empty(path):  
    """
    Parameters
    ----------
    path : str
        Path to directory of image files.
    """
    return os.path.getsize(path) > 0

if __name__ == "__main__":
    get_empty('../data/train/')    
