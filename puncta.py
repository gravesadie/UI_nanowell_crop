import numpy as np
from scipy.ndimage import shift, binary_dilation, binary_closing, binary_opening
import tifffile as tiff
from pathlib import Path
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from collections import defaultdict
import pandas as pd
import os
from skimage.measure import regionprops, label
from skimage.feature import blob_log
from skimage.transform import rescale
import random
from datetime import datetime
from skimage.draw import disk
from skimage.feature import peak_local_max
from collections import Counter
import pickle
import gc

# iterate through Processed wells directory
# open each mask in BF folder and each image in mCherry folder
# insert 'tune_counting' function to check puncta detection before proceeding in bulk
# insert 'count_puncta' function to count puncta, with built in QC
# save pngs of annotations
# save excel sheet with puncta counts, puncta sizes, puncta intensities