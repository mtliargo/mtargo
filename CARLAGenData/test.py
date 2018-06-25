'''
Generate CARLA Data
'''

import argparse
import logging
import math
import random
import time
import sys
import os
from os.path import join
import numpy as np

import sys
sys.path.insert(0, '../util')
import platform_config as pc

print(pc.hostname)
print(pc.data_dir)