import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
from window import InputWindow


NUMBER = 4

inputWindow = InputWindow(NUMBER)
inputWindow.show()

if inputWindow.interrupted is True:
    exit(1)
