# ENPM661
# Project 3 Phase 2
# Group 38

# ===== Libraries =====
import numpy as np
import cv2
import math
from math import pi
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.patches import Polygon
import time

# ===== User Inputs (Error-checking and robot dimensions) =====
print("This code will implement A* Algorithm for a Rigid Robot")
# Error-checking functions
def inputNum(message):
    """Only takes an integer as input"""
    while True:
        user = input(message)
        try:
            output = int(user)
            break;
        except ValueError:
            print('\n!!! ERROR !!! Input needs to be a number. Please try again.\n')
    return output

def inputNumZeroPos(message):
    """Only allows numbers 0 and above"""
    output = inputNum(message)
    while output < 0:
        print("\n!!! ERROR !!! Please enter a number that is 0 or positive.\n")
        output = inputNum(message)
    return output

def inputNumX(message):
    """Only allows numbers between start and end"""
    output = inputNum(message)
    while output < 0 or output > 300:
        print("\n!!! ERROR !!! Please enter a number between 0 and 300.\n")
        output = inputNum(message)
    return output

def inputNumY(message):
    """Only allows numbers between start and end"""
    output = inputNum(message)
    while output < 0 or output > 200:
        print("\n!!! ERROR !!! Please enter a number between 0 and 200.\n")
        output = inputNum(message)
    return output

# Robot radius and clearance
radius = inputNumZeroPos("Please enter the radius of the rigid robot: ")
clearance = inputNumZeroPos("Please enter the desired clearance: ")
