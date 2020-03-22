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

# ===== User Inputs (Coordinates) =====
# Start point coordinates
start_cords_x = inputNumX("Please enter the starting x coordinate: ")
start_cords_y = inputNumY("Please enter the starting y coordinate: ")
# Obstacle check
while inside_obstacle(start_cords_x,start_cords_y) == 1:
        print("\n!!! ERROR !!! Coordinates inside obstacle. Please try again.\n")
        start_cords_x = inputNumX("Please enter the starting x coordinate: ")
        start_cords_y = inputNumY("Please enter the starting y coordinate: ")
# Goal point coordinates
goal_cords_x = inputNumX("Please enter the goal x coordinate: ")
goal_cords_y = inputNumY("Please enter the goal y coordinate: ")
# Obstacle check
while inside_obstacle(goal_cords_x,goal_cords_y) == 1:
        print("\n!!! ERROR !!! Coordinates inside obstacle. Please try again.\n")
        goal_cords_x = inputNumX("Please enter the goal x coordinate: ")
        goal_cords_y = inputNumY("Please enter the goal y coordinate: ")
# Step size movement
step_size = inputNum("Please enter step size movement of unit between 1 and 10: ")
while step_size > 10 or step_size < 1:
    print("\n!!! ERROR !!!\nPlease enter a step size unit between 1 and 10. Please try again.\n")
    step_size = int(input("Please enter step size movement of unit between 1 and 10"))
# Starting angle
start_theta = inputNum("Please enter the starting angle in degrees: ")
start_theta = start_theta*pi/180 # convert deg to radians
