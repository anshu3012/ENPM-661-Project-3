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

# ===== Action Set =====
# Threshold
thresholdXY = 0.5 # Euclidean threshold distance of 0.5 units for x,y
thresholdTheta = 30 # Angle [deg] threshold for theta

def move_straight(point):
    point_x = point[0]
    point_y = point[1]
    theta = point[2]
    base_cost = step_size
    if inside_obstacle(point_x,point_y) == False:
        new_point = [point_x + step_size*math.cos(theta), point_y +  step_size*math.sin(theta), theta]
        return new_point, base_cost
    else:
        return None, None

def move_up1(point):
    point_x = point[0]
    point_y = point[1]
    theta = point[2] + 30*math.pi/180
    base_cost = step_size
    if inside_obstacle(point_x,point_y) == False:
        new_point = [point_x +  step_size*math.cos(theta), point_y +  step_size*math.sin(theta), theta]
        return new_point, base_cost
    else:
        return None, None

def move_up2(point):
    point_x = point[0]
    point_y = point[1]
    theta = point[2] + 2*30*math.pi/180
    base_cost = step_size
    if inside_obstacle(point_x,point_y) == False:
        new_point = [point_x +  step_size*math.cos(theta), point_y +  step_size*math.sin(theta), theta]
        return new_point, base_cost
    else:
        return None, None

def move_down1(point):
    point_x = point[0]
    point_y = point[1]
    theta = point[2] - 30*math.pi/180
    base_cost = step_size
    if inside_obstacle(point_x,point_y) == False:
        new_point = [point_x +  step_size*math.cos(theta), point_y +  step_size*math.sin(theta), theta]
        return new_point, base_cost
    else:
        return None, None

def move_down2(point):
    point_x = point[0]
    point_y = point[1]
    theta = point[2] - 2*30*math.pi/180
    base_cost = step_size
    if inside_obstacle(point_x,point_y) == False:
        new_point = [point_x +  step_size*math.cos(theta), point_y +  step_size*math.sin(theta), theta]
        return new_point, base_cost
    else:
        return None, None

def generate_node_location(action, point):
    if action == 'move_straight':
        return move_straight(point)
    if action == 'move_up1':
        return move_up1(point)
    if action == 'move_up2':
        return move_up2(point)
    if action == 'move_down1':
        return move_down1(point)
    if action == 'move_down2':
        return move_down2(point)
