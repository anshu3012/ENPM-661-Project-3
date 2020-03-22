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

# ===== Obstacle Check =====
def inside_obstacle(x,y):
    """Returns true if x,y coordinate is inside obstacle
    Returns false if x,y coordinate is inside obstacle"""
    global rh1, rh2, rh3, rh4, r1, r2, r3, r4, p1, p2, p3, p4, p5, p6
    y = abs(200 - y) # convert from opencv to matplotlib
    # circle
    if ((x - cirX)**2 + (y - cirY)**2) <= cirR**2:
        #print("HIT CIRCLE")
        return True
    # ellipse
    if ((x - ellX)**2 / ellA**2) + ((y - ellY)**2 / ellB**2) <= 1:
        #print("HIT ELLIPSE")
        return True
    # rhombus
    coff1 = np.polyfit([rh1[0], rh2[0]], [rh1[1], rh2[1]], 1)
    coff2 = np.polyfit([rh2[0], rh3[0]], [rh2[1], rh3[1]], 1)
    coff3 = np.polyfit([rh3[0], rh4[0]], [rh3[1], rh4[1]], 1)
    coff4 = np.polyfit([rh4[0], rh1[0]], [rh4[1], rh1[1]], 1)

    line1 = round(y - coff1[0] * x - (coff1[1]))
    line2 = round(y - coff2[0] * x - (coff2[1]))
    line3 = round(y - coff3[0] * x - (coff3[1]))
    line4 = round(y - coff4[0] * x - (coff4[1]))

    if line1>=0 and line3<=0 and line2>=0 and line4<=0:
        #print("HIT RHOMBUS")
        return True
    # rectangle
    coff1 = np.polyfit([r1[0], r2[0]], [r1[1], r2[1]], 1)
    coff2 = np.polyfit([r2[0], r3[0]], [r2[1], r3[1]], 1)
    coff3 = np.polyfit([r3[0], r4[0]], [r3[1], r4[1]], 1)
    coff4 = np.polyfit([r4[0], r1[0]], [r4[1], r1[1]], 1)

    line1 = round(y - coff1[0] * x - (coff1[1]))
    line2 = round(y - coff2[0] * x - (coff2[1]))
    line3 = round(y - coff3[0] * x - (coff3[1]))
    line4 = round(y - coff4[0] * x - (coff4[1]))

    if line1>=0 and line3<=0 and line4>=0 and line2<=0:
        #print("HIT RECTANGLE")
        return True
    # custom polygon
    coff1 = np.polyfit([p1[0], p2[0]], [p1[1], p2[1]], 1)
    coff2 = np.polyfit([p2[0], p3[0]], [p2[1], p3[1]], 1)
    coff3 = np.polyfit([p3[0], p4[0]], [p3[1], p4[1]], 1)
    coff4 = np.polyfit([p4[0], p5[0]], [p4[1], p5[1]], 1)
    coff5 = np.polyfit([p5[0], p6[0]], [p5[1], p6[1]], 1)
    coff6 = np.polyfit([p6[0], p1[0]], [p6[1], p1[1]], 1)

    # To divide the figure into 4 triangles we make temporary lines
    coff_temp51 = np.polyfit([p5[0], p1[0]], [p5[1], p1[1]], 1)
    coff_temp52 = np.polyfit([p5[0], p2[0]], [p5[1], p2[1]], 1)
    coff_temp53 = np.polyfit([p5[0], p3[0]], [p5[1], p3[1]], 1)

    line1 = round(y - coff1[0] * x - (coff1[1]))
    line2 = round(y - coff2[0] * x - (coff2[1]))
    line3 = round(y - coff3[0] * x - (coff3[1]))
    line4 = round(y - coff4[0] * x - (coff4[1]))
    line5 = round(y - coff5[0] * x - (coff5[1]))
    line6 = round(y - coff6[0] * x - (coff6[1]))
    line_temp51 = round(y - coff_temp51[0] * x - (coff_temp51[1]))
    line_temp52 = round(y - coff_temp52[0] * x - (coff_temp52[1]))
    line_temp53 = round(y - coff_temp53[0] * x - (coff_temp53[1]))

    # Different flags for different triangles in the poly shape
    flag1 = False
    flag2 = False
    flag3 = False
    flag4 = False
    if line1 >= 0 and line_temp51 <= 0 and line_temp52 <= 0:
        flag1 = True
    if line_temp52 >= 0 and line2 >= 0 and line_temp53 <= 0:
        flag2 = True
    if line_temp53 >= 0 and line3 <= 0 and line4 <= 0:
        flag3 = True
    if line_temp51 >= 0 and line5 <= 0 and line6 >= 0:
        flag4 = True
    if flag1 or flag2 or flag3 or flag4 is True:
        #print("HIT POLYGON")
        return True
    # border
    b1 = [rc, rc]
    b2 = [300 - rc, rc]
    b3 = [300 - rc, 200 - rc]
    b4 = [rc, 200 - rc]
    if x <= b1[0] or x >= b2[0]:
        #print("HIT BORDER")
        return True
    if y <= b1[1] or y >= b3[1]:
        #print("HIT BORDER")
        return True
    return False
