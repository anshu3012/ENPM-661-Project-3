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

# ===== Map =====
# Map dimensions
xAxis = 300
yAxis = 200
rc = radius + clearance

# Generate map
fig, ax = plt.subplots()
ax.set(xlim=(0, xAxis), ylim = (0, yAxis))
plt.grid()
plt.xlim(0,xAxis)
plt.ylim(0,yAxis)
plt.title('ENPM661 Project-3 Phase-2 Group-38',fontsize=10)
plt.savefig('ENPM661-Project3-Phase2-Group38-map.png', bbox_inches='tight')

# To calculate the new intersection point of the lines after extension
def intersect_point(c1, c2):
    det = abs(c1[0] - c2[0])

    x_inter, y_inter = None, None
    if det != 0:
        x_inter = int(round(abs((c1[1] - c2[1])) / det))
        y_inter = int(round(abs(((c1[0] * c2[1]) - (c2[0] * c1[1]))) / det))

    return [x_inter, y_inter]

# Obstacles
# circle
cirX = 300 - 75 # center X pos
cirY = 200 - 50 # center Y pos
cirR = 25 + rc # radius
ax.add_artist(plt.Circle((cirX, cirY), cirR, color = "red"))
# ellipse
ellX = 300 - 150 # center X pos
ellY = 200 - 100 # center Y pos
ellA = 40 + rc # semi-major axis
ellB = 20 + rc # semi-minor axis
ax.add_artist(Ellipse((ellX, ellY), ellA*2, ellB*2, color = "red"))
# rhombus
rhoTX = 300 - 75 # top X pos
rhoTY = 10 + 30 # top Y pos
rhoRX = 300 - 75 + 50/2 # right X pos
rhoRY = 10 + 30/2 # right Y pos
rhoBX = 300 - 75 # bottom X pos
rhoBY = 10 # bottom Y pos
rhoLX = 300 - 75 - 50/2 # left X pos
rhoLY = 10 + 30/2 # left Y pos
rh1 = [rhoLX, abs(200 - rhoLY)] # left vertice
rh2 = [rhoTX, abs(200 - rhoTY)] # top vertice
rh3 = [rhoRX, abs(200 - rhoRY)] # right vertice
rh4 = [rhoBX, abs(200 - rhoBY)] # bottom vertice
def rhombus(rc):
    global rh1, rh2, rh3, rh4
    increase = rc
    coff1 = np.array(np.polyfit([rh1[0], rh2[0]], [rh1[1], rh2[1]], 1))
    coff2 = np.array(np.polyfit([rh2[0], rh3[0]], [rh2[1], rh3[1]], 1))
    coff3 = np.array(np.polyfit([rh3[0], rh4[0]], [rh3[1], rh4[1]], 1))
    coff4 = np.array(np.polyfit([rh4[0], rh1[0]], [rh4[1], rh1[1]], 1))
    if increase < 1:
        return [[rh1[0],abs(200-rh1[1])],[rh2[0],abs(200-rh2[1])],[rh3[0],abs(200-rh3[1])],[rh4[0],abs(200-rh4[1])]]
    else:
        # change the intercept formed by the lines
        coff1[1] = coff1[1] - (increase / (math.sin(1.57 - math.atan(coff1[0]))))
        coff2[1] = coff2[1] - (increase / (math.sin(1.57 - math.atan(coff2[0]))))
        coff3[1] = coff3[1] + (increase / (math.sin(1.57 - math.atan(coff3[0]))))
        coff4[1] = coff4[1] + (increase / (math.sin(1.57 - math.atan(coff4[0]))))
        rh2 = intersect_point(coff1, coff2)
        rh3 = intersect_point(coff2, coff3)
        rh4 = intersect_point(coff3, coff4)
        rh1 = intersect_point(coff4, coff1)
        return [[rh1[0],abs(200-rh1[1])],[rh2[0],abs(200-rh2[1])],[rh3[0],abs(200-rh3[1])],[rh4[0],abs(200-rh4[1])]]
rhombus = rhombus(rc)
ax.add_artist(Polygon(rhombus, color = "red"))
# Rectangle vertices (bottom-left polygon)
recBX = 95 # bottom X pos
recBY = abs(200 - 30) # bottom Y pos
recLX = 95 - 75*math.cos(30*math.pi/180) # left X pos
recLY = abs(200 - (30 + 75*math.sin(30*math.pi/180))) # left Y pos
recRX = 95 + 10*math.cos(60*math.pi/180) # right X pos
recRY = abs(200 - (30 + 10*math.sin(60*math.pi/180))) # right Y pos
recTX = 95 - 75*math.cos(30*math.pi/180) + 10*math.cos(60*math.pi/180) # top X pos
recTY = abs(200 - (30 + 75*math.sin(30*math.pi/180) + 10*math.sin(60*math.pi/180))) # top Y pos
# Rectangle vertice coordinates
r1 = [recTX, recTY]
r2 = [recRX, recRY]
r3 = [recBX, recBY]
r4 = [recLX, recLY]
def rectangle(rc):
    increase = rc
    global r1, r2, r3, r4
    coff1 = np.array(np.polyfit([r1[0], r2[0]], [r1[1], r2[1]], 1))
    coff2 = np.array(np.polyfit([r2[0], r3[0]], [r2[1], r3[1]], 1))
    coff3 = np.array(np.polyfit([r3[0], r4[0]], [r3[1], r4[1]], 1))
    coff4 = np.array(np.polyfit([r4[0], r1[0]], [r4[1], r1[1]], 1))
    if increase < 1:
        return [[r1[0],abs(200-r1[1])],[r2[0],abs(200-r2[1])],[r3[0],abs(200-r3[1])],[r4[0],abs(200-r4[1])]]

    else:
        # change the intercept formed by the lines
        coff1[1] = coff1[1] - (increase / (math.sin(1.57 - math.atan(coff1[0]))))
        coff2[1] = coff2[1] + (increase / (math.sin(1.57 - math.atan(coff2[0]))))
        coff3[1] = coff3[1] + (increase / (math.sin(1.57 - math.atan(coff3[0]))))
        coff4[1] = coff4[1] - (increase / (math.sin(1.57 - math.atan(coff4[0]))))
        r2 = intersect_point(coff1, coff2)
        r3 = intersect_point(coff2, coff3)
        r4 = intersect_point(coff3, coff4)
        r1 = intersect_point(coff4, coff1)
        return [[r1[0],abs(200-r1[1])],[r2[0],abs(200-r2[1])],[r3[0],abs(200-r3[1])],[r4[0],abs(200-r4[1])]]

rectangle = rectangle(rc)
ax.add_artist(Polygon(rectangle, color = "red"))
# custom polygon
p1 = [25, 15]
p2 = [75, 15]
p3 = [100, 50]
p4 = [75, 80]
p5 = [50, 50]
p6 = [20, 80]
def polygon(rc):
    increase = rc
    global p1, p2, p3, p4, p5, p6
    coff1 = np.array(np.polyfit([p1[0], p2[0]], [p1[1], p2[1]], 1))
    coff2 = np.array(np.polyfit([p2[0], p3[0]], [p2[1], p3[1]], 1))
    coff3 = np.array(np.polyfit([p3[0], p4[0]], [p3[1], p4[1]], 1))
    coff4 = np.array(np.polyfit([p4[0], p5[0]], [p4[1], p5[1]], 1))
    coff5 = np.array(np.polyfit([p5[0], p6[0]], [p5[1], p6[1]], 1))
    coff6 = np.array(np.polyfit([p6[0], p1[0]], [p6[1], p1[1]], 1))

    if increase < 1:
        return [[p1[0],abs(200-p1[1])],[p2[0],abs(200-p2[1])],[p3[0],abs(200-p3[1])],\
                [p4[0],abs(200-p4[1])],[p5[0],abs(200-p5[1])],[p6[0],abs(200-p6[1])]]

    else:
        # change the intercept formed by the lines
        coff1[1] = coff1[1] - (increase / (math.sin(1.57 - math.atan(coff1[0]))))
        coff2[1] = coff2[1] - (increase / (math.sin(1.57 - math.atan(coff2[0]))))
        coff3[1] = coff3[1] + (increase / (math.sin(1.57 - math.atan(coff3[0]))))
        coff4[1] = coff4[1] + (increase / (math.sin(1.57 - math.atan(coff4[0]))))
        coff5[1] = coff5[1] + (increase / (math.sin(1.57 - math.atan(coff5[0]))))
        coff6[1] = coff6[1] - (increase / (math.sin(1.57 - math.atan(coff6[0]))))

        # Keep the slope constant but changing intercept find the intersection point
        p2 = intersect_point(coff1, coff2)
        p3 = intersect_point(coff2, coff3)
        p4 = intersect_point(coff3, coff4)
        p5 = intersect_point(coff4, coff5)
        p6 = intersect_point(coff5, coff6)
        p1 = intersect_point(coff6, coff1)

        return [[p1[0],abs(200-p1[1])],[p2[0],abs(200-p2[1])],[p3[0],abs(200-p3[1])],\
                [p4[0],abs(200-p4[1])],[p5[0],abs(200-p5[1])],[p6[0],abs(200-p6[1])]]
polygon = polygon(rc)
ax.add_artist(Polygon(polygon, color = "red"))
# border
borderTop = [[300,200],[300,200 - rc],[0,200 - rc],[0,200]] # top border
borderRight = [[300,200],[300,0],[300 - rc,0],[300 - rc,200]] # right border
borderBottom = [[300,0 + rc],[300,0],[0,0],[0,0 + rc]] # bottom border
borderLeft = [[0 + rc,200],[0 + rc,0],[0,0],[0,200]] # left border
ax.add_artist(Polygon(borderTop, color = "red"))
ax.add_artist(Polygon(borderRight, color = "red"))
ax.add_artist(Polygon(borderBottom, color = "red"))
ax.add_artist(Polygon(borderLeft, color = "red"))

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

# ===== Node class and functions =====
class Node:
    def __init__(self, point):
        self.point = point # [x, y, theta]
        self.cost = math.inf  # initially all the new nodes have infinite cost attached to them
        self.parent = None

def pop_queue_element(queue):  # Priority Queue, outputs the node with least cost attached to it
    min_a = 0
    for elemt in range(len(queue)):
        if queue[elemt].cost < queue[min_a].cost:
            min_a = elemt
    return queue.pop(min_a)

def count_entry_points(point):
    point_x = point[0]
    point_y = point[1]
    count = 0
    if point_y > 0:
        count += 1
    if point_y < 200:
        count += 1
    if point_x > 0:
        count += 1
    if point_x < 300:
        count += 1
    if point_x < 300 and point_y > 0:
        count += 1
    if point_x > 0 and point_y > 0:
        count += 1
    if point_x < 300 and point_y < 200:
        count += 1
    if point_x > 0 and point_y < 200:
        count += 1
    return count

def cost_to_goal(point, goal_node_pos):
    point = [point[0],point[1]]
    point_x = point[0]
    point_y = point[1]
    goal_x = goal_node_pos[0]
    goal_y = goal_node_pos[1]
    euc_dist = np.sqrt((point_x-goal_x)**2 + (point_y-goal_y)**2)
    point = np.array(point)
    goal = np.array(goal_node_pos)
    euc_dist = np.linalg.norm(point - goal)
    return euc_dist

def find_node(point, queue):
    for elem in queue:
        if elem.point == [[int(point[0]/thresholdXY)],[int(point[1]/thresholdXY)],[int(point[2]/thresholdTheta)]]:
            return queue.index(elem)
        else:
            return None

# ===== Generate graph =====
ax.set_aspect('equal')
print("Generating graph")
# A star search algorithm
def a_star_algo(rc, step_size, start_node_pos, goal_node_pos):
    # Initial parameters
    start_node = Node(start_node_pos)
    start_node.cost = 0
    entry_points = count_entry_points(goal_node_pos)
    visited = np.zeros([int(300/thresholdXY),int(200/thresholdXY),int(360/thresholdTheta)])
    queue = [start_node]
    actions = ["move_straight", "move_up1", "move_up2", "move_down1", "move_down2"]
    #actions = ["move_straight"]
    counter = 0
    # Show goal region
    ax.add_artist(plt.Circle((goal_cords_x, goal_cords_y), 1.5, color = "green"))
    # A star search
    while queue:
        current_node = pop_queue_element(queue)
        current_point = current_node.point
        visited[int(current_point[0]/thresholdXY)][int(current_point[1]/thresholdXY)][int(current_point[2]/thresholdTheta)] = 1
        if ((current_point[0] - goal_node_pos[0])**2 + (current_point[1] - goal_node_pos[1])**2) <= 1.5**2:
            print("Goal reached")
            return new_node.parent

        for action in actions:
            new_point, base_cost = generate_node_location(action, current_point)
            if new_point is not None:

                new_node = Node(new_point)
                new_node.parent = current_node

                if visited[int(new_point[0]/thresholdXY)][int(new_point[1]/thresholdXY)][int(new_point[2]/thresholdTheta)] == 0:
                    new_node.cost = base_cost + new_node.parent.cost + cost_to_goal(new_point, goal_node_pos)
                    visited[int(new_point[0]/thresholdXY)][int(new_point[1]/thresholdXY)][int(new_point[2]/thresholdTheta)] = 1
                    queue.append(new_node)
                    X = np.array(current_point[0])
                    Y = np.array(current_point[1])
                    U = np.array(new_point[0]) - X
                    V = np.array(new_point[1]) - Y
                    ax.quiver(X,Y,U,V,units='xy' ,scale=1, color= 'black',headwidth = 1,headlength=0)
                else:
                    node_exist_index = find_node(new_point, queue)
                    if node_exist_index is not None:
                        temp_node = queue[node_exist_index]
                        if temp_node.cost > base_cost + new_node.parent.cost + cost_to_goal(new_point, goal_node_pos):
                            temp_node.cost = base_cost + new_node.parent.cost + cost_to_goal(new_point, goal_node_pos)
                            temp_node.parent = current_node
            else:
                continue
    return None

start_node_pos = [start_cords_x, start_cords_y, start_theta]
goal_node_pos = [goal_cords_x, goal_cords_y] # goal theta ignored
# Record time
start_time = time.time()
# Search map
result = a_star_algo(rc, step_size, start_node_pos, goal_node_pos)
# Print final time
print("Time explored = %2.3f seconds " % (time.time() - start_time))

# ===== Backtrack =====

print("Backtracking...")
def track_back(node):
    p = list()
    p.append(node.parent)
    parent = node.parent
    if parent is None:
        return p
    while parent is not None:
        p.append(parent)
        parent = parent.parent
    p_rev = list(p)
    return p_rev

if result is not None:
    nodes_list = track_back(result)
    x = []
    y = []
    for elem in nodes_list:
        x.insert(0,elem.point[0])
        y.insert(0,elem.point[1])
    for index in range(1,len(x)):
        X = x[index - 1]
        Y = y[index - 1]
        U = x[index] - X
        V = y[index] - Y
        #print("x ", X, "y ", Y, "u ", U, "v ", V)
        plt.quiver(X,Y,U,V,units='xy' ,scale=1, color= 'blue',headwidth = 1,headlength=0)
    print("Search complete. Close window to close program")

else:
    print("Sorry, result could not be reached")

# Show map
plt.show()
plt.close()

# END
