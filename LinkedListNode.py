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