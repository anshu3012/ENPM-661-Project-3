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
