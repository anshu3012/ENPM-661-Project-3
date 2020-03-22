
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
        print("x ", X, "y ", Y, "u ", U, "v ", V)
        plt.quiver(X,Y,U,V,units='xy' ,scale=1, color= 'blue',headwidth = 1,headlength=0)
    print("Search complete. Close window to close program")

else:
    print("Sorry, result could not be reached")

# Show map
plt.show()
plt.close()

# END
