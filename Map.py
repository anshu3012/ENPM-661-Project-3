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