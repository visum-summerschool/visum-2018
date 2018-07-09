import cv2
import numpy as np
import skimage.filters
from utils.priodict import priorityDictionary
from shapely.geometry import Polygon,Point
import scipy.interpolate as interpolate
import numpy
import scipy.ndimage.morphology as morpho
from PIL import Image, ImageDraw

"""
_______________________________________________________________________________
        SHORTEST PATH FOR ARBITRARY GRAPHS
_______________________________________________________________________________
"""

def Dijkstra(G,start,end=None):
    """
    Find shortest paths from the start vertex to all
    vertices nearer than or equal to the end.

    The input graph G is assumed to have the following
    representation: A vertex can be any object that can
    be used as an index into a dictionary.  G is a
    dictionary, indexed by vertices.  For any vertex v,
    G[v] is itself a dictionary, indexed by the neighbors
    of v.  For any edge v->w, G[v][w] is the length of
    the edge.  This is related to the representation in
    <http://www.python.org/doc/essays/graphs.html>
    where Guido van Rossum suggests representing graphs
    as dictionaries mapping vertices to lists of neighbors,
    however dictionaries of edges have many advantages
    over lists: they can store extra information (here,
    the lengths), they support fast existence tests,
    and they allow easy modification of the graph by edge
    insertion and removal.  Such modifications are not
    needed here but are important in other graph algorithms.
    Since dictionaries obey iterator protocol, a graph
    represented as described here could be handed without
    modification to an algorithm using Guido's representation.

    Of course, G and G[v] need not be Python dict objects;
    they can be any other object that obeys dict protocol,
    for instance a wrapper in which vertices are URLs
    and a call to G[v] loads the web page and finds its links.
    
    The output is a pair (D,P) where D[v] is the distance
    from start to v and P[v] is the predecessor of v along
    the shortest path from s to v.
    
    Dijkstra's algorithm is only guaranteed to work correctly
    when all edge lengths are positive. This code does not
    verify this property for all edges (only the edges seen
    before the end vertex is reached), but will correctly
    compute shortest paths even for some graphs with negative
    edges, and will raise an exception if it discovers that
    a negative edge has caused it to make a mistake.
    """

    D = {}	# dictionary of final distances
    P = {}	# dictionary of predecessors
    Q = priorityDictionary()   # est.dist. of non-final vert.
    Q[start] = 0
    
    for v in Q:
        D[v] = Q[v]
        if v == end: break
        
        for w in G[v]:
            vwLength = D[v] + G[v][w]
        
        # THE FOLLOWING LINES OF CODE ALLOW TO CHANGE THE Dijkstra into the A* algorithm
        #for w in G[v]:
        #    if end:  # use A*, if end node is provided
        #        def heuristic(n1, n2):  # using Manhattan distance as the heuristic
        #            return abs(n1[0]-n2[0]) + abs(n1[1]-n2[1])
        #        vwLength = D[v] + G[v][w] + heuristic(w, end)
        #
        #    else:
        #        vwLength = D[v] + G[v][w]
        
            if w in D:
                if vwLength < D[w]:
                    print("Dijkstra: found better path to already-final vertex")
            elif w not in Q or vwLength < Q[w]:
                Q[w] = vwLength
                P[w] = v
    
    return (D,P)

def shortestPath(G,start,end):
    """
    Find a single shortest path from the given start vertex
    to the given end vertex.
    The input has the same conventions as Dijkstra().
    The output is a list of the vertices in order along
    the shortest path.
    """

    D,P = Dijkstra(G,start,end)
    Path = []
    while 1:
        Path.append(end)
        if end == start: break
        end = P[end]
    Path.reverse()
    return Path

   
def dist_with_prior(weight,u,v):
    
    alpha = 0.15
    beta = 0.25
    delta = 1.85
    beta2 = 3

    M,transformed_model = weight
    prior = max(transformed_model[u],transformed_model[v])
    
    d = np.sqrt((u[0]-v[0])**2+(u[1]-v[1])**2)
    z = 255-min(M[u],M[v])
    
    f_ = d*(alpha*np.exp(beta*z+beta2*prior)+delta)
    #f_ = d*(alpha*np.exp(beta*z)+delta)
    return f_

end_point_flag = (9999,9999)
def build_graph(magnitude, end_points, direction="up", subimage =None,dist_func=dist_with_prior):
        
    G = dict()
    for i in range(subimage[0],subimage[2]):
        for j in range(subimage[1],subimage[3]):
            G[(i,j)] = dict()
            if direction=="up":
                other_vertex = [(i,j-1),(i-1,j-1),(i-1,j),(i-1,j+1),(i,j+1)]
            if direction=="down":
                other_vertex = [(i,j-1),(i+1,j-1),(i+1,j),(i+1,j+1),(i,j+1)]
            if direction=="all":
                other_vertex = [(i,j-1),(i+1,j-1),(i+1,j),(i+1,j+1),(i,j+1),(i-1,j+1),(i-1,j),(i-1,j-1)]
            for v in other_vertex:
                if v[0]>=subimage[0] and v[0]<subimage[2] and v[1]>=subimage[1] and v[1]<subimage[3]:
                    G[(i,j)][(v[0],v[1])] = dist_func(magnitude,(i,j),v)

    for point in end_points:
        G[point][end_point_flag] = 0
    return G
    
    
"""
_______________________________________________________________________________
        SHORTEST PATH FOR GRID-LIKE GRAPHS
_______________________________________________________________________________
"""

def dist_matrix(M):
    """
    Assigns to each pixel a weight based on parameters alpha, beta and delta.
    Weight decreases as the intensity value increases.
    """
    alpha = 0.15
    beta = 0.1
    delta = 1.85
    
    z = 255-M
    f = alpha*np.exp(beta*z)+delta
    return f


# Find the shortest path from all the points in the first/last row to any point in 
#    the last/first row.
def shortest_path_grid(matrix,start="last"):
    SQRT2 = np.sqrt(2)
    if start == "first":
        matrix = np.flip(matrix,axis=0)
    distances = np.zeros(matrix.shape)
    for i in range(1,matrix.shape[0]):
        for j in range(matrix.shape[1]):
            local_dists = np.asarray([np.inf]*3)
            if j>0:
                local_dists[0] = SQRT2*np.mean([matrix[i,j],matrix[i-1,j-1]])+distances[i-1,j-1]
            local_dists[1] = np.mean([matrix[i,j],matrix[i-1,j]])+distances[i-1,j]
            if j<(matrix.shape[1]-1):
                local_dists[2] = SQRT2*np.mean([matrix[i,j],matrix[i-1,j+1]])+distances[i-1,j+1]
            distances[i,j] = min(local_dists)
            
    shortest_paths = []
    for j in range(matrix.shape[1]):
        curr_j = j
        shortest_paths.append([curr_j])
        for i in range(matrix.shape[0]-1,0,-1):
            local_dists = np.asarray([np.inf]*3)
            if curr_j>0:
                local_dists[0] = SQRT2*np.mean([matrix[i,curr_j],matrix[i-1,curr_j-1]])+distances[i-1,curr_j-1]
            local_dists[1] = np.mean([matrix[i,curr_j],matrix[i-1,curr_j]])+distances[i-1,curr_j]
            if curr_j<(matrix.shape[1]-1):
                local_dists[2] = SQRT2*np.mean([matrix[i,curr_j],matrix[i-1,curr_j+1]])+distances[i-1,curr_j+1]
            delta_j = -1 + np.argmin(local_dists)
            curr_j+=delta_j
            shortest_paths[-1].append(curr_j)
        
    if start == "first":
        for l in shortest_paths:
            l.reverse()

    return shortest_paths

# Euclidean distance between two points
def compute_euclidean_distance(a,b):
    return np.sqrt(np.sum((a-b)**2))
    
"""
_______________________________________________________________________________
        OTHER FUNCTIONS
_______________________________________________________________________________
"""
# Given the full list of ground truth points for an image returns only a specific set 
# of points based on the value specified in the parameter "_type":
# left_boundary - The boundary points of the breast on the left
# right_boundary - The boundary points of the breast on the right
# left_extrema - The lateral extrema point of the breast on the left contour
# middle_extrema - The medial extrema point (assumed equal for the two breasts)
# right_extrema - The lateral extrema point of the breast on the right contour
# top_point - Jugular notch
# left_nipple - The nipple on the left of the image
# right_nipple - The nipple on the right of the image
def get_keypoints(ground_truth,_type):
    if _type == "left_boundary":
        return np.flip(ground_truth[0:34].reshape([-1,2]),axis=1)
        #return ground_truth[0:34].reshape([-1,2])
    
    if _type == "right_boundary":
        return np.flip(np.flip(ground_truth[34:68].reshape([-1,2]),axis=0),axis=1)
    
    if _type == "left_extrema":
        return ground_truth[0:2].reshape([-1,2])
    
    if _type == "middle_extrema":
        return ground_truth[32:34].reshape([-1,2])
        
    if _type == "right_extrema":
        return ground_truth[34:36].reshape([-1,2])

    if _type == "top_point":
        return np.flip(ground_truth[68:70].reshape([-1,2]),axis=1)
    
    if _type == "left_nipple":
        return np.flip(ground_truth[70:72].reshape([-1,2]),axis=1)
        
    if _type == "right_nipple":
        return np.flip(ground_truth[72:74].reshape([-1,2]),axis=1)

def normal_prob(points,mean,std):
    alpha = 1/np.sqrt(2*np.pi*std**2)
    beta = -((points-mean)**2/(2*std**2))
    return alpha*np.exp(beta)

def gradient(img):
    gx = skimage.filters.sobel_h(img)
    gy = skimage.filters.sobel_v(img)
    magnitude = np.sqrt(gx**2+gy**2)
    return magnitude

def spline(points,n_points=100):
    t = np.arange(0, 1.0000001, 1/n_points)
    x = points[:,0]
    y = points[:,1]
    tck, u = interpolate.splprep([x, y], s=0)
    out = interpolate.splev(t, tck)
    return out

def circle(center,radius,shape):
    img = Image.new('L', (shape[1], shape[0]), 0)
    coords = (center[1]-radius,center[0]-radius,center[1]+radius,center[0]+radius)
    ImageDraw.Draw(img).ellipse(coords, outline=1, fill=1)
    mask = numpy.array(img)
    return mask

"""

OLD FUNCTIONS




def shortest_path_grid2(matrix, start="last"):
    SQRT2 = np.sqrt(2)
    if start == "first":
        matrix = np.flip(matrix,axis=0)
    distances = np.zeros(matrix.shape)
    for i in range(1,matrix.shape[0]):
        for j in range(matrix.shape[1]):
            local_dists = np.asarray([np.inf]*3)
            if j>0:
                local_dists[0] = matrix[i,j]+distances[i-1,j-1]
            local_dists[1] = matrix[i,j]+distances[i-1,j]
            if j<(matrix.shape[1]-1):
                local_dists[2] = matrix[i,j]+distances[i-1,j+1]
            distances[i,j] = min(local_dists)
    

    shortest_paths = np.zeros(matrix.shape)
    shortest_paths[matrix.shape[0]-1,:] = 1
    for i in range(matrix.shape[0]-1,0,-1):
        for j in range(matrix.shape[1]):
            if shortest_paths[i,j] == 1:
                local_dists = np.asarray([np.inf]*3)
                if j>0:
                    local_dists[0] = distances[i-1,j-1]
                local_dists[1] = distances[i-1,j]
                if j<(matrix.shape[1]-1):
                    local_dists[2] = SQRT2+distances[i-1,j+1]
                delta_j = -1 + np.argmin(local_dists)
                shortest_paths[i-1,j+delta_j] = 1

    if start == "first":
        shortest_paths = np.flip(shortest_paths,axis=0)
        
    return shortest_paths

"""