# This file is part of baseline1 of the VISUM challenge

# Outside Imports
import numpy as np
from shapely.geometry import Polygon,LineString,Point
from matplotlib import pyplot as plt
import scipy.ndimage.morphology as morpho
import cv2

# VISUM Baseline Imports
from utils import dists
import baseline1.config as config


# Preprocess image.
#    args:
#        img - numpy array: image to preprocess
#        points - numpy array: coordinates of points to be adjusted
#        ret_ori_shape - bool: if true the original shape is returned
#        ret_scalling_factor - bool: if true, the scalling factor is returned
#
# Resizes the image so that the horizontal length is equal to the value specified
# in config.py. Point coordinates (points) are optional.
def preprocess_img(img, points=None, ret_ori_shape=False, ret_scalling_factor=False):
    # The scalling factor is computed so that the image.shape[1]==config.image_size
    ori_shape = img.shape
    scalling_factor = config.image_size / ori_shape[1]
    
    # Resize image
    img = cv2.resize(img,dsize =(0,0), fx = scalling_factor,fy = scalling_factor)
    
    # list with variables to be returned
    return_list =[img]
    
    if ret_ori_shape:
        return_list.append(ori_shape)
        
    if ret_scalling_factor:
        return_list.append(scalling_factor)
       
    if  np.all(points) != None:    
        # Points are scalled by the same factor
        points*=scalling_factor
        return_list.append(points)
        
    return tuple(return_list)
    
# Find the extrema points of the breast.
#    args:
#        shape - initial image shape
#        M - image of the gradient magnitude
#        debug_verbose - if true, results of intermediate steps are printed
#
# The algorithm works by first trying to identify the edges of the patient's body, then 
# detecting the lateral extrema point of each breast contour. The medial contour extrema
# point of both breasts is assumed to be the same and computed as the mean of the 
# coordinates of the two lateral extrema points. The jugular notch is computed as the
# half way point between the medial contour extrema point and the top of the image.
#
# The bottom half of the image is transformed into a graph, where each pixel is a vertex 
# connected to adjacent pixels. The weight of each edge in the graph (between adjacent pixels)
# is assigned based on the local gradient magnitude.
# In this graph, strong paths between the top and bottom regions of the image are computed and 
# become candidates for the edges of the patient's body. Two of these paths are then selected
# and grown upwards until a stop condition is met. The point where each path stops is 
# considered to be the lateral extrema point of the corresponding breast. The medial contour
# extrema and jugular notch are then computed.
def find_extrema_points2(shape, M, debug_verbose=True):    
    
    # Compute a distance matrix where we assign a gradient dependent value to each pixel.
    # Only the bottom half of the image is important.
    dist_mat = dists.dist_matrix(M)
    subimage = [shape[0]//2,0,shape[0],shape[1]]
    bottom_dist_mat = dist_mat[subimage[0]:subimage[2],subimage[1]:subimage[3]]
    
    # Computation of the shortest paths between all points in the bottom row and the middle row.
    paths_bottom = dists.shortest_path_grid(bottom_dist_mat,start="last")
    
    # Computation of the shortest paths between all points in the middle row and the bottom row.
    # "dists.shortest_path_grid" returns the paths reversed.
    paths_middle = dists.shortest_path_grid(bottom_dist_mat,start="first")
    
    # Computation of the strong paths between the two regions. 
    # A path is considered strong path between regions "A" and "B" if:
    #    - It is the shortest path between at least one point in "A" and region "B"
    #    - It is the shortest path between at least one point in "B" and region "A"
    final_segments = []
    for segment in paths_bottom:
        if segment in paths_middle:
            final_segments.append(segment)
    
    # Select the segments which will originate the extrema points of the breast
    # The selection is done as follows:
    # - Paths which end near the middle vertical line of the image are discarded
    # - The two most central paths from the ones remaining are selected.
    left_start,right_start = select_start_points(final_segments,shape)
    
    # If debug_verbose then the gradient image is shown along with the selected candidate points
    if debug_verbose:
        plt.clf()
        plt.imshow(M)
        plt.scatter(left_start,M.shape[0]//2,c="r")
        plt.scatter(right_start,M.shape[0]//2,c="r")
        plt.savefig(config.debug_path+"segments_detection.png")
        
    # Grow segments vertically until the last point of the segment is reached
    # (stop condition is met). The path stops growing if, all the last "LENGTH"
    # points have a gradient magnitude smaller than "THRESHOLD".
    # "LENGTH" and "THRESHOLD" are parameters defined by the user.
    pel = grow_segment(M, (M.shape[0]//2,left_start))
    per = grow_segment(M, (M.shape[0]//2,right_start))
    
    # Finally the medial and jugular notch points are located.
    pm = ((pel[0]+per[0])//2,(pel[1]+per[1])//2)
    pt = (pm[0]//2,pm[1])
    
    # If debug_verbose then the gradient image is shown along with the detected points. 
    if debug_verbose:
        plt.clf()
        plt.imshow(M)
        plt.scatter(pel[1],pel[0],c="r")
        plt.scatter(pm[1],pm[0],c="r")
        plt.scatter(per[1],per[0],c="r")
        plt.scatter(pt[1],pt[0],c="r")
        plt.savefig(config.debug_path+"extrema_points.png")
    
    return pel,pm,per,pt

# Detect the breast contour
#    args:
#        M - image of the gradient magnitude
#        pl,pr - initial and end point of the breast contour. pl should be on the left of pr
#        debug_verbose - if true, results of intermediate steps are printed
# First, a weighted graph based on the image gradient is computed. The breast contour is
# computed as the shortest path between start and end points of the breast.
# To avoid contours very similar to straight lines a circle between the two points is created and 
# the weights of these points increased.
def breast_contour(M, pl, pr, debug_verbose=True):
    
    # The circle between the two points is created.
    center = (np.asarray(pl)+np.asarray(pr))/2
    radius = dists.compute_euclidean_distance(np.asarray(pl),np.asarray(pr))/2
    shape_prior = dists.circle(center,radius,M.shape)
    
    # If debug_verbose then the gradient image is shown along with the previously defined circle
    if debug_verbose:
        plt.clf()
        plt.imshow(M)
        circle1 = plt.Circle(center[::-1], radius, color='r')
        ax = plt.gca()
        ax.add_artist(circle1)
        plt.savefig(config.debug_path+"circle_shown.png")
    
    # Values on which the weights of graph edges will be based on.
    # Gradient magnitude image "M" and a shape prior (circle)
    weight = [M,shape_prior]
     
    # Coordinates for the creation of the graph
    # To reduce the time needed to process each breast contour only a rectangle where the
    # contour is expected to be is used.
    shape = M.shape
    maximum_breast_length = 600
    maximum_breast_to_side = 100
    top_limit = min(pl[0], pr[0])
    limits =    [top_limit,                                       # top
                 max(pl[1]-maximum_breast_to_side, 0),            # left
                 min(top_limit+maximum_breast_length, shape[0]),  # bottom
                 min(pr[1]+maximum_breast_to_side, shape[1])      # right
                ]
    
    # The graph is built using the limits proposed
    G = dists.build_graph(weight, {pr}, direction="all",subimage=limits,dist_func=dists.dist_with_prior)
    
    # Find the shortest path
    boundary = dists.shortestPath(G, pl, dists.end_point_flag)
    boundary = np.asarray(boundary)
    
    # Remove the last point of the boundary. In this implementation of dists.build_graph 
    # the last point is a flag (9999,9999) so it should be removed.
    boundary = boundary[0:-1,:]
    
    return boundary
    
# Detects the nipple in the image.
#    args:
#        img - patients image
#        boundary - boundary of the breast
#        nipple_params - nipple model computed during training.
#        debug_verbose - if true, results of intermediate steps are printed
# First we assign to all pixels of the image the probability of being a nipple location 
# We then select the maximum probability point as the nipple
def nipple(img, boundary, nipple_params, debug_verbose=True):
    
    # Parameters computed during training:
    means = nipple_params[0,:]
    stds = nipple_params[1,:]
    
    #Compute the breast mask
    breast_mask = get_breast_mask([*img.shape[0:2]],boundary)
    
    # Compute the probability image based on the angle
    x,y = np.nonzero(breast_mask)
    mid_point = (boundary[0]+boundary[-1])/2
    angle_image = np.zeros(breast_mask.shape)
    for i in range(x.shape[0]):
        vec = (x[i],y[i]) - mid_point
        angle = np.arctan2(vec[0], vec[1])
        
        coords = (x[i].astype(int), y[i].astype(int))
        angle_image[coords[0], coords[1]] = angle
    angle_prob = dists.normal_prob(angle_image,means[0],stds[0])
        
    # Compute the probability image based on distance
    distance_image = morpho.distance_transform_edt(breast_mask)
    dist_prob = dists.normal_prob(distance_image,means[1],stds[1])
    
    # Compute the probability image based on color
    mean_color = np.asarray( [np.average(img[:,:,0],weights=breast_mask),
                              np.average(img[:,:,1],weights=breast_mask),
                              np.average(img[:,:,2],weights=breast_mask)
                             ])
    color_image = img-mean_color
    red_prob = dists.normal_prob(color_image[:,:,0],means[2],stds[2])
    blue_prob = dists.normal_prob(color_image[:,:,1],means[3],stds[3])
    green_prob = dists.normal_prob(color_image[:,:,2],means[4],stds[4])
    color_prob = (red_prob+blue_prob+green_prob)/3
    
    # Compute the final probability:
    prob = angle_prob*dist_prob*color_prob*breast_mask
    
    # Find the nipple position as the maximum of the probability image
    nip = np.unravel_index(np.argmax(prob),prob.shape)
    
    if debug_verbose:
        plt.clf()
        plt.imshow(breast_mask)
        plt.savefig(config.debug_path+"breast_mask.png")
        plt.imshow(angle_image)
        plt.savefig(config.debug_path+"angle_image.png")
        plt.imshow(distance_image)
        plt.savefig(config.debug_path+"distance_image.png")
        plt.imshow(color_image)
        plt.savefig(config.debug_path+"color_image.png")

    if debug_verbose:
        plt.clf()
        plt.imshow(prob)
        plt.savefig(config.debug_path+"prob.png")
        plt.imshow(angle_prob)
        plt.savefig(config.debug_path+"angle_prob.png")
        plt.imshow(dist_prob)
        plt.savefig(config.debug_path+"dist_prob.png")
        plt.imshow(color_prob)
        plt.savefig(config.debug_path+"color_prob.png")
    
    return nip

# Function used to obtain the mask of the breast given the boundary points
#    args:
#        shape - image shape
#        breast - points of the breast boundary
#        debug_verbose - if true, results of intermediate steps are printed
# 1. The points in the contour are set to 1
# 2. The points in the line between the two extrema are set to 1
# 3. The points in the black hole created by the first two operations are set to 1
def get_breast_mask(shape, breast, debug_verbose=True):
    # Fill all contour points
    mask = np.zeros(shape)
    x,y = dists.spline(breast,n_points=8000)
    x = np.clip(np.round(x).astype(int),0,mask.shape[0]-1)
    y = np.clip(np.round(y).astype(int),0,mask.shape[1]-1)
    mask[x,y] = 1
                
    # Fill all points at the top
    x = np.clip(np.linspace(breast[0,0],breast[-1,0],2000),0,mask.shape[0]-1)
    y = np.clip(np.linspace(breast[0,1],breast[-1,1],2000),0,mask.shape[1]-1)
    x = np.round(x).astype(int)
    y = np.round(y).astype(int)
    mask[x,y] = 1
                
    # Fill black holes of the mask
    mask_full = morpho.binary_fill_holes(mask)
    
    if debug_verbose:
        plt.clf()
        plt.imshow(mask_full)
        plt.savefig(config.debug_path+"mask.png")
        
    return mask_full


# Given a list of strong paths this function selects one point in each side of 
# the patient's body boundary near the breast lateral extrema points:
#    args:
#        final_segments - strong paths between the mid and bottom regions of the image
#        shape - image shape
# 1. final paths which the top point is near the center of the image are discarded.
# 2. from the remaining paths the two most central are selected. 
def select_start_points(final_segments,shape):
    
    valid_L_js = []
    valid_R_js = []
    
    L_distances = []
    R_distances = []
    
    img_len = shape[1]
    half_img_len = img_len/2
    
    for segment in final_segments:
        j = segment[-1]
        if (np.abs(j-half_img_len)/img_len)>0.2:
            
            dist = (np.abs(j-half_img_len)/img_len)
            
            if (j-half_img_len)<0:
                valid_L_js.append(j)
                L_distances.append(dist)
            else:
                valid_R_js.append(j)
                R_distances.append(dist)

    left = [x for _, x in sorted(zip(L_distances, valid_L_js))]
    right = [x for _, x in sorted(zip(R_distances, valid_R_js))]

    return left[0], right[0]
    
# Given a starting point and a weighted image this funtion travels upwards in
# the direction of maximum intensity until the end conditions are met.
# In this case, the process stops when a sequence of pixels with intensity 
# smaller than threshold and length equal to distance has been travelled.
def grow_segment(M, point):
    threshold = 20
    distance = 12
    
    travelling = True
    count = 0
    while travelling:
        max_gradient_dir = -1 + np.argmax(M[point[0]-1,point[1]-1:point[1]+2])
        point = (point[0]-1,point[1]+max_gradient_dir) 
        
        count = count+1 if M[point]<threshold else 0        
        travelling = count<distance
    
    return point
