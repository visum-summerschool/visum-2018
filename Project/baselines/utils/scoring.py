from shapely.geometry import LineString, Point
import numpy as np
import scipy.interpolate as interpolate
import pickle as pkl
import sys
import os



def generate_scores(predictions, ground_truth, imgs_shapes):
    """
    Receives three lists predictions, ground_truths and imgs_shapes and 
    computes the score for each entry. Each index on the lists should be of the
    same image.
    """
    assert len(predictions) == len(ground_truth)
    assert len(predictions) == len(imgs_shapes)
    
    all_distances = []
    for i in range(len(predictions)):
        ori_shape = imgs_shapes[i]
        dets = predictions[i]
        grth = ground_truth[i]
        distances = measure_distances(dets, grth, ori_shape)
        all_distances.append(distances)
    return all_distances
    
def measure_distances(detections, keypoints, ori_shape):
    """
    Receives a list with the detected points, the ground_truth points and the 
    the original shape and computes the scores for that image.
    """
    
    # diagonal line length (used to normalize the error of each point)
    diagonal_shape = compute_euclidean_distance(np.zeros([2]),
                                                         [*ori_shape[0:2]])
        
    # compute the error for each point
    jugular_notch_error = compute_euclidean_distance(keypoints[68:70],
                                                detections[68:70])
    l_nipple_error = compute_euclidean_distance(keypoints[70:72],
                                                detections[70:72])
    r_nipple_error = compute_euclidean_distance(keypoints[72:74],
                                                detections[72:74])
    
    
    # compute the error for each boundary
    l_breast_error = get_curves_distance(keypoints[0:34],detections[0:34],
                                         n_points=diagonal_shape)
    r_breast_error = get_curves_distance(keypoints[34:68],detections[34:68],
                                         n_points=diagonal_shape)
    
    # normalize all errors
    l_nipple_error /= diagonal_shape
    r_nipple_error /= diagonal_shape
    jugular_notch_error /= diagonal_shape
    l_breast_error /= diagonal_shape
    r_breast_error /= diagonal_shape
    
    breast_error = (l_breast_error+r_breast_error)/2
    nipple_error = (l_nipple_error+r_nipple_error)/2
    
    # Individual scores are given for each task
    score = [breast_error, nipple_error, jugular_notch_error]
    return score 
    
def compute_euclidean_distance(a,b):
    """
    Euclidean distance between two points
    """
    return np.sqrt(np.sum((a-b)**2))

def get_curves_distance(points_a,points_b,n_points):
    """
    Distance between two curves. The curve is obtained by spline interpolation 
    of the given points. The distance between curve_a and curve_b is the mean 
    of the distance between 17 points on curve_a and the closest point of each 
    in curve_b. This value is computed in the two directions and the mean 
    returned
    """
    points_a = points_a.reshape([-1,2])
    points_b = points_b.reshape([-1,2])
    
    distance = curve_dist_aux(points_a,points_b,n_points)
    distance+= curve_dist_aux(points_b,points_a,n_points)
    distance/=2
    return distance  

def curve_dist_aux(curve_points,points,n_points):
    curve = spline(curve_points,n_points)
    curve = np.stack(curve,axis=1)
    curve = LineString(curve)
    distance = 0
    for point in points:
        distance+=curve.distance(Point(point))
    distance/=len(points)
    return distance
    
def spline(points,n_points=100):
    t = np.arange(0, 1.0000001, 1/n_points)
    x = points[:,0]
    y = points[:,1]
    tck, u = interpolate.splprep([x, y], s=0)
    out = interpolate.splev(t, tck)
    return out