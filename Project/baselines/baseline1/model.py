import numpy as np
from utils import dists, scoring
import process_image as proc
from matplotlib import pyplot as plt
import pickle
import config
import time

models_loaded = False
left_nipple_params = None
right_nipple_params = None
mean_model = None

# Load the models created during trainining
def load_models():
    global left_nipple_params,right_nipple_params,mean_model
    left_nipple_params = np.load("models/left_nipple_params.npy")
    right_nipple_params = np.load("models/right_nipple_params.npy")
    mean_model = np.load("models/mean_model.npy")


def test(img, testing=False, ground_truth=None, debug_verbose=False, suffix = "",time_debug=True):
    
    if not models_loaded:
        load_models()
    
    start_time = time.time()
    times = [start_time]
    
    # Preprocess_image
    img, ori_shape, scalling_factor = proc.preprocess_img(img,
                                ret_ori_shape=True, ret_scalling_factor=True)
    
    # Compute gradient magnitude
    grey = np.average(img,axis=2)
    M = dists.gradient(grey)
    times.append(time.time())
    # Find breast extrema points
    try:
        pl,pm,pr,pt = proc.find_extrema_points2(img.shape,M)
        times.append(time.time())
        # Find contour of each breast
        l_boundary = proc.breast_contour(M,pl,pm)
        r_boundary = proc.breast_contour(M,pm,pr)
        times.append(time.time())
        # Find contour of each nipple
        l_nipple = proc.nipple(img, l_boundary, left_nipple_params)
        r_nipple = proc.nipple(img, r_boundary, right_nipple_params)
        times.append(time.time())
        # Save detections    
        if debug_verbose:
            plt.clf()
            plt.imshow(img)
            plt.scatter(pl[1], pl[0], c='r')
            plt.scatter(pm[1], pm[0], c='r')
            plt.scatter(pr[1], pr[0], c='r')
            plt.scatter(pt[1], pt[0], c='r')
            plt.scatter(l_nipple[1], l_nipple[0], c='r')
            plt.scatter(r_nipple[1], r_nipple[0], c='r')

            plt.plot(l_boundary[:,1], l_boundary[:,0], c='r')
            plt.plot(r_boundary[:,1], r_boundary[:,0], c='r')

            plt.show()

        detections = points_to_detections(l_boundary,r_boundary,l_nipple,r_nipple,pt,scalling_factor)

        if time_debug:
            print("debug times: ","{:.1f}".format(times[1]-times[0]),"{:.1f}".format(times[2]-times[1]),
                                  "{:.1f}".format(times[3]-times[2]),"{:.1f}".format(times[4]-times[3]))
    except Exception as e:
        
        detections = mean_model/scalling_factor
        print("Detection failed!")
        if not testing:
            raise e
    
    
    # If a ground truth is given this function also computes a score
    if np.all(ground_truth) != None:
        scores = scoring.measure_distances(detections, ground_truth, ori_shape)
        return detections, scores
        
    print("\tFinished:", suffix, "took", time.time()-start_time, "s")
    return detections

def points_to_detections(l_boundary,r_boundary,nippleL,nippleR,jugular_notch,scalling_factor):
    
    l_breast_dets=[]
    index = np.round(np.linspace(0,len(l_boundary)-1,17)).astype(int)
    l_breast_dets = l_boundary[index]
    l_breast_dets = np.flip(np.asarray(l_breast_dets),axis=1)
    l_breast_dets = np.reshape(l_breast_dets,[-1])/scalling_factor
    
    r_breast_dets=[]
    index = np.round(np.linspace(0,len(r_boundary)-1,17)).astype(int)
    r_breast_dets = r_boundary[index]
    r_breast_dets = np.flip(np.asarray(r_breast_dets),axis=1)
    r_breast_dets = np.reshape(r_breast_dets,[-1])/scalling_factor
    
    nippleLdet = np.asarray(nippleL[::-1])/scalling_factor
    nippleRdet = np.asarray(nippleR[::-1])/scalling_factor
    jugular_notch_det = np.asarray(jugular_notch[::-1])/scalling_factor

    detection = np.concatenate([l_breast_dets,r_breast_dets,jugular_notch_det,nippleLdet,nippleRdet])
    return detection
