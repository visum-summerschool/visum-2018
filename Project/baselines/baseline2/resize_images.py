import numpy as np 
from skimage import transform 
from scipy import misc

def resize(images): 
    X = []
    orig_shape = []
    for i in range(images.shape[0]):
        rows, columns, channels = images[i].shape
        orig_shape.append([rows, columns, channels])
        x1 = rows/1536
        x2 = columns/2048
        aux = np.array(images[i])
        aux = transform.resize(aux, (aux.shape[0] / x1, aux.shape[1] / x2))
        aux = misc.imresize(aux,25)
        aux = np.reshape(aux,(384,512,3))
        X.append(aux)

    return X, orig_shape


def resize_keypoints_to_original_size(keypoint_predictions, X_original):
    X_original = np.array(X_original)
    keypoint_predictions = np.array(keypoint_predictions)
    
    final_predictions = [] 

    for i in range(X_original.shape[0]):
        rows, columns, channels = np.shape(X_original[i])
        x1 = rows / 1536
        x2 = columns / 2048
        for j in range(74): 
            if(j % 2 == 0):
                keypoint_predictions[i][j] *= x2
            else: 
                keypoint_predictions[i][j] *= x1
        final_predictions.append(keypoint_predictions[i] * 4)
        
    return final_predictions