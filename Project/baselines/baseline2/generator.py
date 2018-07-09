import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.transform import rotate as rotate_

def translate_points(point,translation): 
    point = point + translation 
    
    return point

def rotate_points(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy



class Generator(object):
    def __init__(self,
                 X_train,
                 Mask_train,
                 Y_train,
                 batchsize=2,
                 flip_ratio=0.1,
                 translation_ratio=0.1,
                 rotate_ratio=0.1,
                 contrast_ratio=0.1,
                 flip_indices=[(0,34),(1,35),(2,36),(3,37),(4,38),(5,39),(6,40),(7,41),(8,42),(9,43),(10,44),(11,45),(12,46),(13,47),(14,48),(15,49),(16,50),(17,51)
                ,(18,52),(19,53),(20,54),(21,55),(22,56),(23,57),(24,58),(25,59),(26,60),(27,61),(28,62),(29,63),(30,64),(31,65)
                ,(32,66),(33,67),(68,68),(69,69),(70,72),(71,73)]
                 ):
        """
        Arguments
        ---------
        """
        self.X_train = X_train
        self.Mask_train = Mask_train
        self.Y_train = Y_train
        self.size_train = X_train.shape[0]
        self.batchsize = batchsize
        self.flip_ratio = flip_ratio
        self.translation_ratio = translation_ratio
        self.rotate_ratio = rotate_ratio
        self.contrast_ratio = contrast_ratio
        self.flip_indices = flip_indices

    

    def _random_indices(self, ratio):
        """Generate random unique indices according to ratio"""
        size = int(self.actual_batchsize * ratio)
        return np.random.choice(self.actual_batchsize, size, replace=False)
    
    def flip(self):
        """Flip image batch"""
        indices = self._random_indices(self.flip_ratio)
        self.inputs[indices] = self.inputs[indices,:,::-1,:]
        self.mask[indices] = self.mask[indices, :, ::-1]
        self.targets[indices, ::2] = 1 - self.targets[indices, ::2]
        for a, b in self.flip_indices:
            self.targets[indices, a], self.targets[indices, b] = (self.targets[indices, b], self.targets[indices, a])      
            
            
    def translation(self): 
        """Translation"""
        x_t = [] 
        y_t = [] 
        
        indices = self._random_indices(self.translation_ratio)
        tx = np.random.randint(-50, 50)
        ty = np.random.randint(-30, 30)
        
        x_t = self.targets[indices, ::2]
        y_t = self.targets[indices, 1::2]
        
        for i in range(np.shape(x_t)[0]):
            x_t[i] = translate_points(x_t[i],tx/512)
            y_t[i] = translate_points(y_t[i],ty/512)
            
            
        self.targets[indices, ::2] = x_t
        self.targets[indices, 1::2] = y_t 
        
        
        mask = self.mask[indices,:,:]
        image = self.inputs[indices,:,:,:]
        
        
        for i in range(np.shape(mask)[0]): 
             mask[i,:,:,0] = cv2.warpAffine(mask[i,:,:,0],np.float32([[1,0,tx],[0,1,ty]]),(512,384))
             
             
        mask = np.reshape(mask,(-1,384,512))
        
             
        for i in range(np.shape(image)[0]):
            for j in range(3):
                image[i,:,:,j] = cv2.warpAffine(image[i,:,:,j],np.float32([[1,0,tx],[0,1,ty]]),(512,384))
            
        
        mask = np.reshape(mask,(-1,384,512,1))

        self.mask[indices] = mask[:]
        self.inputs[indices,:,:,:] = image[:,:,:,:]


    def rotate(self):
        """Rotate slighly the image and the targets."""
        indices = self._random_indices(self.rotate_ratio)
        angle = np.random.randint(-10, 10)

        M = cv2.getRotationMatrix2D((512/2,384/2),angle,1)
        for i in indices: 
            for j in range(3): 
                self.inputs[i,:,:,j] = cv2.warpAffine(self.inputs[i,:,:,j],M,(512,384))
            self.mask[i,:,:,0] = cv2.warpAffine(self.mask[i,:,:,0],M,(512,384))
            
        x_r = []
        y_r = [] 
               
        for i in range(np.shape(self.mask)[0]): 
            x_r.append(self.targets[i][0:74:2])
            y_r.append(self.targets[i][1:75:2])
            x_r[i], y_r[i] = rotate_points((256/512,192/512),(x_r[i],y_r[i]),(-angle * 2 * np.pi)/360)
            self.targets[i][0:74:2] = x_r[i] 
            self.targets[i][1:75:2] = y_r[i]
            

    def generate(self, batchsize=32): 
        """Generator"""
        while True:
            cuts = [(b, min(b + self.batchsize, self.size_train)) for b in range(0, self.size_train, self.batchsize)]
            for start, end in cuts:
                self.inputs = self.X_train[start:end].copy()
                self.mask = self.Mask_train[start:end].copy()
                self.targets = self.Y_train[start:end].copy()
                self.actual_batchsize = self.inputs.shape[0]  # Need this to avoid indices out of bounds
                self.flip()
                self.translation()

                yield (self.inputs, {'heatmaps': self.mask, 'keypoints': self.targets})
    