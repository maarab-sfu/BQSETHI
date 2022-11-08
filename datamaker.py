from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


## Imports
import numpy as np
import pywt
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.io as spio
from sklearn import preprocessing
from scipy.misc import imresize
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from tensorflow.contrib.learn.python.learn import monitors as monitor_lib

def dataMakerIN():
    mat = spio.loadmat('datasets/IN/Indian_pines_corrected.mat', squeeze_me=True)                                                                         
    All_features_ = mat['indian_pines_corrected']

##    cA = cH44 = cV44 = cD44 = np.zeros((10, 10, 200))
##    cH43 = cV43 = cD43 = np.zeros((19, 19, 200))
##    cH42 = cV42 = cD42 = np.zeros((37, 37, 200))
##    cH41 = cV41 = cD41 = np.zeros((73, 73, 200))
    for i in range(200):
        ( cA,
         (cH44, cV44, cD44),
         (cH43, cV43, cD43),
         (cH42, cV42, cD42),
         (cH41, cV41, cD41)) = pywt.wavedec2(All_features_[:,:,i], 'db1', level = 4)
        
        cA = cA.astype(int)
        
        cH44 = cH44.astype(int)
        cH43 = cH43.astype(int)
        cH42 = cH42.astype(int)
        cH41 = cH41.astype(int)

        cV44 = cV44.astype(int)
        cV43 = cV43.astype(int)
        cV42 = cV42.astype(int)
        cV41 = cV41.astype(int)

        cD44 = cD44.astype(int)
        cD43 = cD43.astype(int)
        cD42 = cD42.astype(int)
        cD41 = cD41.astype(int)

        
        spio.savemat('data/IN/allLabels/cA' + str(i) + '.mat',{"cA":cA})
        
        spio.savemat('data/IN/allLabels/cH44' + str(i) + '.mat',{"cH44":cH44})
        spio.savemat('data/IN/allLabels/cV44' + str(i) + '.mat',{"cV44":cV44})
        spio.savemat('data/IN/allLabels/cD44' + str(i) + '.mat',{"cD44":cD44})

        spio.savemat('data/IN/allLabels/cH43' + str(i) + '.mat',{"cH43":cH43})
        spio.savemat('data/IN/allLabels/cV43' + str(i) + '.mat',{"cV43":cV43})
        spio.savemat('data/IN/allLabels/cD43' + str(i) + '.mat',{"cD43":cD43})
        
        spio.savemat('data/IN/allLabels/cH42' + str(i) + '.mat',{"cH42":cH42})
        spio.savemat('data/IN/allLabels/cV42' + str(i) + '.mat',{"cV42":cV42})
        spio.savemat('data/IN/allLabels/cD42' + str(i) + '.mat',{"cD42":cD42})

        spio.savemat('data/IN/allLabels/cH41' + str(i) + '.mat',{"cH41":cH41})
        spio.savemat('data/IN/allLabels/cV41' + str(i) + '.mat',{"cV41":cV41})
        spio.savemat('data/IN/allLabels/cD41' + str(i) + '.mat',{"cD41":cD41})
    
def dataMakerUP():        
    mat = spio.loadmat('datasets/UP/PaviaU.mat', squeeze_me=True)                                                                         
    All_features_ = mat['paviaU']

##    cA = cH44 = cV44 = cD44 = np.zeros((39, 22, 103))
##    cH43 = cV43 = cD43 = np.zeros((77, 43, 103))
##    cH42 = cV42 = cD42 = np.zeros((153, 85, 103))
##    cH41 = cV41 = cD41 = np.zeros((305, 170, 103))
    
    for i in range(103):
        ( cA,
         (cH44, cV44, cD44),
         (cH43, cV43, cD43),
         (cH42, cV42, cD42),
         (cH41, cV41, cD41)) = pywt.wavedec2(All_features_[:,:,i], 'db1', level = 4)
        
        cA = cA.astype(int)
        cH44 = cH44.astype(int)
        cH43 = cH43.astype(int)
        cH42 = cH42.astype(int)
        cH41 = cH41.astype(int)

        cV44 = cV44.astype(int)
        cV43 = cV43.astype(int)
        cV42 = cV42.astype(int)
        cV41 = cV41.astype(int)

        cD44 = cD44.astype(int)
        cD43 = cD43.astype(int)
        cD42 = cD42.astype(int)
        cD41 = cD41.astype(int)

        
        spio.savemat('data/UP/allLabels/cA' + str(i) + '.mat',{"cA":cA})
        
        spio.savemat('data/UP/allLabels/cH44' + str(i) + '.mat',{"cH44":cH44})
        spio.savemat('data/UP/allLabels/cV44' + str(i) + '.mat',{"cV44":cV44})
        spio.savemat('data/UP/allLabels/cD44' + str(i) + '.mat',{"cD44":cD44})

        spio.savemat('data/UP/allLabels/cH43' + str(i) + '.mat',{"cH43":cH43})
        spio.savemat('data/UP/allLabels/cV43' + str(i) + '.mat',{"cV43":cV43})
        spio.savemat('data/UP/allLabels/cD43' + str(i) + '.mat',{"cD43":cD43})
        
        spio.savemat('data/UP/allLabels/cH42' + str(i) + '.mat',{"cH42":cH42})
        spio.savemat('data/UP/allLabels/cV42' + str(i) + '.mat',{"cV42":cV42})
        spio.savemat('data/UP/allLabels/cD42' + str(i) + '.mat',{"cD42":cD42})

        spio.savemat('data/UP/allLabels/cH41' + str(i) + '.mat',{"cH41":cH41})
        spio.savemat('data/UP/allLabels/cV41' + str(i) + '.mat',{"cV41":cV41})
        spio.savemat('data/UP/allLabels/cD41' + str(i) + '.mat',{"cD41":cD41})
    
def dataMakerKSC():        
    mat = spio.loadmat('datasets/KSC/KSC.mat', squeeze_me=True)                                                                         
    All_features_ = mat['KSC']
##    cA = cH44 = cV44 = cD44 = np.zeros((32, 39, 176))
##    cH43 = cV43 = cD43 = np.zeros((64, 77, 176))
##    cH42 = cV42 = cD42 = np.zeros((128, 154, 176))
##    cH41 = cV41 = cD41 = np.zeros((256, 307, 176))
    for i in range(176):
        ( cA,
         (cH44, cV44, cD44),
         (cH43, cV43, cD43),
         (cH42, cV42, cD42),
         (cH41, cV41, cD41)) = pywt.wavedec2(All_features_[:,:,i], 'db1', level = 4)



        cA = cA.astype(int)
        
        cH44 = cH44.astype(int)
        cH43 = cH43.astype(int)
        cH42 = cH42.astype(int)
        cH41 = cH41.astype(int)

        cV44 = cV44.astype(int)
        cV43 = cV43.astype(int)
        cV42 = cV42.astype(int)
        cV41 = cV41.astype(int)

        cD44 = cD44.astype(int)
        cD43 = cD43.astype(int)
        cD42 = cD42.astype(int)
        cD41 = cD41.astype(int)

        
        spio.savemat('data/KSC/allLabels/cA' + str(i) + '.mat',{"cA":cA})
        
        spio.savemat('data/KSC/allLabels/cH44' + str(i) + '.mat',{"cH44":cH44})
        spio.savemat('data/KSC/allLabels/cV44' + str(i) + '.mat',{"cV44":cV44})
        spio.savemat('data/KSC/allLabels/cD44' + str(i) + '.mat',{"cD44":cD44})

        spio.savemat('data/KSC/allLabels/cH43' + str(i) + '.mat',{"cH43":cH43})
        spio.savemat('data/KSC/allLabels/cV43' + str(i) + '.mat',{"cV43":cV43})
        spio.savemat('data/KSC/allLabels/cD43' + str(i) + '.mat',{"cD43":cD43})
        
        spio.savemat('data/KSC/allLabels/cH42' + str(i) + '.mat',{"cH42":cH42})
        spio.savemat('data/KSC/allLabels/cV42' + str(i) + '.mat',{"cV42":cV42})
        spio.savemat('data/KSC/allLabels/cD42' + str(i) + '.mat',{"cD42":cD42})

        spio.savemat('data/KSC/allLabels/cH41' + str(i) + '.mat',{"cH41":cH41})
        spio.savemat('data/KSC/allLabels/cV41' + str(i) + '.mat',{"cV41":cV41})
        spio.savemat('data/KSC/allLabels/cD41' + str(i) + '.mat',{"cD41":cD41})

def dataMaker():
    dataMakerIN()
    dataMakerUP()
    dataMakerKSC()
    
def main(unused_argv):
    dataMaker()

if __name__ == "__main__":
  tf.app.run()
