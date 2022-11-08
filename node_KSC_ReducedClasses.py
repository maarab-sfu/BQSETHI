from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


## Imports
import numpy as np

import tensorflow as tf
import logging
##logging.getLogger("tensorflow").setLevel(logging.ERROR)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.io as spio
import math
##import pickle
##from sklearn import preprocessing
##from scipy.misc import imresize
##from sklearn.decomposition import PCA
##from sklearn.model_selection import train_test_split
##from tensorflow.contrib.learn.python.learn import monitors as monitor_lib
from enum import Enum

tf.logging.set_verbosity(tf.logging.INFO)

def convbnrelu3d(inputs, filters, kernel_size, strides, padding, initializer, regularizer, mode):
    convolution = tf.layers.conv3d(
        inputs = inputs,
        filters = filters,
        kernel_size = kernel_size,
        strides = strides,
        padding = padding,
        kernel_initializer=initializer,
        kernel_regularizer=regularizer,
        activation=None)
##    bn = tf.layers.batch_normalization(convolution, axis=-1, center=True, scale=True, training=(mode == tf.estimator.ModeKeys.TRAIN))
    return tf.nn.relu(convolution)

def convbnrelu2d(inputs, filters, kernel_size, padding, initializer, regularizer, mode):
    convolution = tf.layers.conv2d(
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        padding=padding,
        kernel_initializer=initializer,
        kernel_regularizer=regularizer,
        activation=None)
##    bn = tf.layers.batch_normalization(convolution, axis=-1, center=True, scale=True, training=(mode == tf.estimator.ModeKeys.TRAIN))
    return tf.nn.relu(convolution)


def convsumbnrelu3d(inputs, relu, filters, kernel_size, strides, padding, initializer, regularizer, mode):
    convolution = tf.layers.conv3d(
        inputs = inputs,
        filters = filters,
        kernel_size = kernel_size,
        strides = strides,
        padding = padding,
        kernel_initializer=initializer,
        kernel_regularizer=regularizer,
        activation=None)
    summation = relu + convolution
##    bn = tf.layers.batch_normalization(summation, axis=-1, center=True, scale=True, training=(mode == tf.estimator.ModeKeys.TRAIN))
    return (tf.nn.relu(convolution), summation)

def convsumbnrelu2d(inputs, relu, filters, kernel_size, padding, initializer, regularizer, mode):
    convolution = tf.layers.conv2d(
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        padding=padding,
        kernel_initializer=initializer,
        kernel_regularizer=regularizer,
        activation=None)

    summation = convolution + relu
##    bn = tf.layers.batch_normalization(summation, axis=-1, center=True, scale=True, training=(mode == tf.estimator.ModeKeys.TRAIN))
    return (tf.nn.relu(convolution), summation)

def SSRN_model_fn(features, labels, mode):

    regularizer = tf.contrib.layers.l2_regularizer(scale = 0.0000)
    initializer = tf.contrib.layers.xavier_initializer(uniform = False) #tf.random_normal_initializer(0, 0.1) #glorot_normal_initializer()

    input_layer_spectral = tf.transpose(features["x"], perm= [0, 3, 1, 2])
    input_layer_spectral = tf.reshape(input_layer_spectral, [-1, 176, 7, 7, 1])
    input_layer_spectral = tf.cast(input_layer_spectral, tf.float32)
    conv1 = convbnrelu3d(input_layer_spectral, 16, [7, 1, 1], (2, 1, 1), "valid", initializer, regularizer, mode)
    conv2 = convbnrelu3d(conv1, 16, [7, 1, 1], (1, 1, 1), "same", initializer, regularizer, mode)
    (conv3, summation1) = convsumbnrelu3d(conv2, conv1, 16, [7, 1, 1], (1, 1, 1), "same", initializer, regularizer, mode)
    conv4 = convbnrelu3d(conv3, 16, [7, 1, 1], (1, 1, 1), "same", initializer, regularizer, mode)
    (conv5, summation2) = convsumbnrelu3d(conv4, summation1, 16, [7, 1, 1], (1, 1, 1), "same", initializer, regularizer, mode)
    bn5 = tf.layers.batch_normalization(conv5, axis = -1, center = True, scale = True, training = (mode == tf.estimator.ModeKeys.TRAIN))
    relu5 = tf.nn.relu(bn5)

    conv6 = convbnrelu3d(relu5, 128, [85, 1, 1], (1, 1, 1), "valid", initializer, regularizer, mode)

    ## start of spatial block
    input_layer_spatial = tf.reshape(conv6, [-1, 7, 7, 128])
    conv7 = convbnrelu2d(input_layer_spatial, 16, [3, 3], "valid", initializer, regularizer, mode)
    conv8 = convbnrelu2d(conv7, 16, [3, 3], "same", initializer, regularizer, mode)
    (conv9, summation3) = convsumbnrelu2d(conv8, conv7, 16, [3, 3], "same", initializer, regularizer, mode)
    conv10 = convbnrelu2d(conv9, 16, [3, 3], "same", initializer, regularizer, mode)
    (conv11, summation4) = convsumbnrelu2d(conv10, summation3, 16, [3, 3], "same", initializer, regularizer, mode)

    bn12 = tf.layers.batch_normalization(conv11, axis = -1, center = True, scale = True, training = (mode == tf.estimator.ModeKeys.TRAIN))
    relu12 = tf.nn.relu(bn12)

    pool1 = tf.layers.average_pooling2d(inputs = relu12, pool_size = [5, 5], strides = 1)
    pool1_flat = tf.reshape(pool1, [-1, 16]) #flatten

    dropout = tf.layers.dropout(inputs = pool1_flat, rate = 0.5, training = (mode == tf.estimator.ModeKeys.TRAIN))

    logits = tf.layers.dense(inputs = dropout, units = 16, activation = None, kernel_initializer = initializer)

    ## End of network design

    predictions = {
        ## Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input = logits, axis = 1),
        ## Add `softmax_tensor` to the graph. It is used for PREDICT and by the`logging_hook`.
        "probabilities": tf.nn.softmax(logits, name = "softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode = mode, predictions = predictions)

    ## Calculate Loss (for both TRAIN and EVAL modes)
    onehot_labels = tf.one_hot(indices = tf.cast(labels, tf.int32), depth = 16)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels = onehot_labels, logits = logits, reduction = tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)+ tf.losses.get_regularization_loss()
    #loss = tf.identity(loss, name="loss")


    accuracy_train, update_op1 = tf.metrics.accuracy(labels = labels, predictions = predictions["classes"], name = "my_metric")
    #accuracy_train = tf.identity(accuracy_train, name="accuracy_train")

    ## Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(labels = labels, predictions = predictions["classes"])
    }

    ## Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:

	    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	    running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope = "my_metric")
	    running_vars_initializer = tf.variables_initializer(var_list = running_vars)
		

	    Accuracy_train = tf.summary.scalar('accuracy_train', update_op1)
	    Loss_train = tf.summary.scalar('loss_train', loss)

	    summary_hook = tf.train.SummarySaverHook(
		    save_steps  =128,
		    output_dir = "KSC_model/summary/train",
		    summary_op = [Loss_train, Accuracy_train])

	    logging_hook = tf.train.LoggingTensorHook({"reset":running_vars_initializer, "loss_train": loss,  "accuracy_train":update_op1}, every_n_iter = 128)


	    optimizer = tf.train.RMSPropOptimizer(learning_rate = 0.0001)
	    with tf.control_dependencies(update_ops):
	        train_op = optimizer.minimize(loss = loss, global_step = tf.train.get_global_step())
	    return tf.estimator.EstimatorSpec(mode = mode, loss = loss, train_op = train_op, training_hooks = [summary_hook, logging_hook])


    return tf.estimator.EstimatorSpec(
        mode = mode, loss = loss, eval_metric_ops = eval_metric_ops)




numOfIteration = 0

BWQ = np.array((1261144, 317488, 80392, 20704, 5176))

mat = spio.loadmat('data/KSC/selectedLabels/X_val_original.mat', squeeze_me = True)                                                                                         
X_val_original = mat['X_val_original']

mat = spio.loadmat('data/KSC/selectedLabels/X_val_high.mat', squeeze_me = True)                                                                                         
X_val_high = mat['X_val_high']

mat = spio.loadmat('data/KSC/selectedLabels/X_val_medium.mat', squeeze_me = True)                                                                                         
X_val_medium = mat['X_val_medium']

mat = spio.loadmat('data/KSC/selectedLabels/X_val_low.mat', squeeze_me = True)                                                                                         
X_val_low = mat['X_val_low']

mat = spio.loadmat('data/KSC/selectedLabels/X_val_verylow.mat', squeeze_me = True)                                                                                         
X_val_verylow = mat['X_val_verylow']

mat = spio.loadmat('data/KSC/selectedLabels/y_val.mat', squeeze_me = True)                                                                                         
y_val = mat['y_val']

mat = spio.loadmat('data/KSC/selectedLabels/indices.mat', squeeze_me = True)                                                                                         
indices = mat['indices']
    

class NDSparseMatrix:
    def __init__(self):
        self.elements = {}

    def addValue(self, tuple, value):
        self.elements[tuple] = value

    def readValue(self, tuple):
        try:
            value = self.elements[tuple]
        except KeyError:
            # could also be 0.0 if using floats...
            value = 0.0
        return value

loss_obj = NDSparseMatrix()
##pickle.dump( lossMat, open( "data/IN/selectedLabels/lossMat.file", "wb" ) )

class Node:
    def __init__(self, quality, parent_obj, BW, quantity, availableBands):
        self.quality = quality # 0 is the highest quality, -1 is for root
        self.parent_obj = parent_obj
        self.BW = BW
        self.quantity = int(quantity) # It can be 0
        self.availableBands = availableBands
        self.children = []
        self.status = True
        self.lossUpperBound = 1000000
        if (self.quality == -1):
            self.positions = np.array([], dtype = np.int32)
        elif (self.quality == 0):
            self.positions = np.array([self.quantity], dtype = np.int32)
        elif (self.quality == 1):
            self.positions = np.array([self.parent_obj.quantity, self.quantity], dtype = np.int32)
        elif (self.quality == 2):
            self.positions = np.array([self.parent_obj.parent_obj.quantity, self.parent_obj.quantity, self.quantity], dtype = np.int32)
        elif (self.quality == 3):
            self.positions = np.array([self.parent_obj.parent_obj.parent_obj.quantity, self.parent_obj.parent_obj.quantity,
                             self.parent_obj.quantity, self.quantity], dtype = np.int32)
        else:
            self.positions = np.array([self.parent_obj.parent_obj.parent_obj.parent_obj.quantity, self.parent_obj.parent_obj.parent_obj.quantity,
                             self.parent_obj.parent_obj.quantity, self.parent_obj.quantity, self.quantity], dtype = np.int32)

##        self.positions = np.reshape(self.positions, (1, len(self.positions)))
         
##        print("Hi! I am a Node:) my quality is: " + str(quality)
##              + ", my bandwidth is: " + str(BW) + ", the number of band with this quality: "
##              + str(quantity) + ", the available bands are: " + str(availableBands) + ", and my positions are: " + str(self.positions))
    def introduce(self):
        print("Hi! I am a Node:) my quality is: " + str(self.quality)
              + ", my bandwidth is: " + str(self.BW) + ", the number of band with this quality: "
              + str(self.quantity) + ", the available bands are: " + str(self.availableBands)
              + ", I have " + str(len(self.children)) + " children, my status is: " + str(self.status)
              + ", and my positions are: " + str(self.positions)
              + ", and my lossUpperBound is: " + str(self.lossUpperBound)
              + ", and my parent position is: ", self.parent_obj.positions)
        
def make_tree(rootNode_obj):
##    print("\n\n!!!!!!!!!\n\n")
    if(rootNode_obj.quality >= 4):
##        print("Oops... It is a leaf!")
        return
    availableBW = rootNode_obj.BW - BWQ[rootNode_obj.quality] * rootNode_obj.quantity
##    print(" *********Okay... The avaialble bandwidth is: " + str(availableBW))
    currentQuality = rootNode_obj.quality + 1
##    print("The current quality is: " + str(currentQuality))
    maxNumOfChildren = int(math.floor(availableBW/BWQ[currentQuality]))
##    print("The maximum number of children is: " + str(maxNumOfChildren))
    if (maxNumOfChildren == 0):
        a = Node(currentQuality, rootNode_obj, availableBW, 0, rootNode_obj.availableBands)
        rootNode_obj.children.append(a)
        a.parent_obj = rootNode_obj
        make_tree(rootNode_obj.children[-1])
        return
    
    forRange = min(maxNumOfChildren + 1, rootNode_obj.availableBands)
    for i in range(forRange):
##        print("Now we are in the for loop, step:" + str(i) + "/" + str(forRange - 1))
        a = Node(currentQuality, rootNode_obj, availableBW, i, rootNode_obj.availableBands - i)
        rootNode_obj.children.append(a)
        a.parent_obj = rootNode_obj
        make_tree(rootNode_obj.children[-1])

##def deleteNode(node_obj):
##    node_obj.status = False
##    while(len(node_obj.quality) == 3):
##        for i in range(len(node_obj.children)):
##            node_obj.children[i].status = False
            

def testNode(C, quality, isUpperBound = False): # "quality" is an array of 5 numbers
    global numOfIteration
    numOfIteration += 1
    if (isUpperBound == False):
        newQuality = np.cumsum(quality)
##        print("&&&&&&&&&&&&& This is the actual loss")
    else:
        newQuality = quality
##        print("&&&&&&&&&&&&& This is just an upper bound for loss")

##    print("\n%%%%%%%%%%%%%%%%%%%% The cumulative quality is: ", newQuality)

    
    if (np.array_equal(newQuality, np.array([0,0,0,0,0]))):
##        print("#^^^^^^^^#")
        return 1000000
    
    currentLoss = loss_obj.readValue((1,2,3))
    if(currentLoss != 0):
        return loss_obj.readValue(qualityIndex)
    
    X_val = np.zeros(X_val_original.shape)
##    print("quality shape: ", newQuality[0].shape)
    if(newQuality[0] != 0):
        for l in range(newQuality[0]):                                                                                                
            X_val[:,:,:,indices[l]] = X_val_original[:,:,:,indices[l]]
    if(newQuality[1] != newQuality[0]):
        for l in range(newQuality[1]-newQuality[0]):
            X_val[:,:,:,indices[l+newQuality[0]]] = X_val_high[:,:,:,indices[l+newQuality[0]]]
    if(newQuality[2] != newQuality[1]):
        for l in range(newQuality[2]-newQuality[1]):
            X_val[:,:,:,indices[l+newQuality[1]]] = X_val_medium[:,:,:,indices[l+newQuality[1]]]
    if(newQuality[3] != newQuality[2]):
        for l in range(newQuality[3]-newQuality[2]):
            X_val[:,:,:,indices[l+newQuality[2]]] = X_val_low[:,:,:,indices[l+newQuality[2]]]
    if(newQuality[4] != newQuality[3]):
        for l in range(newQuality[4]-newQuality[3]):
            X_val[:,:,:,indices[l+newQuality[3]]] = X_val_verylow[:,:,:,indices[l+newQuality[3]]]

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x = {"x": X_val},
        y = y_val,
        batch_size = 16,
        num_epochs = 1,
        shuffle = False)
    
    eval_results = C.evaluate(input_fn = eval_input_fn, hooks = None)
    loss_obj.addValue(tuple(newQuality), eval_results['loss'])
    return eval_results

def upperBound(C, node_obj, boundPosition = None, isUpperBound = True): # The quality is a vector from root to the node_obj
    if (np.all(boundPosition) == None):
        boundPosition = node_obj.positions
        if(node_obj.quality >= 3):
            isUpperBound = False
##            print("%$%^#@%$ actual loss instead of an upper bound")
    if(node_obj.lossUpperBound != 1000000):
        return node_obj.lossUpperBound
    if (node_obj.quality == 3):
        boundPosition = np.append(boundPosition, min(node_obj.children[-1].quantity, node_obj.availableBands))
        if (len(boundPosition) < 5):
            quality = np.zeros(5)
##            print(" *.*.*.*.*. positions.shape: ", boundPosition.shape, "quality.shape: ", quality.shape)
            quality[quality.shape[0] - boundPosition.shape[0]:quality.shape[0]] = boundPosition
            boundPosition = quality
##        node_obj.positions = int(node_obj.positions)
##        quality = np.cumsum(boundPosition)
##        print("\n\n *.*.*.*.*.  Here we are in the 'upperBound' function and the quality vector is: ", boundPosition)
        return testNode(C, boundPosition, isUpperBound)['loss']
    boundPosition = np.append(boundPosition, min(node_obj.children[-1].quantity, node_obj.availableBands))
    a = upperBound(C, node_obj.children[0], boundPosition, isUpperBound)
    if (np.all(boundPosition) == None):
        node_obj.lossUpperBound = a
        return node_obj.lossUpperBound
    else:
        return a

def traverseTree(rootNode_obj, positions): # Takes an array with maximum length of 5 and outputs the regarding node
    outputNode_obj = rootNode_obj
    for i in range(len(positions)):
        outputNode_obj = outputNode_obj.children[positions[i]]
    return outputNode_obj


def isAncestor(descendentNode_obj, ancestorNode_obj):
    currentNode_obj = descendentNode_obj
    for i in range(descendentNode_obj.quality - ancestorNode_obj.quality):
        if(currentNode_obj.parent == ancestorNode_obj):
            return True
        currentNode_obj = currentNode_obj.parent
    return False
            

    

def BBInitialLoss(C, rootNode_obj, n):
##    print("$$$\n\n\n We're inside BB initial loss and n is: ", n)
    currentNode_obj = rootNode_obj
    for i in range(n-1):
        k = 0
        currentUpperBound = 1000000
        for j in range(len(currentNode_obj.children)):
##            print('\n\n\n')
##            print("------------------ level and child#: ", i, j)
##            currentNode_obj.children[j].introduce()
            newUpperBound = upperBound(C, currentNode_obj.children[j])
##            print("@@@@ new and current upper bound: ", newUpperBound, currentUpperBound)
            if (newUpperBound < currentUpperBound):
                k = j
##                print("Hello!! k is: ", k)
                currentUpperBound = newUpperBound
        currentNode_obj = currentNode_obj.children[k]
##        currentNode_obj.introduce()
    currentNode_obj = currentNode_obj.children[-1]
##    currentNode_obj.introduce()
##    print("The initial Loss is: ", testNode(C, currentNode_obj.positions, False))
    if (n == 1):
        rootNode_obj.lossUpperBound = testNode(C, currentNode_obj.positions, False)['loss']
    else:
        rootNode_obj.lossUpperBound = currentUpperBound
        
    return currentNode_obj

def BB4(C, currentNode_obj):
##    print("\n$$$ We are in BB4")
    currentNode_obj.introduce()
    parentNode_obj = currentNode_obj.parent_obj
    for i in range(len(parentNode_obj.children)):
        if (parentNode_obj.children[i] == currentNode_obj):
            continue
        parentNode_obj.children[i].status = False

def BB3(C, currentNode_obj):
    newNode_obj = None
##    print("\n$$$ We are in BB3")
    currentNode_obj.introduce()
    currentLoss = testNode(C, currentNode_obj.positions)['loss']
    parentNode_obj = currentNode_obj.parent_obj
    gParentNode_obj = parentNode_obj.parent_obj
    
    for i in range(len(gParentNode_obj.children)):
##        print("$$$ we are in BB3 and i is: ",i,"/",len(gParentNode_obj.children))
        if (gParentNode_obj.children[i] == parentNode_obj):
            continue
        if (gParentNode_obj.children[i].status == False):
            continue            
        if (upperBound(C, gParentNode_obj.children[i]) >= currentLoss):
            gParentNode_obj.children[i].status = False
        else:
            newNode_obj = gParentNode_obj.children[i].children[-1]
            newLoss = testNode(C, newNode_obj.positions)['loss']
            if ( newLoss < currentLoss):
                currentLoss = newLoss
                parentNode_obj.status = False
##                print("#@#@#@#@@#@#@#@# The current node has been changed in BB3 and ...")
    if (newNode_obj != None):
        currentNode_obj = newNode_obj
##        print("\n\n The current Node is changed to: ")
        currentNode_obj.introduce()
##        print("\n\n")
    return currentNode_obj

def BB2(C, currentNode_obj):
    newNode_obj = None
##    print("\n\n$$$ We are in BB2")
    currentNode_obj.introduce()
    currentLoss = testNode(C, currentNode_obj.positions)['loss']
    parentNode_obj = currentNode_obj.parent_obj
    gParentNode_obj = parentNode_obj.parent_obj
    ggParentNode_obj = gParentNode_obj.parent_obj

    for i in range(len(ggParentNode_obj.children)):
##        print("$$$ we are in BB2 and i is: ",i,"/",len(ggParentNode_obj.children))
##        if (ggParentNode_obj.children[i] == gParentNode_obj):
##            continue
        if (ggParentNode_obj.children[i].status == False):
            continue            
        if (upperBound(C, ggParentNode_obj.children[i]) >= currentLoss):
            ggParentNode_obj.children[i].status = False
        else:
            initialNode_obj = BBInitialLoss(C, ggParentNode_obj.children[i], 2)
            BB4(C, initialNode_obj)
            tempNode_obj = BB3(C, initialNode_obj)
            newLoss = testNode(C, tempNode_obj.positions)['loss']
            if ( newLoss < currentLoss):
                currentLoss = newLoss
                parentNode_obj.status = False
                gParentNode_obj.status = False
                newNode_obj = tempNode_obj
##                print(" BB2 and ...!\n")
            else:
##                print(" ...not in BB2! Just an upper bound!")
                ggParentNode_obj.children[i].status == False
    if (newNode_obj != None):
        currentNode_obj = newNode_obj
##        print("\n\n The current Node is changed to: ")
        currentNode_obj.introduce()
##        print("\n\n")
    return currentNode_obj

def BB1(C, currentNode_obj):
    newNode_obj = None
##    print("\n\n$$$ We are in BB1")
    currentNode_obj.introduce()
    currentLoss = testNode(C, currentNode_obj.positions)['loss']
    parentNode_obj = currentNode_obj.parent_obj
    gParentNode_obj = parentNode_obj.parent_obj
    ggParentNode_obj = gParentNode_obj.parent_obj
    gggParentNode_obj = ggParentNode_obj.parent_obj

    for i in range(len(gggParentNode_obj.children)):
##        print("$$$ we are in BB1 and i is: ",i,"/",len(gggParentNode_obj.children))
##        if (gggParentNode_obj.children[i] == ggParentNode_obj):
##            continue
        if (gggParentNode_obj.children[i].status == False):
            continue            
        if (upperBound(C, gggParentNode_obj.children[i]) >= currentLoss):
            gggParentNode_obj.children[i].status = False
        else:
            initialNode_obj = BBInitialLoss(C, gggParentNode_obj.children[i], 3)
            BB4(C, initialNode_obj)
            tempNode_obj = BB3(C, initialNode_obj)
            newTempNode_obj = BB2(C, tempNode_obj)
            newLoss = testNode(C, newTempNode_obj.positions)['loss']
            if ( newLoss < currentLoss):
                currentLoss = newLoss
                parentNode_obj.status = False
                gParentNode_obj.status = False
                ggParentNode_obj.status = False
                newNode_obj = newTempNode_obj
##                print(" BB1 and ...!\n")
            else:
##                print(" ...not in BB1! Just an upper bound!")
                gggParentNode_obj.children[i].status == False
                
    if (newNode_obj != None):
        currentNode_obj = newNode_obj
##        print("\n\n The current Node is changed to: ")
        currentNode_obj.introduce()
##        print("\n\n")
    return currentNode_obj


def BB0(C, currentNode_obj):
    newNode_obj = None
##    print("\n\n\n$$$ We are in BB0")
    currentNode_obj.introduce()
    currentLoss = testNode(C, currentNode_obj.positions)['loss']
##    print("$$$ The current loss is: ", currentLoss)
    parentNode_obj = currentNode_obj.parent_obj
    gParentNode_obj = parentNode_obj.parent_obj
    ggParentNode_obj = gParentNode_obj.parent_obj
    gggParentNode_obj = ggParentNode_obj.parent_obj
    ggggParentNode_obj = gggParentNode_obj.parent_obj

    for i in range(len(ggggParentNode_obj.children)):
##        print("\n\n\n$$$ we are in BB0 and i is: ",i,"/",len(ggggParentNode_obj.children))
##        if (ggggParentNode_obj.children[i] == gggParentNode_obj):
##            BB1(C, ggggParentNode_obj.children[i])
        if (ggggParentNode_obj.children[i].status == False):
            continue
        uBound = upperBound(C, ggggParentNode_obj.children[i])
        if (uBound >= currentLoss):
            ggggParentNode_obj.children[i].status = False
##            print("$$$ The upper bound is lower than the current loss", uBound, currentLoss)
        else:
            initialNode_obj = BBInitialLoss(C, ggggParentNode_obj.children[i], 4)
            BB4(C, initialNode_obj)
            tempNode_obj = BB3(C, initialNode_obj)
            newTempNode_obj = BB2(C, tempNode_obj)
            newNewTempNode_obj = BB1(C, newTempNode_obj)
            newLoss = testNode(C, newNewTempNode_obj.positions)['loss']
            if ( newLoss < currentLoss):
                currentLoss = newLoss
                parentNode_obj.status = False
                gParentNode_obj.status = False
                ggParentNode_obj.status = False
                gggParentNode_obj.status = False
                newNode_obj = newTempNode_obj
##                print(" BB0 and ...!\n")
            else:
##                print(" ...not in BB1! Just an upper bound!")
                gggParentNode_obj.children[i].status == False
                
    if (newNode_obj != None):
        currentNode_obj = newNode_obj
##        print("\n\n The current Node is changed to: ")
        currentNode_obj.introduce()
##        print("\n\n")
    return currentNode_obj

           
def main(unused_argv):

    
    
    ## Loading Data
    
##    with open( "data/IN/selectedLabels/lossMat.file", "rb" ) as f:
##        loss_obj = pickle.load(f)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config = config)
    
    ## Create the Estimator
    SSRN_classifier = tf.estimator.Estimator(model_fn=SSRN_model_fn, model_dir = "KSC_model_ReducedClasses/", config = tf.contrib.learn.RunConfig(
            save_checkpoints_steps = 128,
            save_checkpoints_secs = None,
            keep_checkpoint_max = 201,
            save_summary_steps = 500))
    
##    currentNode_obj = BBInitialLoss(SSRN_classifier, root)
    
    for i in range(50):
        file_obj = open("KSC_ReducedClasses.txt", "a")
        root = Node(-1, None, (i+1)*50000, 0, 176)
        make_tree(root)
        currentNode_obj = BBInitialLoss(SSRN_classifier, root, 5)
        finalNode_obj = BB0(SSRN_classifier, currentNode_obj)
        file_obj.write(" " + str((i+1)*50000)+ " " + str(finalNode_obj.positions)+ " " + str(testNode(SSRN_classifier, finalNode_obj.positions)) + str(numOfIteration) + "\n")
        global numOfIteration
        numOfIteration = 0
        file_obj.close()
    

    
if __name__ == "__main__":
  tf.app.run()
