from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


## Imports
import numpy as np
##import pywt
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.io as spio
##from sklearn import preprocessing
##from scipy.misc import imresize
##from sklearn.decomposition import PCA
##from sklearn.model_selection import train_test_split
from tensorflow.contrib.learn.python.learn import monitors as monitor_lib

tf.logging.set_verbosity(tf.logging.INFO)

def convbnrelu3d(inputs, filters, kernel_size, strides, padding, initializer, regularizer, mode):
    convolution = tf.layers.conv3d(
        inputs = inputs,
        filters = filters,
        kernel_size = kernel_size,
        strides = strides,
        padding = padding,
        kernel_initializer = initializer,
        kernel_regularizer = regularizer,
        activation = None)
##    bn = tf.layers.batch_normalization(convolution, axis=-1, center=True, scale=True, training=(mode == tf.estimator.ModeKeys.TRAIN))
    return tf.nn.relu(convolution)

def convbnrelu2d(inputs, filters, kernel_size, padding, initializer, regularizer, mode):
    convolution = tf.layers.conv2d(
        inputs = inputs,
        filters = filters,
        kernel_size = kernel_size,
        padding = padding,
        kernel_initializer = initializer,
        kernel_regularizer = regularizer,
        activation = None)
##    bn = tf.layers.batch_normalization(convolution, axis=-1, center=True, scale=True, training=(mode == tf.estimator.ModeKeys.TRAIN))
    return tf.nn.relu(convolution)


def convsumbnrelu3d(inputs, relu, filters, kernel_size, strides, padding, initializer, regularizer, mode):
    convolution = tf.layers.conv3d(
        inputs = inputs,
        filters = filters,
        kernel_size = kernel_size,
        strides = strides,
        padding = padding,
        kernel_initializer = initializer,
        kernel_regularizer = regularizer,
        activation = None)
    summation = relu + convolution
##    bn = tf.layers.batch_normalization(summation, axis=-1, center=True, scale=True, training=(mode == tf.estimator.ModeKeys.TRAIN))
    return (tf.nn.relu(convolution), summation)

def convsumbnrelu2d(inputs, relu, filters, kernel_size, padding, initializer, regularizer, mode):
    convolution = tf.layers.conv2d(
        inputs = inputs,
        filters = filters,
        kernel_size = kernel_size,
        padding = padding,
        kernel_initializer = initializer,
        kernel_regularizer = regularizer,
        activation = None)

    summation = convolution + relu
##    bn = tf.layers.batch_normalization(summation, axis=-1, center=True, scale=True, training=(mode == tf.estimator.ModeKeys.TRAIN))
    return (tf.nn.relu(convolution), summation)

def SSRN_model_fn(features, labels, mode):

    regularizer = tf.contrib.layers.l2_regularizer(scale = 0.0000)
    initializer = tf.contrib.layers.xavier_initializer(uniform = False) #tf.random_normal_initializer(0, 0.1) #glorot_normal_initializer()

    input_layer_spectral = tf.transpose(features["x"], perm= [0, 3, 1, 2])
    input_layer_spectral = tf.reshape(input_layer_spectral, [-1, 103, 7, 7, 1])
    input_layer_spectral = tf.cast(input_layer_spectral, tf.float32)
    conv1 = convbnrelu3d(input_layer_spectral, 24, [7, 1, 1], (2, 1, 1), "valid", initializer, regularizer, mode)
    print("shape of conv1 is: ", conv1.shape) #(?, 49, 7, 7, 24)
    conv2 = convbnrelu3d(conv1, 24, [7, 1, 1], (1, 1, 1), "same", initializer, regularizer, mode)
    print("shape of conv2 is: ", conv2.shape) #(?, 49, 7, 7, 24)
    (conv3, summation1) = convsumbnrelu3d(conv2, conv1, 24, [7, 1, 1], (1, 1, 1), "same", initializer, regularizer, mode)
    conv4 = convbnrelu3d(conv3, 24, [7, 1, 1], (1, 1, 1), "same", initializer, regularizer, mode)
    (conv5, summation2) = convsumbnrelu3d(conv4, summation1, 24, [7, 1, 1], (1, 1, 1), "same", initializer, regularizer, mode)
    bn5 = tf.layers.batch_normalization(conv5, axis = -1, center = True, scale = True, training = (mode == tf.estimator.ModeKeys.TRAIN))
    relu5 = tf.nn.relu(bn5)

    conv6 = convbnrelu3d(relu5, 128, [49, 1, 1], (1, 1, 1), "valid", initializer, regularizer, mode)
    print("shape of conv6 is: ", conv6.shape) #(?, 1, 7, 7, 128)

    ## start of spatial block
    input_layer_spatial = tf.reshape(conv6, [-1, 7, 7, 128])
    conv7 = convbnrelu2d(input_layer_spatial, 24, [3, 3], "valid", initializer, regularizer, mode)
    print("shape of conv7 is: ", conv7.shape) #(?, 5, 5, 24)
    conv8 = convbnrelu2d(conv7, 24, [3, 3], "same", initializer, regularizer, mode)
    (conv9, summation3) = convsumbnrelu2d(conv8, conv7, 24, [3, 3], "same", initializer, regularizer, mode)
    conv10 = convbnrelu2d(conv9, 24, [3, 3], "same", initializer, regularizer, mode)
    (conv11, summation4) = convsumbnrelu2d(conv10, summation3, 24, [3, 3], "same", initializer, regularizer, mode)

    bn12 = tf.layers.batch_normalization(conv11, axis = -1, center = True, scale = True, training = (mode == tf.estimator.ModeKeys.TRAIN))
    relu12 = tf.nn.relu(bn12)
    print("shape of relu12 is: ", relu12.shape) #(?, 5, 5, 24)

    pool1 = tf.layers.average_pooling2d(inputs = relu12, pool_size = [5, 5], strides = 1)
    pool1_flat = tf.reshape(pool1, [-1, 24]) #flatten

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
		    output_dir = "UP_model/summary/train",
		    summary_op = [Loss_train, Accuracy_train])

	    logging_hook = tf.train.LoggingTensorHook({"reset":running_vars_initializer, "loss_train": loss,  "accuracy_train":update_op1}, every_n_iter = 128)


	    optimizer = tf.train.RMSPropOptimizer(learning_rate = 0.0003)
	    with tf.control_dependencies(update_ops):
	        train_op = optimizer.minimize(loss = loss, global_step = tf.train.get_global_step())
	    return tf.estimator.EstimatorSpec(mode = mode, loss = loss, train_op = train_op, training_hooks = [summary_hook, logging_hook])


    return tf.estimator.EstimatorSpec(
        mode = mode, loss = loss, eval_metric_ops = eval_metric_ops)

def scenario(All_features_, All_labels):
    
    All_features = All_features_
    for quality in range(5):
        cAA = np.zeros((610, 340, 103))
        cA = None
        
        if(quality != 0):
            for i in range(103):
                cA = pywt.wavedec2(All_features_[:,:,i], 'db1', level = quality)[0]
                cAA[:,:,i] = imresize(cA, (610, 340)).astype('float32')
            All_features = cAA
        data = All_features.reshape(610 * 340, 103)                                                                                                         
        data = preprocessing.scale(data.astype('float64'))
        All_features = data.reshape(610, 340, 103)
        pad_length = 3                                                                                                                                     
        pad_depth = 0                                                                                                                                     
        All_features_padded = np.lib.pad(All_features, ((pad_length, pad_length), (pad_length, pad_length), (pad_depth, pad_depth)), 'constant', constant_values = 0)
                                                                                                                                                           
        for L in  range(1,8):
            count = 0;
            All_features_inBlocks = np.zeros((610 * 340, 7, 7, 103))
            All_labels_inBlocks = np.zeros((610 * 340))

            for i in range(3, 610 + 3):
                for j in range(3, 340 + 3):
                    if All_labels[i - 3, j - 3] == L:
                        All_features_inBlocks[count, :, :, :] = All_features_padded[i - 3 : i + 4, j - 3 : j + 4, :].reshape((1, 7, 7, 103))
                        All_labels_inBlocks[count] = All_labels[i - 3, j - 3] - 1 # original labels are from 1-16                                                
                        count = count + 1;                                                                                                                     
                                                                                                                                                           
                                                                                                                                                           
            ## Divide training, evaluation and test data (20%, 10%, 70%)                                                                                   
            if (L == 1):
                X_train, X, y_train, y = train_test_split(All_features_inBlocks[0 : count, :, :, :], All_labels_inBlocks[0 : count], test_size = int(0.9 * count), random_state = 42)  
                X_val, X_test, y_val, y_test = train_test_split(X, y, train_size = int(0.11 * len(y)), random_state = 42)                                                        
                                                                                                                                                           
            else:                                                                                                                                          
                X_train_n, X, y_train_n, y = train_test_split(All_features_inBlocks[0 : count, :, :, :], All_labels_inBlocks[0 : count], test_size=int(0.9 * count), random_state = 42)
                X_val_n, X_test_n, y_val_n, y_test_n = train_test_split(X, y, train_size = int(0.11 * len(y)), random_state = 42)                                                
                X_test = np.concatenate((X_test, X_test_n))                                                                                                
                X_train = np.concatenate((X_train, X_train_n))                                                                                             
                X_val = np.concatenate((X_val, X_val_n))                                                                                                   
                y_test = np.concatenate((y_test, y_test_n))                                                                                                
                y_train = np.concatenate((y_train, y_train_n))                                                                                             
                y_val = np.concatenate((y_val, y_val_n))
                
        if (quality == 0):                                                                                             
            X_train_original = X_train
            X_val_original = X_val                  
            y_train_original = y_train                                         
            y_val_original = y_val
            
        if (quality == 1):                                                                                                
            X_train_high = X_train
            X_val_high = X_val                  
            y_train_high = y_train                                         
            y_val_high = y_val
            
        if (quality == 2):                                                                                                
            X_train_medium = X_train
            X_val_medium = X_val                  
            y_train_medium = y_train                                         
            y_val_medium = y_val
            
        if (quality == 3):                                                                                                
            X_train_low = X_train
            X_val_low = X_val                  
            y_train_low = y_train                                         
            y_val_low = y_val
            
        if (quality == 4):                                                                                               
            X_train_verylow = X_train
            X_val_verylow = X_val                  
            y_train_verylow = y_train                                         
            y_val_verylow = y_val
        
    spio.savemat('data/UP/allLabels/y_train.mat', {"y_train" : y_train_original})
    spio.savemat('data/UP/allLabels/y_val.mat', {"y_val" : y_val_original})

    spio.savemat('data/UP/allLabels/X_train_original.mat', {"X_train_original" : X_train_original})
    spio.savemat('data/UP/allLabels/X_train_high.mat', {"X_train_high" : X_train_high})
    spio.savemat('data/UP/allLabels/X_train_medium.mat', {"X_train_medium" : X_train_medium})
    spio.savemat('data/UP/allLabels/X_train_low.mat', {"X_train_low" : X_train_low})
    spio.savemat('data/UP/allLabels/X_train_verylow.mat', {"X_train_verylow" : X_train_verylow})
    
    spio.savemat('data/UP/allLabels/X_val_original.mat', {"X_val_original" : X_val_original})
    spio.savemat('data/UP/allLabels/X_val_high.mat', {"X_val_high" : X_val_high})
    spio.savemat('data/UP/allLabels/X_val_medium.mat', {"X_val_medium" : X_val_medium})
    spio.savemat('data/UP/allLabels/X_val_low.mat', {"X_val_low" : X_val_low})
    spio.savemat('data/UP/allLabels/X_val_verylow.mat', {"X_val_verylow" : X_val_verylow})
        

def makeTrainingSet(indices):
    mat = spio.loadmat('data/UP/allLabels/X_train_original.mat', squeeze_me = True)                                                                                         
    X_train_original = mat['X_train_original']
    mat = spio.loadmat('data/UP/allLabels/X_train_high.mat', squeeze_me = True)                                                                                         
    X_train_high = mat['X_train_high']
    mat = spio.loadmat('data/UP/allLabels/X_train_medium.mat', squeeze_me = True)                                                                                         
    X_train_medium = mat['X_train_medium']
    mat = spio.loadmat('data/UP/allLabels/X_train_low.mat', squeeze_me = True)                                                                                         
    X_train_low = mat['X_train_low']
    mat = spio.loadmat('data/UP/allLabels/X_train_verylow.mat', squeeze_me = True)                                                                                         
    X_train_verylow = mat['X_train_verylow']
    
    mat = spio.loadmat('data/UP/allLabels/y_train.mat', squeeze_me = True)                                                                                         
    y_train = mat['y_train']
    
    X_train = np.zeros(X_train_original.shape, dtype = np.float64)

    for j in range(X_train.shape[0]):
        quality = np.sort(np.random.randint(103, size = 5)) #quality = 5 means the band has not been picked
        for l in range(quality[0]):                                                                                                
            X_train[j,:,:,indices[l]] = X_train_original[j,:,:,indices[l]]
        if(quality[1] != quality[0]):
            for l in range(quality[1]-quality[0]):
                X_train[j,:,:,indices[l+quality[0]]] = X_train_high[j,:,:,indices[l+quality[0]]]
        if(quality[2] != quality[1]):
            for l in range(quality[2]-quality[1]):
                X_train[j,:,:,indices[l+quality[1]]] = X_train_medium[j,:,:,indices[l+quality[1]]]
        if(quality[3] != quality[2]):
            for l in range(quality[3]-quality[2]):
                X_train[j,:,:,indices[l+quality[2]]] = X_train_low[j,:,:,indices[l+quality[2]]]
        if(quality[4] != quality[3]):
            for l in range(quality[4]-quality[3]):
                X_train[j,:,:,indices[l+quality[3]]] = X_train_verylow[j,:,:,indices[l+quality[3]]]
    
    X_train_final = X_train
    y_train_final = y_train
    
    for i in range(2):
        print (i)
        for j in range(X_train.shape[0]):
            quality = np.sort(np.random.randint(103, size = 5)) #quality = 5 means the band has not been picked
            for l in range(quality[0]):                                                                                                
                X_train[j,:,:,indices[l]] = X_train_original[j,:,:,indices[l]]
            if(quality[1] != quality[0]):
                for l in range(quality[1]-quality[0]):
                    X_train[j,:,:,indices[l+quality[0]]] = X_train_high[j,:,:,indices[l+quality[0]]]
            if(quality[2] != quality[1]):
                for l in range(quality[2]-quality[1]):
                    X_train[j,:,:,indices[l+quality[1]]] = X_train_medium[j,:,:,indices[l+quality[1]]]
            if(quality[3] != quality[2]):
                for l in range(quality[3]-quality[2]):
                    X_train[j,:,:,indices[l+quality[2]]] = X_train_low[j,:,:,indices[l+quality[2]]]
            if(quality[4] != quality[3]):
                for l in range(quality[4]-quality[3]):
                    X_train[j,:,:,indices[l+quality[3]]] = X_train_verylow[j,:,:,indices[l+quality[3]]]
        
        X_train_final = np.concatenate((X_train_final, X_train), axis = 0)
        y_train_final = np.concatenate((y_train_final, y_train), axis = 0)
                                                                                                                                                                   
##    spio.savemat('data/UP/allLabels/y_train.mat',{"y_train":y_train_final})
##    spio.savemat('data/UP/allLabels/X_train.mat',{"X_train":X_train_final})
    return (X_train_final, y_train_final)

def main(unused_argv):

    mat = spio.loadmat('data/UP/allLabels/indices.mat', squeeze_me = True)                                                                                         
    indices = mat['indices']
    
##    mat = spio.loadmat('datasets/UP/paviaU.mat', squeeze_me=True)                                                                         
##    All_features_ = mat['paviaU']                                                                                                      
##                                                                                                                                                   
##    mat = spio.loadmat('datasets/UP/paviaU_gt.mat', squeeze_me=True)                                                                                
##    All_labels = mat['paviaU_gt']
##    for i in range(All_labels.shape[0]):
##        for j in range(All_labels.shape[1]):
##            if(All_labels[i, j] == 4):
##                All_labels[i, j] = 2
##            elif(All_labels[i, j] == 5):
##                All_labels[i, j] = 4
##            elif(All_labels[i, j] == 6):
##                All_labels[i, j] = 5
##            elif(All_labels[i, j] == 7):
##                All_labels[i, j] = 1
##            elif(All_labels[i, j] == 8):
##                All_labels[i, j] = 6
##            elif(All_labels[i, j] == 9):
##                All_labels[i, j] = 7
##    scenario(All_features_, All_labels)
    (X_train, y_train) = makeTrainingSet(indices)
    
##    mat = spio.loadmat('data/UP/allLabels/X_train.mat', squeeze_me = True)                                                                                         
##    X_train = mat['X_train']
##    mat = spio.loadmat('data/UP/allLabels/y_train.mat', squeeze_me = True)                                                                                         
##    y_train = mat['y_train']
    mat = spio.loadmat('data/UP/allLabels/X_val_original.mat', squeeze_me = True)                                                                                         
    X_val = mat['X_val_original']
    mat = spio.loadmat('data/UP/allLabels/y_val.mat', squeeze_me = True)                                                                                         
    y_val = mat['y_val']
    
    print("X_train.shape and X_val.shape", X_train.shape, X_val.shape)
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config = config)
    
    ## Create the Estimator
    SSRN_classifier = tf.estimator.Estimator(model_fn = SSRN_model_fn, model_dir = "UP_model/", config = tf.contrib.learn.RunConfig(
            save_checkpoints_steps = 128,
            save_checkpoints_secs = None,
            keep_checkpoint_max = 201,
            save_summary_steps = 500))

    ## Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x = {"x": X_train},
        y = y_train,
        batch_size = 16,
        num_epochs = 100,
        shuffle = True)

    ## Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x = {"x": X_val},
        y = y_val,
        batch_size = 16,
        num_epochs = 1,
        shuffle = False)

    ## Set up logging
    print(type(X_val))
    print(X_val.shape)

    tensors_to_log = {}
    logging_hook = tf.train.LoggingTensorHook(tensors = tensors_to_log, every_n_iter = 128)
    validation_monitor_eval = tf.contrib.learn.monitors.ValidationMonitor(input_fn = eval_input_fn, every_n_steps = 128)
    hooks = monitor_lib.replace_monitors_with_hooks([validation_monitor_eval, logging_hook], SSRN_classifier)

    SSRN_classifier.train(input_fn = train_input_fn, steps = None, hooks = hooks)
##    eval_results = SSRN_classifier.evaluate(input_fn = eval_input_fn, hooks = None)
##    print("val results:",eval_results)

if __name__ == "__main__":
  tf.app.run()
