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
    input_layer_spectral = tf.reshape(input_layer_spectral, [-1, 200, 7, 7, 1])
    input_layer_spectral = tf.cast(input_layer_spectral, tf.float32)
    conv1 = convbnrelu3d(input_layer_spectral, 24, [7, 1, 1], (2, 1, 1), "valid", initializer, regularizer, mode)
    conv2 = convbnrelu3d(conv1, 24, [7, 1, 1], (1, 1, 1), "same", initializer, regularizer, mode)
    (conv3, summation1) = convsumbnrelu3d(conv2, conv1, 24, [7, 1, 1], (1, 1, 1), "same", initializer, regularizer, mode)
    conv4 = convbnrelu3d(conv3, 24, [7, 1, 1], (1, 1, 1), "same", initializer, regularizer, mode)
    (conv5, summation2) = convsumbnrelu3d(conv4, summation1, 24, [7, 1, 1], (1, 1, 1), "same", initializer, regularizer, mode)
    bn5 = tf.layers.batch_normalization(conv5, axis = -1, center = True, scale = True, training = (mode == tf.estimator.ModeKeys.TRAIN))
    relu5 = tf.nn.relu(bn5)

    conv6 = convbnrelu3d(relu5, 128, [97, 1, 1], (1, 1, 1), "valid", initializer, regularizer, mode)

    ## start of spatial block
    input_layer_spatial = tf.reshape(conv6, [-1, 7, 7, 128])
    conv7 = convbnrelu2d(input_layer_spatial, 24, [3, 3], "valid", initializer, regularizer, mode)
    conv8 = convbnrelu2d(conv7, 24, [3, 3], "same", initializer, regularizer, mode)
    (conv9, summation3) = convsumbnrelu2d(conv8, conv7, 24, [3, 3], "same", initializer, regularizer, mode)
    conv10 = convbnrelu2d(conv9, 24, [3, 3], "same", initializer, regularizer, mode)
    (conv11, summation4) = convsumbnrelu2d(conv10, summation3, 24, [3, 3], "same", initializer, regularizer, mode)

    bn12 = tf.layers.batch_normalization(conv11, axis = -1, center = True, scale = True, training = (mode == tf.estimator.ModeKeys.TRAIN))
    relu12 = tf.nn.relu(bn12)

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
		    output_dir = "justQuality/IN_model/summary/train",
		    summary_op = [Loss_train, Accuracy_train])

	    logging_hook = tf.train.LoggingTensorHook({"reset":running_vars_initializer, "loss_train": loss,  "accuracy_train":update_op1}, every_n_iter = 128)


	    optimizer = tf.train.RMSPropOptimizer(learning_rate = 0.0003)
	    with tf.control_dependencies(update_ops):
	        train_op = optimizer.minimize(loss = loss, global_step = tf.train.get_global_step())
	    return tf.estimator.EstimatorSpec(mode = mode, loss = loss, train_op = train_op, training_hooks = [summary_hook, logging_hook])


    return tf.estimator.EstimatorSpec(
        mode = mode, loss = loss, eval_metric_ops = eval_metric_ops)

def makeTrainingSet(indices):
    mat = spio.loadmat('data/IN/allLabels/X_train_original.mat', squeeze_me = True)                                                                                         
    X_train_original = mat['X_train_original']
    mat = spio.loadmat('data/IN/allLabels/X_train_high.mat', squeeze_me = True)                                                                                         
    X_train_high = mat['X_train_high']
    mat = spio.loadmat('data/IN/allLabels/X_train_medium.mat', squeeze_me = True)                                                                                         
    X_train_medium = mat['X_train_medium']
    mat = spio.loadmat('data/IN/allLabels/X_train_low.mat', squeeze_me = True)                                                                                         
    X_train_low = mat['X_train_low']
    mat = spio.loadmat('data/IN/allLabels/X_train_verylow.mat', squeeze_me = True)                                                                                         
    X_train_verylow = mat['X_train_verylow']
        
    mat = spio.loadmat('data/IN/allLabels/y_train.mat', squeeze_me = True)                                                                                         
    y_train = mat['y_train']
    

    X_train_final = np.concatenate((X_train_original, X_train_high, X_train_medium, X_train_low, X_train_verylow), axis = 0)
    y_train_final = np.concatenate((y_train, y_train, y_train, y_train, y_train), axis = 0)

    p = np.random.permutation(len(y_train_final))
    y_train_final = y_train_final[p]
    X_train_final = X_train_final[p]

                                                                                                                                                                   
    return (X_train_final, y_train_final)

def main(unused_argv):

    mat = spio.loadmat('data/IN/allLabels/indices.mat', squeeze_me = True)                                                                                         
    indices = mat['indices']    
##    (X_train, y_train) = makeTrainingSet(indices)
    
    mat = spio.loadmat('data/IN/allLabels/X_val_original.mat', squeeze_me = True)                                                                                         
    X_val_original = mat['X_val_original']
    mat = spio.loadmat('data/IN/allLabels/X_val_high.mat', squeeze_me = True)                                                                                         
    X_val_high = mat['X_val_high']
    mat = spio.loadmat('data/IN/allLabels/X_val_medium.mat', squeeze_me = True)                                                                                         
    X_val_medium = mat['X_val_medium']
    mat = spio.loadmat('data/IN/allLabels/X_val_low.mat', squeeze_me = True)                                                                                         
    X_val_low = mat['X_val_low']
    mat = spio.loadmat('data/IN/allLabels/X_val_verylow.mat', squeeze_me = True)                                                                                         
    X_val_verylow = mat['X_val_verylow']
    
    mat = spio.loadmat('data/IN/allLabels/y_val.mat', squeeze_me = True)                                                                                         
    y_val = mat['y_val']
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    
    ## Create the Estimator
    SSRN_classifier = tf.estimator.Estimator(model_fn = SSRN_model_fn, model_dir = "IN_model/", config = tf.contrib.learn.RunConfig(
            save_checkpoints_steps = 128,
            save_checkpoints_secs = None,
            keep_checkpoint_max = 201,
            save_summary_steps = 500))

    ## Train the model
##    train_input_fn = tf.estimator.inputs.numpy_input_fn(
##        x = {"x": X_train},
##        y = y_train,
##        batch_size = 16,
##        num_epochs = 100,
##        shuffle = True)

    ## Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x = {"x": X_val_original},
        y = y_val,
        batch_size = 16,
        num_epochs = 1,
        shuffle = False)

    ## Set up logging

    tensors_to_log = {}
    logging_hook = tf.train.LoggingTensorHook(tensors = tensors_to_log, every_n_iter = 128)
    validation_monitor_eval = tf.contrib.learn.monitors.ValidationMonitor(input_fn = eval_input_fn, every_n_steps = 128)
    hooks = monitor_lib.replace_monitors_with_hooks([validation_monitor_eval, logging_hook], SSRN_classifier)

##    SSRN_classifier.train(input_fn = train_input_fn, steps = None, hooks = hooks)
    
    file_obj = open("justQuality_IN2.txt", "a")
    
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x = {"x": X_val_verylow},
            y = y_val,
            batch_size = 16,
            num_epochs = 1,
            shuffle = False)
    eval_results = SSRN_classifier.evaluate(input_fn = eval_input_fn, hooks = None)
    print("val results:", eval_results)
    file_obj.write("Very Low:" + " " + str(eval_results['accuracy']) + " " + str(eval_results['loss']) + "\n")

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x = {"x": X_val_low},
            y = y_val,
            batch_size = 16,
            num_epochs = 1,
            shuffle = False)
    eval_results = SSRN_classifier.evaluate(input_fn = eval_input_fn, hooks = None)
    print("val results:", eval_results)
    file_obj.write("Low:" + " " + str(eval_results['accuracy']) + " " + str(eval_results['loss']) + "\n")

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x = {"x": X_val_medium},
            y = y_val,
            batch_size = 16,
            num_epochs = 1,
            shuffle = False)
    eval_results = SSRN_classifier.evaluate(input_fn = eval_input_fn, hooks = None)
    print("val results:", eval_results)
    file_obj.write("Medium:" + " " + str(eval_results['accuracy']) + " " + str(eval_results['loss']) + "\n")

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x = {"x": X_val_high},
            y = y_val,
            batch_size = 16,
            num_epochs = 1,
            shuffle = False) 
    eval_results = SSRN_classifier.evaluate(input_fn = eval_input_fn, hooks = None)
    print("val results:", eval_results)
    file_obj.write("High:" + " " + str(eval_results['accuracy']) + " " + str(eval_results['loss']) + "\n")

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x = {"x": X_val_original},
            y = y_val,
            batch_size = 16,
            num_epochs = 1,
            shuffle = False)
    eval_results = SSRN_classifier.evaluate(input_fn = eval_input_fn, hooks = None)
    print("val results:", eval_results)
    file_obj.write("Original:" + " " + str(eval_results['accuracy']) + " " + str(eval_results['loss']) + "\n")


    file_obj.close()

if __name__ == "__main__":
  tf.app.run()
