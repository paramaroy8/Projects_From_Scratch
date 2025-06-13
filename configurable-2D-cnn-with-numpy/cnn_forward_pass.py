'''
The forward_pass function will take the configured matrix that contains the layer details based on user configuration, build the CNN model to perform the CNN forward pass.
'''

import numpy as np
from cnn_building_blocks import *


def forward_pass(seq, image):
    """
    Input Args:
        seq: a list of dictionaries that contains layer name with initial details in the user-defined order
        image: one image or a set of images
    
    Output Args:
        all_probs: calculated probability of all classes {dictionary or matrix}
        prediction: predicted class(es) {list}
        predicted_class_score: score(s) of predicted class(es) {list or dictionary}
        
    """
    # model_matrix starts as an image and later becomes the prediction after passing through various layers.
    
    model_matrix = image
    
    
    for d in seq:
        # Layer Details are stored as a dictionary.
        layer_details = d["layer_details"]
        
        if d["layer"] == "conv":
            
            # build convolution layer with the layer details           
             
            
            # initialize it with image since that is the first input to the convolutional layers
            stacked_fmaps = []
            
            # process padding
            # first check if padding is defined or needed, then complete the processing it
            # automatically setting padding 0 if not defined
            
            if layer_details.get("padding", 0):
                pad = layer_details["padding"]
                pad_mode = layer_details.get("pad_mode", "edge")
                
                model_matrix = np.pad(model_matrix,
                                     ((pad, pad), # top, bottom; padding across height
                                     (pad, pad), # left, right; padding across width
                                     (0, 0)), # no padding for channels
                                     pad_mode # for simplicity, it will be edge
                                    ) 
            
            # perform convolution for each filter, store individual f_map and stack them for the next layer
            for f in range(layer_details["n_filters"]):
                
                # store the feature map after convolution layer
                feature_map = conv_2dim(model_matrix, 
                                    layer_details["kernel"][f], 
                                    layer_details["stride"], 
                                    layer_details["bias"]
                                   )
                # perform leaky relu (or other activation )to ignore certain value
                feature_map = leaky_relu(feature_map, layer_details["relu_constant"])
                
                # store each feature map
                stacked_fmaps.append(feature_map)
            
            # stack all feature maps to be used as input for next layer
            stacked_fmaps = np.stack(stacked_fmaps, axis=0)
            
            '''
            After convlution layer, stacking places the channel before height and width, 
            making the order: channel, height, width.
            But we want to keep it in the order: height, width, channel.
            
            So, after each convolution, we will transpose it in the desired order.
            '''
            stacked_fmaps = np.transpose(stacked_fmaps, (1, 2, 0))
            
            # store it to the model_matrix
            model_matrix = np.copy(stacked_fmaps)
                      
            
        
        elif d["layer"] == "pool":
            # build pooling layer with the layer details
            
            # process padding
            if layer_details.get("padding", 0):
                pad = layer_details["padding"]
                pad_mode = layer_details.get("pad_mode", "edge")
                
                model_matrix = np.pad(model_matrix,
                                     ((pad, pad), # top, bottom; padding across height
                                     (pad, pad), # left, right; padding across width
                                     (0, 0)), # no padding for channels
                                     pad_mode # for simplicity, it will be edge
                                    )
                
            # apply max pooling
            # if the pool_size is given as an integer, then pooling_kernel is a square matrix
            # else, pool_size dimensions will be stored separately
            
            if isinstance(layer_details["pool_size"], int):
                pool_height = pool_width = layer_details["pool_size"]
            else:
                pool_height, pool_width = layer_details["pool_size"]
            
            # check what type of pooling is chosen
            if layer_details["type"] == "max":
                # store the feature map after pooling
                model_matrix = max_pooling_2dim(model_matrix, (pool_height, pool_width), layer_details["stride"])
        
        
        
        elif d["layer"] == "fc":
            # build fully connected layer with the layer details
            
            # first flatten the model_matrix
            model_matrix = flatten_feature(model_matrix)
            
            # generate weight matrix
            M = layer_details["neurons"]  # number of neurons in the FC layer defined by the user
            N = model_matrix.shape[0]     # number of elements in the flattened vector of shape (N, )
            
            
            weight_matrix = np.ones((M, N)) * 0.01
            
            # check the type of fc layer if it is hidden layer or output layer
            if layer_details["fc_mode"] == "hidden":
                model_matrix = fc_layer(model_matrix, weight_matrix, layer_details["bias_vector"])
                
                '''
                In case of the hidden layer, leaky_relu is used as activation.
                '''
                if layer_details["activation"] == "leaky_relu":
                    model_matrix = leaky_relu(model_matrix, layer_details["activation_constant"])
            else:
                # output layer
                
                # activation is already handled in this layer
                prediction, predicted_class_prob, all_probs = output_layer(model_matrix, 
                                                                           weight_matrix,
                                                                           layer_details["bias_vector"],
                                                                           layer_details["activation"]
                                                                          ) 
        
        else:
            raise ValueError("Undefined Layer")
                
    
    return prediction, predicted_class_prob, all_probs
