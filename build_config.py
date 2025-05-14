'''
The function in this file will create necessary initializations for the CNN model based on the user-defined configuration.
'''

import numpy as np
from cnn_building_blocks import *



def config_builder(architecture, input_channels):
    """
    Input Args:
                Takes in architecture configuration; a list of dictionaries
                input_channels: scalar value that defines how many channels each image will have
    
    Return:
                seq: a list of dictionaries, in the order preserved by the model architect
    """
    
    # initialize sequence list
    seq = []
    
    '''
    For the first convolution layer, the number of channels is equal to the input channel.
    After that, the each convlution kernel channel is based on the number of filters used in previous layer.
    '''
    current_channels = input_channels
    
    '''
    This function will only be used for initialization of the weights. 
    After that, all processes will take place inside the forward pass function.
    '''
    
    
    # loop through each dictionary in the list
    for d in architecture:
        if d["layer"] == "conv":
            '''
            If the kernel dimensions is given as an integer, then the kernel has same height and width.           
            But we also want to consider situations where kernel height ≠ kernel width.
            '''
            
            if isinstance(d["kernel_size"], int):
                kernel_height = kernel_width = d["kernel_size"]
            else:
                kernel_height, kernel_width = d["kernel_size"]
                          
            
            number_of_filters = d["filters"]
            
            conv_kernel = np.ones((number_of_filters, kernel_height, kernel_width, current_channels)) * 0.01
            
            bias = d.get("bias", 0.01) # if "bias" key is not defined, then a value of 0.01 is assigned
            relu_constant = d.get("relu_constant", 0.01)
            
                       
            conv_dict = {"kernel" : conv_kernel, 
                         "n_filters" : number_of_filters,
                         "stride": d["stride"], 
                         "bias": bias, 
                         "padding": d["padding"],
                         "pad_mode": d["pad_mode"],
                         "relu_constant": relu_constant,
                         "activation": d["activation"]
                        }
            
            seq.append({"layer": "conv", "layer_details": conv_dict})
            
            # update number of channels for the next convolutional layer
            current_channels = number_of_filters
        
        elif d["layer"] == "pool":
            
            pool_size = d["kernel_size"]
            stride = d.get("stride", 1) # if stride key is not defined, assume it to be 1
            padding = d.get("padding", 0)
            
            pool_dict = {"pool_size": pool_size,
                         "type": d["type"],
                         "stride": stride,
                         "padding": padding
                        }
            
            seq.append({"layer": "pool", "layer_details": pool_dict})
        
        elif d["layer"] == "fc":
            
            '''
            There are two types of FC layers: hidden and output.
            
            Each neuron has its own weights and biases.
            '''
            # process bias
            bias_vector = d.get("bias_vector", None) # if bias_vector is not defined, initialize the parameter as None
            
            if bias_vector is None:
                random_bias = np.random.rand(d["units"], ) * 0.01  # vector with small random values 
                zero_bias = np.zeros(d["units"], )                 # 0 vector
                
                bias_vector = random_bias if d["fc_mode"] == "output" else zero_bias
                
            # process activation
            activation = d.get("activation", None)
            
            if activation is None:
                activation = "softmax" if d["fc_mode"] == "output" else "leaky_relu"
                
            # process activation constant
            activation_constant = d.get("activation_constant", 0) # if activation constant is not define, assume it to be 0
                     
            fc_dict = {"neurons": d["units"],
                       "fc_mode": d["fc_mode"],
                       "bias_vector": bias_vector,
                       "activation": activation, # softmax for output layer, leaky_relu for hidden layer
                       "activation_constant": activation_constant # alpha for leaky relu
                      }
            
            seq.append({"layer": "fc", "layer_details": fc_dict})        
        
        else:
            raise ValueError(f'Unknown Layer : {d["layer"]}')
    
    return seq