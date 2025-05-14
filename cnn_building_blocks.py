import numpy as np

def conv_2dim(conv_matrix, kernel, stride, bias):
    """
    Input args:
    
        • conv_matrix: matrix on which the convolution will be taken place by a filter(kernel)
        • conv_matrix order: height, width, channel
        • kernel: filter matrix that will slide over the conv_matrix and perform convolution
        • kernel order: height, width, channel
        • stride: custom defined; value >= 1
        • bias: negative threshold 
        
    Return:
        
        • f_map: feature map 2D matrix with order: height, width
    """
    
    conv_height, conv_width, conv_channel = conv_matrix.shape
    k_height, k_width, k_channel = kernel.shape
    
    # define feature map dimension
    feature_height = int(np.floor((conv_height - k_height) / stride)) + 1
    feature_width = int(np.floor((conv_width - k_width) / stride)) + 1
    
    feature_map = np.zeros((feature_height, feature_width))
    
    for i in range(feature_height):
        for j in range(feature_width):
            # extract the portion of the conv_matrix for convolution
            patch = conv_matrix[i*stride : i*stride + k_height, j*stride : j*stride + k_width, :]
            
            # calculate weighted sum
            feature_map[i, j] = np.sum(patch * kernel) 
            
    # implement addition of bias
    feature_map += bias
    
    return feature_map


def max_pooling_2dim(padded_f_maps, pooling_dim=(2, 2), stride=1):
    """
    Assuming the input argument feature map already handles necessary padding.
    """
    
    f_map_height, f_map_width, f_map_channel = padded_f_maps.shape
    
    '''
    If stride chosen by the user is bigger than the feature map height or width, then max pooling cannot be implemented.
    '''
    
    if stride > f_map_height or stride > f_map_width:
        raise ValueError("Stride is bigger than Feature Map dimensions.")
        
    pool_height, pool_width = pooling_dim
    
    new_fmap_height = int(np.floor((f_map_height - pool_height) / stride)) + 1
    new_fmap_width = int(np.floor((f_map_width - pool_width) / stride)) + 1
    
    # number of channels remain the same
    new_fmap = np.zeros((new_fmap_height, new_fmap_width, f_map_channel))
    
    for i in range(new_fmap_height):
        for j in range(new_fmap_width):
            for k in range(f_map_channel):
                pool_patch = padded_f_maps[i*stride:i*stride+pool_height, j*stride:j*stride+pool_width, k]
                
                new_fmap[i, j, k] = np.max(pool_patch)
    
    return new_fmap


def flatten_feature(feature_map):
    """
    Input Arg:
                Multi channel feature map
    Return:
                1D vector
    """
    return feature_map.flatten()


def fc_layer(core_vector, weight_matrix, bias_vector):
    """
    Input Args:
                core_vector : 1D vector, shape = (N, ), where N = number of elements in the flattened array
                weight_matrix : 2D matrix, shape = (M, N), where, each row of M is used in corresponding neuron
                bias_vector : 1D vector, shape = (M, ), where each row value is used in corresponding neuron
    
    Return:
                fc_output : 1D vector, shape = (M, ), each element is corresponding output from each neuron    
    """
    
    fc_output = np.dot(weight_matrix, core_vector) + bias_vector
    
    return fc_output


def my_softmax(raw_scores):
    """
    Input Args:
                raw_scores: takes raw scores of last fully connected layer as input
        
    Return:
                all_probs : a numpy array of probabilities corresponding to the raw_scores
    """
    
    # np.exp(numpy_array) computes exponential of the entire array
    
    '''
    We substract maximum value to ensure numerical stability after using exponential.
    
    Subtraction makes sure the value is 0 or less so that after expoentiation, the output is a small value, which
    is easy for computer to handle.
    
    Subtraction doesn't change the relationship between values. e.g. if A > B, then, A-5 > B-5 is still valid.
    '''
    
    exp_scores = np.exp(raw_scores - np.max(raw_scores))
    
    # np.sum(numpy_array) is designed to optimally for numpy arrays
    all_sum = np.sum(exp_scores)
    
    prob = exp_scores / all_sum
    
    return prob


def output_layer(prev_output, weight_matrix, bias_vector, activation):
    """
    Input Args:
                prev_output : output from the previous(last) hidden layer, shape : (M, ), also known as logits.
                            M is the number of neurons in the previous layer
                weight_matrix : shape = (M, N)
                               N is the number of neurons in this layer
                bias_vector : shape = (N, )
                activation : softmax or any other activation function
                              
                
    Return:
                highest_pos : the class with the highest score is returned, scalar value
                highest_prob : the class with the highest probability
                all_probs : probabilities of all classes   
    """   
    
    
    fc_output = np.dot(weight_matrix, prev_output) + bias_vector # weighted sum and bias vector
    
    
    highest_score = np.max(fc_output)
    
    '''
    np.where() returns a tuple because it is designed to handle multi-dimensional arrays. 
    It returns a tuple of arrays.
    
    To handle 1D array, just add [0] with the fucntion.
    '''
    
    # find all positions with this highest score 
    # if multiple classes have highest value, all of them will be stored as array in predicted_class variable
    highest_pos = np.where(fc_output == highest_score)[0]
    
    if activation == "softmax":
        all_probs = my_softmax(fc_output)
    
    highest_prob = all_probs[highest_pos[0]]
    
    return highest_pos, highest_prob, all_probs


def relu(m):
    """
    Elementwise compares the vector or matrix or scalar m with 0, and replaces with the maximum element.   
    We can also compare with values other than 0.
    
    Input Args:
        m: 1D or multi-dimensional matrix or vector or scalar
    
    Return:
        elementwise compared matrix or vector or scalar
    """
    return np.maximum(0, m)


def leaky_relu(m, alpha=0.01):
    """
    Elementwise compares the vector or matrix or scalar with 0.
    If an element is greater than 0, the original value remains.
    Else, this function replaces it with alpha times original value. This allows a small value to pass through.
    Alpha can be user defined. The default is 0.01.
    By setting alpha = 0, this function acts like ReLU.
    
    Input Args:
                1D vector or 2D matrix
    
    Return:
                1D vector, elementwise compared with 0

    """
    
    new_m = np.where(m < 0, alpha * m, m)
    
    return new_m