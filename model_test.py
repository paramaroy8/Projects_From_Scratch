'''
The test function will receive the user inputs and call necessary functions to process and generate output.
'''

from build_config import config_builder

from cnn_forward_pass import forward_pass



def test(architecture, image):
    """
    Initialize parameters based on the user-defined configuration in architecture variable using config_builder 
    function, and then use forward pass to build model and implement forward pass.
    
    Input Args:
                architecture: user defined architecture configuration
    
    
    """
    input_channels = image.shape[-1]
    
    seq = config_builder(architecture, input_channels)
    
    prediction, predicted_class_score, all_probs = forward_pass(seq, image)
    
    print(f'all probability: {all_probs}\n\npredicted class(es): {prediction} with score = {predicted_class_score}')