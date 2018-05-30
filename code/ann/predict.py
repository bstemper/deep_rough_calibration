# --------------------- Predict function   -------------------------------- #

import tensorflow as tf


def predict(test_inputs, nn, sess):

    # Initialization
    test_feed_dict = {nn.inputs : test_inputs,
    				  nn.training_phase: False}

    # Run session through the computational graph. 
    predictions, jac = sess.run([nn.predictions, nn.jac], 
                                feed_dict=test_feed_dict)

    return predictions, jac[0]

        


    