# --------------------- Predict function   -------------------------------- #

import tensorflow as tf


def predict(test_inputs, nn, sess):

    # Initialization
    test_feed_dict = {nn.inputs : test_inputs}

    # Run session through the computational graph. 
    test_results = sess.run(nn.predictions, feed_dict=test_feed_dict)

    return test_results

        


    