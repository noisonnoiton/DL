import tensorflow as tf
from Utils.data_utils import *

class SqueezeNet(object):
    ''' Load a pretrained SqueezeNet model '''
    def __init__(self, sess):
        maybe_download_and_extract("http://cs231n.stanford.edu/squeezenet_tf.zip")
        
        #Load tensorflow graph and weights
        saver =  tf.train.import_meta_graph('./Utils/squeezenet.ckpt.meta')
        saver.restore(sess, "./Utils/squeezenet.ckpt")
        graph = tf.get_default_graph()
        
        # Define model variables 
        self.inputs = graph.get_tensor_by_name("input_image:0")
        self.targets = tf.placeholder('int32', shape=[None], name='labels')
        self.outputs = graph.get_tensor_by_name("classifier/Reshape:0")