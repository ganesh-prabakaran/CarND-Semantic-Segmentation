import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Constant parameters

EPOCH = 40
BATCH_SIZE = 5
LEARNING_RATE = 0.001
KEEP_PROB = 0.8
BETA = 0.0001

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    
    #load model from the path provided
    tf.saved_model.loader.load(sess,[vgg_tag],vgg_path)
    
    #get graph of the loaded model
    savedGraph = tf.get_default_graph()
    
    #get tensors for input , keep_prob ,layre3 output , layer4 output , layer7 output
    image_input= savedGraph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob=savedGraph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out=savedGraph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out=savedGraph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out=savedGraph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    
    return image_input, keep_prob, layer3_out, layer4_out, layer7_out

tests.test_load_vgg(load_vgg, tf)

def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    
    #adjust the depth of the layer 7 output to required final classes
    vgg_layer7_out_1x = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, strides = (1,1), padding = "SAME", 
        kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3), kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
        name = "DECON_vgg_layer7_out_1x")

        
    #upsample 2x times 
    vgg_layer7_out_2x = tf.layers.conv2d_transpose(vgg_layer7_out_1x, num_classes, 4, strides = (2,2), padding = "SAME",
        kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3), kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
        name = "DECON_vgg_layer7_out_2x")                                                

    #use skip connection to add layer 4 output with above layer
    #but make sure to have same depth by using a 1x1 conv

    vgg_layer4_out_1x = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, strides = (1,1), padding = "SAME", 
        kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3), kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
        activation = tf.nn.relu, name = "DECON_vgg_layer4_out_1x")

    vgg_layer4_7_combined = tf.add( vgg_layer4_out_1x, vgg_layer7_out_2x, name = "DECON_vgg_layer4_7_combined")
    
    #upsample 2x times 
    vgg_layer4_out_4x = tf.layers.conv2d_transpose(vgg_layer4_7_combined, num_classes, 4, strides = (2,2), padding = "SAME",
        kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3), kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
        name = "DECON_vgg_layer7_out_4x")
        
    #use skip connection to add layer 3 output with above
    #but make sure to have same depth by using a 1x1 conv

    vgg_layer3_out_1x = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, strides = (1,1), padding = "SAME",
        kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3), kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
        name = "DECON_vgg_layer3_out_1x")
    
    vgg_layer3_7_combined = tf.add(vgg_layer4_out_4x, vgg_layer3_out_1x, name = "DECON_vgg_layer3_7_combined")
    
    ##upsample 8x times

    nn_last_layer = tf.layers.conv2d_transpose(vgg_layer3_7_combined, num_classes, 16, strides = (8,8), padding = "SAME",
        kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3),kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
        name = "DECON_nn_last_layer")
    
    
    return nn_last_layer

tests.test_layers(layers)

def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
 
    #reshape from 4d to 2d for ease in classification
    logits = tf.reshape(nn_last_layer, (-1, num_classes),name="modified_logits")
    correct_label = tf.reshape(correct_label, (-1,num_classes),name="modified_correct_label")
    
    #apply cross entrophy
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=correct_label)
    
    vars   = tf.trainable_variables()
    # Add regularization 
    lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name ]) * BETA
    cross_entropy_loss = tf.reduce_mean(cross_entropy + lossL2)


    #add adam optimizer
    opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
                                 
    #Perform optimization
    train_op = opt.minimize(
                     cross_entropy_loss, name="train_op"
                  )

    
    return logits, train_op, cross_entropy_loss


tests.test_optimize(optimize)

def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
   
    sess.run(tf.global_variables_initializer())
    
 
    for epoch in range(epochs):
        tot_loss=0.0000
        count=0
        for X_batch , y_batch in get_batches_fn(batch_size):
            count+=1
            loss, _ = sess.run([cross_entropy_loss, train_op], feed_dict={
                input_image: X_batch,
                correct_label: y_batch,
                keep_prob: 0.8 
            })
            tot_loss +=loss
        
        print ("EPOCH : " ,epoch, " Loss : " ,tot_loss/count ) 

 
        

tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)
    
    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)
   
    
    with tf.Session() as sess:
       
    # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # Build NN using load_vgg, layers, and optimize function
        correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes], name='modified_correct_label')
       
        input_layer, KEEP_PROB, layer3, layer4, layer7 = load_vgg(sess, vgg_path)
        
        nn_last_layer = layers(layer3, layer4, layer7, num_classes)
        
        logits, train_op, cross_entropy_loss= optimize(nn_last_layer, correct_label, LEARNING_RATE, num_classes)
       
        print (" Training started  - Epoch: ",EPOCH, " Batch size : " ,BATCH_SIZE , " Learning rate : ",LEARNING_RATE  )
        
        # Train NN using the train_nn function
        train_nn(sess, EPOCH, BATCH_SIZE, get_batches_fn, train_op, cross_entropy_loss, input_layer,correct_label, KEEP_PROB, LEARNING_RATE)    
        
        # Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, KEEP_PROB, input_layer)
        
if __name__ == '__main__':
    run()
