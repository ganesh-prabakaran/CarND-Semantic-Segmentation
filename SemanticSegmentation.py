
# coding: utf-8

# In[1]:


import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
from tqdm import *


# In[2]:


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


# In[3]:


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
    
    # Load saved model
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
   
    ## Restore tensors from VGG16
    graph = tf.get_default_graph()
    vgg_input_tensor = graph.get_tensor_by_name(vgg_input_tensor_name)
    vgg_keep_prob_tensor = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    vgg_layer3_out_tensor = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    vgg_layer4_out_tensor = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    vgg_layer7_out_tensor = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    
    return vgg_input_tensor, vgg_keep_prob_tensor, vgg_layer3_out_tensor, vgg_layer4_out_tensor, vgg_layer7_out_tensor
tests.test_load_vgg(load_vgg, tf)


# In[4]:


TRANSFER_LEARNING_MODE = True


# In[5]:


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    with tf.name_scope("decoder"):
        # TODO: Implement function
        if TRANSFER_LEARNING_MODE:
            #Prevent gradient from changing in back propagation
            vgg_layer7_out = tf.stop_gradient(vgg_layer7_out)
            vgg_layer4_out = tf.stop_gradient(vgg_layer4_out)
            vgg_layer3_out = tf.stop_gradient(vgg_layer3_out)

        # Upsample

        vgg_layer7_out_1x = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, strides = (1,1), padding = "SAME",        kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3), kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),         activation = tf.nn.relu, name = "DECON_vgg_layer7_out_1x")

        vgg_layer7_out_2x = tf.layers.conv2d_transpose(vgg_layer7_out_1x, num_classes, 4, strides = (2,2), padding = "SAME",        kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3), kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),         activation = tf.nn.relu, name = "DECON_vgg_layer7_out_2x")

        vgg_layer4_out_1x = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, strides = (1,1), padding = "SAME",        kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3), kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),         activation = tf.nn.relu, name = "DECON_vgg_layer4_out_1x")

        vgg_layer4_7_combined = tf.add(vgg_layer7_out_2x, vgg_layer4_out_1x, name = "DECON_vgg_layer4_7_combined")

        vgg_layer4_out_4x = tf.layers.conv2d_transpose(vgg_layer4_7_combined, num_classes, 4, strides = (2,2), padding = "SAME",        kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3), kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),         activation = tf.nn.relu, name = "DECON_vgg_layer7_out_4x")

        vgg_layer3_out_1x = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, strides = (1,1), padding = "SAME",        kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3), kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),         activation = tf.nn.relu, name = "DECON_vgg_layer3_out_1x")

        vgg_layer3_7_combined = tf.add(vgg_layer4_out_4x, vgg_layer3_out_1x, name = "DECON_vgg_layer3_7_combined")

        nn_last_layer = tf.layers.conv2d_transpose(vgg_layer3_7_combined, num_classes, 16, strides = (8,8), padding = "SAME",        kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3),kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),         activation = tf.nn.relu, name = "DECON_nn_last_layer")
    

    return nn_last_layer
tests.test_layers(layers)


# In[6]:


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, accuracy_op, cross_entropy_loss)
    """
    # TODO: Implement function
    
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = tf.reshape(correct_label, (-1, num_classes))
    
    #Compute Loss 
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels)
        
    #Perform accuracy operation
    prediction_comp = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy_op = tf.reduce_mean(tf.cast(prediction_comp, tf.float32), name="accuracy_op")
    
    loss_operation = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    TRANSFER_LEARNING_MODE = False
    if TRANSFER_LEARNING_MODE:
        trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "decoder")
        training_op = optimizer.minimize(loss_operation, var_list = trainable_variables, name = "training_op")
    else:
        training_op = optimizer.minimize(loss_operation, name = "training_op")
    return logits, training_op, accuracy_op, loss_operation
tests.test_optimize(optimize)


# In[7]:


def evaluate(image_shape, sess, input_image, correct_label,              keep_prob, loss_op, accuracy_op, is_training):
    data_folder = '.\data'
    data_generator_function = helper.gen_batch_function(data_folder, image_shape)
    batch_size = 8
    data_generator = data_generator_function(batch_size)
    num_examples = len(image_paths)
    total_loss = 0
    total_acc = 0
    processed_flg = False
    processed_cnt = 0
    while (~processed_flg):
        processed_cnt += batch_size
        if processed_cnt >= num_examples:
            processed_flg = True
            batch_size = processed_cnt - num_examples
        X_batch, y_batch = next(data_generator)
        loss, accuracy = sess.run([loss_op, accuracy_op], 
                                  feed_dict={input_image: X_batch, correct_label: y_batch,
                                             keep_prob: 1.0, is_training :False})
        total_loss += (loss * X_batch.shape[0])
        total_acc += (accuracy * X_batch.shape[0])
    return total_loss/num_examples, total_acc/num_examples
    


# In[8]:


def save_model(sess):
    if "saved_model" in os.listdir(os.getcwd()):
        shutil.rmtree("./saved_model")
    
    builder = tf.saved_model.builder.SavedModelBuilder("./saved_model")
    builder.add_meta_graph_and_variables(sess, ["vgg16"])
    builder.save()


# In[9]:


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    #tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        epochs = 2
        batch_size = 5
        
        # TF placeholders
        
        correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes], name = 'correct_label')
        learning_rate = tf.placeholder(tf.float32, name = 'learning_rate')
        
        input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)
        
        nn_last_layer = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)
        
        logits, train_op, cross_entropy_loss = optimize(nn_last_layer, correct_label, learning_rate, num_classes)
        
        
        # TODO: Train NN using the train_nn function

        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image, correct_label, keep_prob, learning_rate)
        
        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video


# In[ ]:


if __name__ == '__main__':
    run()
