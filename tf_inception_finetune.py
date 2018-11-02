"""
Example TensorFlow script for finetuning a VGG model on your own data.
Uses tf.contrib.data module which is in release v1.2
Based on PyTorch example from Justin Johnson
(https://gist.github.com/jcjohnson/6e41e8512c17eae5da50aebef3378a4c)
Required packages: tensorflow (v1.2)
Download the weights trained on ImageNet for VGG:
```
wget http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz
tar -xvf vgg_16_2016_08_28.tar.gz
rm vgg_16_2016_08_28.tar.gz
```
For this example we will use a tiny dataset of images from the COCO dataset.
We have chosen eight types of animals (bear, bird, cat, dog, giraffe, horse,
sheep, and zebra); for each of these categories we have selected 100 training
images and 25 validation images from the COCO dataset. You can download and
unpack the data (176 MB) by running:
```
wget cs231n.stanford.edu/coco-animals.zip
unzip coco-animals.zip
rm coco-animals.zip
```
The training data is stored on disk; each category has its own folder on disk
and the images for that category are stored as .jpg files in the category folder.
In other words, the directory structure looks something like this:
coco-animals/
  train/
    bear/
      COCO_train2014_000000005785.jpg
      COCO_train2014_000000015870.jpg
      [...]
    bird/
    cat/
    dog/
    giraffe/
    horse/
    sheep/
    zebra/
  val/
    bear/
    bird/
    cat/
    dog/
    giraffe/
    horse/
    sheep/
    zebra/
"""

import argparse
import os
import random

import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.python.slim.nets.inception_v3 as icp_raw
# import tensorflow.contrib.
import pretrained_model.models_inception_v3 as icp
from pretrained_model.models_inception_preprocessing import preprocess_for_train_label, preprocess_for_val_label


parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', default='coco-animals/train')
parser.add_argument('--log_dir', default='log/inception')
parser.add_argument('--val_dir', default='coco-animals/val')
parser.add_argument('--model_path', default='vgg_16.ckpt', type=str)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--out_fea', default=512, type=int, )
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--num_epochs1', default=10, type=int)
parser.add_argument('--num_epochs2', default=10, type=int)
parser.add_argument('--learning_rate1', default=1e-3, type=float)
parser.add_argument('--learning_rate2', default=1e-5, type=float)
parser.add_argument('--dropout_keep_prob', default=0.5, type=float)
parser.add_argument('--weight_decay', default=5e-4, type=float)

VGG_MEAN = [123.68, 116.78, 103.94]


def list_images(directory):
    """
    Get all the images and labels in directory/label/*.jpg
    label_to_id: in inception, label start from 1 to 1001
    """
    print('Listing images ... (%s)' % directory)
    labels = os.listdir(directory)
    # Sort the labels so that training and validation get them in the same order
    labels.sort()

    files_and_labels = []
    for label in labels:
        for f in os.listdir(os.path.join(directory, label)):
            files_and_labels.append((os.path.join(directory, label, f), label))
    random.shuffle(files_and_labels)

    filenames, labels = zip(*files_and_labels)
    filenames = list(filenames)
    labels = list(labels)
    unique_labels = sorted(list(set(labels)))

    label_to_int = {}
    for i, label in enumerate(unique_labels):
        label_to_int[label] = i+1 # in inception, label start from 1 to 1001

    labels = [label_to_int[l] for l in labels]

    return filenames, labels


def _load_image(filename, label):
    '''
    1. Read image by the filename
    2. Convert it to tf.float32 and rescaled it into [0, 1]
    :param filename:
    :param label:
    :return:
    '''
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)  # (1)
    image = tf.cast(image_decoded, tf.float32)
    image = image / 255

    return image, label


def check_accuracy(sess, correct_prediction, is_training, dataset_init_op, prediction=None, label=None, end_points=None):
    """
    Check the accuracy of the model on either train or val (depending on dataset_init_op).
    """
    # Initialize the correct dataset
    sess.run(dataset_init_op)
    num_correct, num_samples = 0, 0
    step = 0
    while True:
        try:
            correct_pred, pre, l, logits = sess.run([correct_prediction, prediction, label, end_points['Logits']], {is_training: False})
            num_correct += correct_pred.sum()
            num_samples += correct_pred.shape[0]
            step += 1
            if step % 10 == 0:
                print('step%d: accuracy = %.5f' % (step, float(num_correct) / num_samples))
                print(pre, l)
                print(logits.shape)
        except tf.errors.OutOfRangeError:
            break

    # Return the fraction of datapoints that were correctly classified
    acc = float(num_correct) / num_samples
    return acc


def main(args):
    # Get the list of filenames and corresponding list of labels for training et validation
    train_filenames, train_labels = list_images(args.train_dir)
    val_filenames, val_labels = list_images(args.val_dir)
    assert set(train_labels) == set(val_labels),\
           "Train and val labels don't correspond:\n{}\n{}".format(set(train_labels),
                                                                   set(val_labels))

    num_classes = len(set(train_labels))

    # --------------------------------------------------------------------------
    # In TensorFlow, you first want to define the computation graph with all the
    # necessary operations: loss, training op, accuracy...
    # Any tensor created in the `graph.as_default()` scope will be part of `graph`
    graph = tf.Graph()
    with graph.as_default():
        # Standard preprocessing for VGG on ImageNet taken from here:
        # https://github.com/tensorflow/models/blob/master/research/slim/preprocessing/vgg_preprocessing.py
        # Also see the VGG paper for more details: https://arxiv.org/pdf/1409.1556.pdf

        # ----------------------------------------------------------------------
        # DATASET CREATION using tf.contrib.data.Dataset
        # https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/data

        # The tf.contrib.data.Dataset framework uses queues in the background to feed in
        # data to the model.
        # We initialize the dataset with a list of filenames and labels, and then apply
        # the preprocessing functions described above.
        # Behind the scenes, queues will load the filenames, preprocess them with multiple
        # threads and apply the preprocessing in parallel, and then batch the data

        # Training dataset
        train_dataset = tf.data.Dataset.from_tensor_slices((train_filenames, train_labels))
        train_dataset = train_dataset.map(_load_image, num_parallel_calls=args.num_workers)
        train_dataset = train_dataset.map(preprocess_for_train_label, num_parallel_calls=args.num_workers)
        train_dataset = train_dataset.shuffle(buffer_size=10*args.batch_size*args.num_workers)  # don't forget to shuffle
        batched_train_dataset = train_dataset.batch(args.batch_size)
        batched_train_dataset = batched_train_dataset.prefetch(args.batch_size)

        # Validation dataset
        val_dataset = tf.data.Dataset.from_tensor_slices((val_filenames, val_labels))
        val_dataset = val_dataset.map(_load_image, num_parallel_calls=args.num_workers)
        val_dataset = val_dataset.map(preprocess_for_val_label, num_parallel_calls=args.num_workers)
        batched_val_dataset = val_dataset.batch(args.batch_size)
        # batched_val_dataset = batched_val_dataset.prefetch(args.batch_size)


        # Now we define an iterator that can operator on either dataset.
        # The iterator can be reinitialized by calling:
        #     - sess.run(train_init_op) for 1 epoch on the training set
        #     - sess.run(val_init_op)   for 1 epoch on the valiation set
        # Once this is done, we don't need to feed any value for images and labels
        # as they are automatically pulled out from the iterator queues.

        # A reinitializable iterator is defined by its structure. We could use the
        # `output_types` and `output_shapes` properties of either `train_dataset`
        # or `validation_dataset` here, because they are compatible.
        iterator = tf.data.Iterator.from_structure(batched_train_dataset.output_types,
                                                   batched_train_dataset.output_shapes)
        images, labels = iterator.get_next()

        train_init_op = iterator.make_initializer(batched_train_dataset)
        val_init_op = iterator.make_initializer(batched_val_dataset)

        # Indicates whether we are in training or in test mode
        is_training = tf.placeholder(tf.bool)

        # ---------------------------------------------------------------------
        # Now that we have set up the data, it's time to set up the model.
        # For this example, we'll use VGG-16 pretrained on ImageNet. We will remove the
        # last fully connected layer (fc8) and replace it with our own, with an
        # output size num_classes=8
        # We will first train the last layer for a few epochs.
        # Then we will train the entire model on our dataset for a few epochs.

        # Get the pretrained model, specifying the num_classes argument to create a new
        # fully connected replacing the last one, called "vgg_16/fc8"
        # Each model has a different architecture, so "vgg_16/fc8" will change in another model.
        # Here, logits gives us directly the predicted scores we wanted from the images.
        # We pass a scope to initialize "vgg_16/fc8" weights with he_initializer

        with slim.arg_scope(icp.inception_v3_arg_scope(weight_decay=args.weight_decay)):
            logits, end_points = icp.inception_v3(images, num_classes=num_classes+1, is_training=is_training,
                                                  dropout_keep_prob=args.dropout_keep_prob, create_aux_logits=False,
                                                  feature_dim=args.out_fea)

        # icp = nets.inception_v3
        # with slim.arg_scope(icp.i(weight_decay=args.weight_decay)):
        #     logits, _ = icp.vgg_16(images, num_classes=num_classes, is_training=is_training,
        #                            dropout_keep_prob=args.dropout_keep_prob)

        # Specify where the model checkpoint is (pretrained weights).
        model_path = args.model_path
        assert(os.path.isfile(model_path))

        # Restore only the layers up to fc7 (included)
        # Calling function `init_fn(sess)` will load all the pretrained weights.
        variables_to_restore = tf.contrib.framework.get_variables_to_restore()
            # exclude=['InceptionV3/Logits/final_conv'])
        # print(variables_to_restore) # print the variables to restore
        init_fn = tf.contrib.framework.assign_from_checkpoint_fn(model_path, variables_to_restore)

        # Initialization operation from scratch for the new "fc8" layers
        # `get_variables` will only return the variables whose name starts with the given pattern
        # final_conv_variables = tf.contrib.framework.get_variables('final_conv')
        final_conv_variables = slim.get_variables('InceptionV3/Logits/Conv2d_1c_1x1')
        print(final_conv_variables) # print the variables of final conv
        # final_conv_init = tf.variables_initializer(final_conv_variables)

        # ---------------------------------------------------------------------
        # Using tf.losses, any loss is added to the tf.GraphKeys.LOSSES collection
        # We can then call the total loss easily
        tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        loss = tf.losses.get_total_loss()
        tf.summary.scalar('loss', loss)

        # summaries合并
        merged = tf.summary.merge_all()

        # First we want to train only the reinitialized last layer fc8 for a few epochs.
        # We run minimize the loss only with respect to the fc8 variables (weight and bias).
        part_optimizer = tf.train.GradientDescentOptimizer(args.learning_rate1)
        part_train_op = part_optimizer.minimize(loss, var_list=final_conv_variables)

        # Then we want to finetune the entire model for a few epochs.
        # We run minimize the loss only with respect to all the variables.
        full_optimizer = tf.train.GradientDescentOptimizer(args.learning_rate2)
        full_train_op = full_optimizer.minimize(loss)

        # Evaluation metrics
        prediction = tf.to_int32(tf.argmax(logits, 1))
        correct_prediction = tf.equal(prediction, labels)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        tf.get_default_graph().finalize()

    # --------------------------------------------------------------------------
    # Now that we have built the graph and finalized it, we define the session.
    # The session is the interface to *run* the computational graph.
    # We can call our training operations with `sess.run(train_op)` for instance
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(graph=graph, config=config) as sess:
        init_fn(sess)  # load the pretrained weights
        # sess.run(final_conv_init)  # initialize the new fc8 layer

        train_writer = tf.summary.FileWriter(args.log_dir + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(args.log_dir + '/test')

        # Update only the last layer for a few epochs.
        max_step = 0
        for epoch in range(args.num_epochs1):
            # Run an epoch over the training data.
            print('######### Starting epoch %d / %d' % (epoch + 1, args.num_epochs1))
            # Here we initialize the iterator with the training set.
            # This means that we can go through an entire epoch until the iterator becomes empty.
            sess.run(train_init_op)

            step = 0
            while True:
                try:
                    _, lss, summary, cpre = sess.run([part_train_op, loss, merged, correct_prediction], {is_training: True})
                    step += 1
                    # print(im[0,:10, :10])
                    # print(conv.shape)
                    train_writer.add_summary(summary, step+epoch*max_step)
                    if step % 10 == 0:
                        print('%d is finished. loss = %.6f; step accuracy = %.2f' % (step, lss, cpre.sum()/cpre.shape[0]))
                except tf.errors.OutOfRangeError:
                    max_step = max(max_step, step)
                    break

            # Check accuracy on the train and val sets every epoch.
            train_acc = check_accuracy(sess, correct_prediction, is_training, train_init_op)
            val_acc = check_accuracy(sess, correct_prediction, is_training, val_init_op)
            print('Train accuracy: %f' % train_acc)
            print('Val accuracy: %f\n' % val_acc)
        train_writer.close()


        # Train the entire model for a few more epochs, continuing with the *same* weights.
        for epoch in range(args.num_epochs2):
            print('######　Starting epoch %d / %d' % (epoch + 1, args.num_epochs2))
            sess.run(train_init_op)
            while True:
                try:
                    _ = sess.run(full_train_op, {is_training: True})
                except tf.errors.OutOfRangeError:
                    break

            # Check accuracy on the train and val sets every epoch
            train_acc = check_accuracy(sess, correct_prediction, is_training, train_init_op)
            val_acc = check_accuracy(sess, correct_prediction, is_training, val_init_op)
            print('Train accuracy: %f' % train_acc)
            print('Val accuracy: %f\n' % val_acc)


def test_raw_inception(args):
    # Get the list of filenames and corresponding list of labels for training et validation
    val_filenames, val_labels = list_images(args.val_dir)

    num_classes = len(set(val_labels))
    print('num_class = ', num_classes)

    # --------------------------------------------------------------------------
    # In TensorFlow, you first want to define the computation graph with all the
    # necessary operations: loss, training op, accuracy...
    # Any tensor created in the `graph.as_default()` scope will be part of `graph`
    graph = tf.Graph()
    with graph.as_default():
        # Standard preprocessing for VGG on ImageNet taken from here:
        # https://github.com/tensorflow/models/blob/master/research/slim/preprocessing/vgg_preprocessing.py
        # Also see the VGG paper for more details: https://arxiv.org/pdf/1409.1556.pdf

        # Validation dataset
        val_dataset = tf.data.Dataset.from_tensor_slices((val_filenames, val_labels))
        val_dataset = val_dataset.map(_load_image, num_parallel_calls=args.num_workers) # map (image, label) to (image, label)
        val_dataset = val_dataset.map(preprocess_for_val_label, num_parallel_calls=args.num_workers)
        batched_val_dataset = val_dataset.batch(args.batch_size)


        # Now we define an iterator that can operator on either dataset.
        # The iterator can be reinitialized by calling:
        #     - sess.run(train_init_op) for 1 epoch on the training set
        #     - sess.run(val_init_op)   for 1 epoch on the valiation set
        # Once this is done, we don't need to feed any value for images and labels
        # as they are automatically pulled out from the iterator queues.

        # A reinitializable iterator is defined by its structure. We could use the
        # `output_types` and `output_shapes` properties of either `train_dataset`
        # or `validation_dataset` here, because they are compatible.
        iterator = tf.data.Iterator.from_structure(batched_val_dataset.output_types,
                                                   batched_val_dataset.output_shapes)
        images, labels = iterator.get_next()

        val_init_op = iterator.make_initializer(batched_val_dataset)

        # Indicates whether we are in training or in test mode
        is_training = tf.placeholder(tf.bool)

        # ---------------------------------------------------------------------
        # Now that we have set up the data, it's time to set up the model.
        # For this example, we'll use VGG-16 pretrained on ImageNet. We will remove the
        # last fully connected layer (fc8) and replace it with our own, with an
        # output size num_classes=8
        # We will first train the last layer for a few epochs.
        # Then we will train the entire model on our dataset for a few epochs.

        # Get the pretrained model, specifying the num_classes argument to create a new
        # fully connected replacing the last one, called "vgg_16/fc8"
        # Each model has a different architecture, so "vgg_16/fc8" will change in another model.
        # Here, logits gives us directly the predicted scores we wanted from the images.
        # We pass a scope to initialize "vgg_16/fc8" weights with he_initializer

        with slim.arg_scope(icp.inception_v3_arg_scope()):
            logits, end_points = icp.inception_v3(images, is_training=is_training, num_classes=num_classes+1,
                                                  dropout_keep_prob=args.dropout_keep_prob, reuse=tf.AUTO_REUSE,
                                                  create_aux_logits=False)

        # Specify where the model checkpoint is (pretrained weights).
        model_path = args.model_path
        assert(os.path.isfile(model_path))

        # Restore only the layers up to fc7 (included)
        # Calling function `init_fn(sess)` will load all the pretrained weights.
        variables_to_restore = tf.contrib.framework.get_variables_to_restore(exclude=[])
        print(variables_to_restore)
        # print(variables_to_restore) # print the variables to restore
        init_fn = tf.contrib.framework.assign_from_checkpoint_fn(model_path, variables_to_restore)

        # ---------------------------------------------------------------------
        # Using tf.losses, any loss is added to the tf.GraphKeys.LOSSES collection
        # We can then call the total loss easily
        tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        loss = tf.losses.get_total_loss()
        tf.summary.scalar('loss', loss)

        # Evaluation metrics
        prediction = tf.to_int32(tf.argmax(logits, 1))
        correct_prediction = tf.equal(prediction, labels)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        tf.get_default_graph().finalize()

    # --------------------------------------------------------------------------
    # Now that we have built the graph and finalized it, we define the session.
    # The session is the interface to *run* the computational graph.
    # We can call our training operations with `sess.run(train_op)` for instance
    with tf.Session(graph=graph) as sess:
        init_fn(sess)  # load the pretrained weights
        # sess.run(final_conv_init)  # initialize the new fc8 layer

        val_acc = check_accuracy(sess, correct_prediction, is_training, val_init_op, prediction, labels, end_points)
        print('Val accuracy: %f\n' % val_acc)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
    # test_raw_inception(args)
