import tensorflow as tf
# from nets.dssd_513 import DSSD
from nets.dssd_321 import DSSD
from datasets.cls_dict_gen import num_cls
import dssd_train, dssd_inference
from datetime import datetime
import numpy as np
print('USING TF 1.13 AUGMENTOR NOT TO BE USED ON SERVER')
print('USING TF 1.13 AUGMENTOR NOT TO BE USED ON SERVER')
print('USING TF 1.13 AUGMENTOR NOT TO BE USED ON SERVER')
print('USING TF 1.13 AUGMENTOR NOT TO BE USED ON SERVER')
print('USING TF 1.13 AUGMENTOR NOT TO BE USED ON SERVER')
from preprocessing import augmentor

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_bool('use_bn', True, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 100000, 'Number of steps to run trainer.')
flags.DEFINE_string('train_dir', 'data/imgs', 'Directory to put the training data.')
flags.DEFINE_string('test_dir', 'data/test', 'Directory to put the test data.')
flags.DEFINE_string('log_dir', f'logs/{datetime.now()}', 'Directory to put the log files.')
flags.DEFINE_integer(
    'img_size', 321, 'size of img')
flags.DEFINE_integer(
    'mb_size', 16, 'The number of samples in each batch.')
flags.DEFINE_integer(
    'ncls', num_cls(), 'Number of classes to use in the dataset.')
flags.DEFINE_integer(
    'nepochs', 100000, 'Number of epochs through the dataset.')
flags.DEFINE_bool(
    'eval', False, 'Eval model')
flags.DEFINE_bool(
    'restore', True, 'Restore model')
flags.DEFINE_bool(
    'training', False, 'Train model')

IMG_SIZE = (FLAGS.img_size, FLAGS.img_size)

train_record = './data/record/train.record'
test_record = './data/record/test.record'


#  Reads the tfrecord file and parses the input into ground thruths and minibatches
def tfrecord_input_fn(filenames, nepochs=FLAGS.nepochs, batch_size=FLAGS.mb_size, shuffle=False, training=False):

    def _input_fn(model):
        def _parse_fn(tf_record):
            features={
            'image/height': tf.FixedLenFeature([], tf.int64),
            'image/width': tf.FixedLenFeature([], tf.int64),
            'image/source_id': tf.FixedLenFeature([], tf.string),
            'image/encoded': tf.FixedLenFeature([], tf.string),
            'image/format': tf.FixedLenFeature([], tf.string),
            'image/filename': tf.FixedLenFeature([], tf.string),
            'image/object/bbox/xmin': tf.VarLenFeature(tf.float32),
            'image/object/bbox/xmax': tf.VarLenFeature(tf.float32),
            'image/object/bbox/ymin': tf.VarLenFeature(tf.float32),
            'image/object/bbox/ymax': tf.VarLenFeature(tf.float32),
            'image/object/class/label': tf.VarLenFeature(tf.int64)
            }

            features = tf.parse_single_example(tf_record, features)
            filename = features['image/filename']
            image = tf.image.decode_png(features['image/encoded'])
            height = tf.cast(features['image/height'], tf.int32)
            width = tf.cast(features['image/width'], tf.int32)
            image = tf.reshape(image,(height, width, 3))
            image_fine = tf.image.convert_image_dtype(image, tf.float32)
            image_fine = tf.image.resize_images(image_fine, IMG_SIZE)

            labels = tf.cast(features['image/object/class/label'].values, tf.int64)
            xmin = tf.cast(features['image/object/bbox/xmin'].values, tf.float32)
            xmax = tf.cast(features['image/object/bbox/xmax'].values, tf.float32)
            ymin = tf.cast(features['image/object/bbox/ymin'].values, tf.float32)
            ymax = tf.cast(features['image/object/bbox/ymax'].values, tf.float32)
            bboxes = tf.stack([ymin, xmin, ymax, xmax], axis=-1)

            # Image preprocessing
            image, labels, bboxes = augmentor.augment(image, labels, bboxes, IMG_SIZE[0], is_training=training)

            """
            The GT's needs to be adjusted to the feature map grids.
            Returns list with all values, workaround for the dataset.map fn
            """

            """anchors is a list, with tuples for each feature_map, (y,x,w,h)"""
            anchors = model.anchorboxes()

            """encode_bboxes return the burnt in new ground truths:
            gbox_labels, gbox_localizations, gbox_scores"""
            gbox_labels, gbox_localizations, gbox_scores = model.encode_bboxes(labels, bboxes, anchors)

            """return a list containing the image and the different gts"""
            if FLAGS.eval:
                gts = [image] + gbox_labels + gbox_scores + gbox_localizations + [image_fine] + [filename] + [height] + [width]
            else:
                gts = [image] + gbox_labels + gbox_scores + gbox_localizations + [image_fine]
            return gts

        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.map(_parse_fn)

        if shuffle:
            dataset = dataset.shuffle(buffer_size=FLAGS.mb_size*64)

        dataset = dataset.repeat(nepochs)

        dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))

        iterator = dataset.make_initializable_iterator()

        return iterator
    return _input_fn

def load_img(i):
    import os
    import imageio
    path = './eval_imgs/'
    imgs = sorted(os.listdir(path))
    img = imageio.imread(os.path.join(path,imgs[i]))
    return img, img.shape[:-1]


def main(_):

    model_fn = lambda: DSSD()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config).as_default():
        if FLAGS.eval:
            import dssd_eval
            input_fn = tfrecord_input_fn(test_record, shuffle=False, training=False, batch_size=FLAGS.mb_size)
            dssd_eval.eval(input_fn, model_fn)
            return

        if FLAGS.training:
            input_fn = tfrecord_input_fn(train_record, shuffle=True, training=True)
            eval_input_fn = tfrecord_input_fn(test_record, batch_size=FLAGS.mb_size)
            dssd_train.train([input_fn, eval_input_fn], model_fn)

        else:
            get_infos = dssd_inference.eval(0, model_fn)
            import imageio
            import matplotlib.pyplot as plt
            vid = imageio.get_reader('output.mp4', 'ffmpeg')

            for num, image in enumerate(vid.iter_data()):
                if num % 2 == 0:
                    infos = get_infos(image, np.shape(image))
                    print([info['class'] for info in infos])
            # for i in range(1):
            #     img, img_shape = load_img(i)
            #     print(np.float32(img_shape))
            #     infos = get_infos(img, img_shape)
            #     print([info['class'] for info in infos])


tf.app.run()


