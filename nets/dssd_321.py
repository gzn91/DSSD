import tensorflow as tf
import tensorflow.contrib as tc
import tensorflow.contrib.slim as slim
from nets.layers import conv2d, conv2d_transpose, l2_normalization
from tensorflow.contrib.slim.nets import resnet_v2
from collections import namedtuple
import numpy as np
from datasets.cls_dict_gen import num_cls

FLAGS = tf.app.flags.FLAGS
SSDParams = namedtuple('SSDParameters', ['img_shape',
                                         'num_classes',
                                         'feat_layers',
                                         'feat_shapes',
                                         'anchor_scale_bounds',
                                         'anchor_scales',
                                         'anchor_aspectratios',
                                         'anchor_steps',
                                         'anchor_offset',
                                         'multibox_l2'
                                         ])
default_params = SSDParams(
    img_shape=(321, 321),
    num_classes=num_cls(),
    feat_layers=['block14', 'block13', 'block12', 'block11', 'block10', 'block9'],
    feat_shapes=[(41, 41), (21, 21), (11, 11), (6, 6), (3, 3), (1, 1)],
    anchor_scale_bounds=[0.1, 0.90],
    anchor_scales=[(0.1, 0.1414), (0.2, 0.2739), (0.375, 0.4541), (0.55, 0.6315), (0.725, 0.8078), (0.9, 0.9836)],
    anchor_aspectratios=[[2, .5],
                         [2, .5, 1.6, 3, 1. / 3],
                         [2, .5, 1.6, 3, 1. / 3],
                         [2, .5, 1.6, 3, 1. / 3],
                         [2, .5, 1.6, 3, 1. / 3],
                         [2, .5]],
    anchor_steps=[8, 16, 32, 64, 110, 321],
    anchor_offset=0.5,
    multibox_l2=[True, False, False, False, False, False]
)

# Compute the scales used for the anchors, different layers will have different scale,
# the scale is computed within the size bounds and will be the lower bound for the lowest layer
# and the largest bound for the highest layer. The layers in between will have a scale in the range
# (lower, higher bound) increasing linearly with each layer.
# def scale(size_bounds, nfeature_maps):
#     def _scale(k):
#         sk = size_bounds[0] + (size_bounds[1] - size_bounds[0]) * (k - 1) / (nfeature_maps - 1)
#         nxt_sk = size_bounds[0] + (size_bounds[1] - size_bounds[0]) * (k) / (nfeature_maps - 1)
#         return (sk, nxt_sk)
#
#     return _scale


class DSSDNet(object):

    def __init__(self, params=default_params):
        self.params = params
        self.filters = 64
        self.kernel_size = 3
        self.linear = lambda x: x

    def multibox_layer(self, inputs, ncls, scales, aratios=[1], multibox_l2norm=False, is_training=False):

        x = conv2d(inputs, filters=256, kernel_size=1, padding='SAME', name='multibox_conv1', training=is_training, use_bn=True)
        x = conv2d(x, filters=256, kernel_size=1, padding='SAME', name='multibox_conv2', training=is_training, use_bn=True)
        x = conv2d(x, filters=1024, kernel_size=1, padding='SAME', name='multibox_conv3', activation_fn=self.linear)
        x += conv2d(inputs, filters=1024, kernel_size=1, padding='SAME', name='multibox_conv_res', activation_fn=self.linear)

        x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.nn.leaky_relu(x)

        # Number of anchors
        num_anchors = len(scales) + len(aratios)

        # Location
        num_loc_pred = num_anchors * 4  # 4 offsets, k = num_anchors, i.e. 4k
        loc_pred = conv2d(x, filters=num_loc_pred, kernel_size=3, name='conv_loc',
                          padding='SAME', activation_fn=self.linear)
        loc_pred = tf.reshape(loc_pred, loc_pred.get_shape().as_list()[:-1] + [num_anchors, 4])

        # Class prediction
        num_cls_pred = num_anchors * ncls  # c = num_classes, i.e. ck
        cls_pred = conv2d(x, filters=num_cls_pred, kernel_size=3, name='conv_cls',
                          padding='SAME', activation_fn=self.linear)
        cls_pred = tf.reshape(cls_pred, cls_pred.get_shape().as_list()[:-1] + [num_anchors, ncls])
        return cls_pred, loc_pred

    def deconv_module(self, x, y, is_training, kernel_size=2, scope=''):
        self.filters //= 2

        with tf.variable_scope(scope):
            # From upsample
            x = conv2d_transpose(x, filters=512, strides=2, padding='VALID',
                                 kernel_size=kernel_size, name='deconv1_x_' + scope, activation_fn=self.linear)

            pad = 'VALID' if x.get_shape().as_list()[1] != y.get_shape().as_list()[1] else 'SAME'
            ks = 2 if pad == 'VALID' else 3
            x = conv2d(x, filters=512, kernel_size=ks, padding=pad, name='conv1_x_' + scope,
                       training=is_training, use_bn=True, activation_fn=self.linear)

            # From downsample
            y = conv2d(y, filters=512, kernel_size=1, padding='SAME', name='conv1_y_' + scope,
                       training=is_training, use_bn=True)
            y = conv2d(y, filters=512, kernel_size=3, padding='SAME', name='conv2_y_' + scope,
                       training=is_training, use_bn=True, activation_fn=self.linear)

            #  eltw prod
            x *= y
            x = tf.nn.leaky_relu(x)

        return x

    """FPN conv nets for feature maps"""

    def ssd_network(self, image, *args, **kwargs):
        is_training, reuse, scope, use_bn = args
        ncls, fmap_layer, anchor_scales, anchor_ratios, multibox_l2 = kwargs.values()
        x = image

        # Down
        print(x.get_shape())
        """THIS MODEL NEEDS TO MODIFY RESNET FROM 2,2,2,1 TO 2,1,2,1"""
        with tf.variable_scope('', reuse=reuse):
            with slim.arg_scope(resnet_v2.resnet_arg_scope()):
                # resnet strides changed, default strides are 2,2,2,1
                net, end_points = resnet_v2.resnet_v2_101(image,
                                                          num_classes=None,
                                                          is_training=False,
                                                          global_pool=False,
                                                          output_stride=None,
                                                          reuse=reuse)
            x_33 = end_points['resnet_v2_101/block2']
            x = end_points['resnet_v2_101/block4']

            print(x.get_shape())
            print(x_33.get_shape())

            # SSD Blocks
            # 17 x 17
            # Dilation block
            # Batch norm and leaky relu
            with tf.variable_scope('block5'):
                x = conv2d(x, 1024, 3, dilations=6, name='dil1', padding='SAME', use_bn=True,
                              training=is_training)
                x_17 = conv2d(x, 1024, 1, dilations=1, name='conv1', padding='SAME', use_bn=True,
                              training=is_training)
                print('shape after block5', x_17.get_shape().as_list())

            # SSD Blocks
            # 9 x 9
            with tf.variable_scope('block6'):
                x = conv2d(x_17, 256, 1, name='conv1', padding='SAME', use_bn=True, training=is_training)
                x_9 = conv2d(x, 512, 3, strides=2, name='conv2', padding='SAME', use_bn=True, training=is_training)
                print('shape after block6', x_9.get_shape().as_list())

            # 5 x 5
            with tf.variable_scope('block7'):
                x = conv2d(x_9, 128, 1, name='conv1', padding='SAME', use_bn=True, training=is_training)
                x_5 = conv2d(x, 256, 3, strides=2, name='conv2', padding='SAME', use_bn=True, training=is_training)
                print('shape after block7', x_5.get_shape().as_list())

            # 3 x 3
            with tf.variable_scope('block8'):
                x = conv2d(x_5, 128, 1, name='conv1', padding='SAME', use_bn=True, training=is_training)
                x_3 = conv2d(x, 256, 3, strides=2, name='conv2', padding='SAME', use_bn=True, training=is_training)
                print('shape after block8', x_3.get_shape().as_list())

            # 1 x 1
            with tf.variable_scope('block9'):
                x = conv2d(x_3, 128, 1, name='conv1', padding='SAME', use_bn=True, training=is_training)
                x_1 = conv2d(x, 256, 3, strides=1, name='conv2', padding='VALID', use_bn=True, training=is_training)
                end_points['block9'] = x_1
                print('shape after block9', x_1.get_shape().as_list())

            # 3 x 3
            with tf.variable_scope('block10'):
                y_3 = self.deconv_module(x_1, x_3, is_training, kernel_size=3, scope='up1')
                print('shape after block10', y_3.get_shape().as_list())
                end_points['block10'] = y_3

            with tf.variable_scope('block11'):
                y_4 = self.deconv_module(y_3, x_5, is_training, scope='up2')
                print('shape after block11', y_4.get_shape().as_list())
                end_points['block11'] = y_4

            with tf.variable_scope('block12'):
                y_8 = self.deconv_module(y_4, x_9, is_training, scope='up3')
                print('shape after block12', y_8.get_shape().as_list())
                end_points['block12'] = y_8

            with tf.variable_scope('block13'):
                y_17 = self.deconv_module(y_8, x_17, is_training, scope='up4')
                print('shape after block13', y_17.get_shape().as_list())
                end_points['block13'] = y_17

            with tf.variable_scope('block14'):
                y_33 = self.deconv_module(y_17, x_33, is_training, scope='up5')
                print('shape after block14', y_33.get_shape().as_list())
                end_points['block14'] = y_33

            self.filters = 64

            predictions = []
            logits = []
            localizations = []
            """From each feature map in VGG send it to ssd multibox layer to predict class and location of objects"""
            for i, layer in enumerate(fmap_layer):
                with tf.variable_scope(layer + '_box', reuse=reuse):
                    prediction, localization = self.multibox_layer(end_points[layer],
                                                                   ncls,
                                                                   anchor_scales[i],
                                                                   anchor_ratios[i],
                                                                   multibox_l2[i],
                                                                   is_training)
                predictions.append(tf.nn.softmax(prediction))
                logits.append(prediction)
                localizations.append(localization)

            return predictions, localizations, logits, end_points


class DSSD(DSSDNet):

    def __init__(self):
        super().__init__()
        params = self.params
        # self.scale = scale(params.anchor_scale_bounds, len(params.feat_layers))
        # SSDParams.anchor_scales = list(map(self.scale, list(range(1, len(params.feat_layers) + 1))))
        net_kwargs = {'num_classes': params.num_classes, 'feat_layers': params.feat_layers,
                      'anchor_scales': self.params.anchor_scales, 'anchor_aspectratios': params.anchor_aspectratios,
                      'multibox_l2': params.multibox_l2}
        print(self.params.anchor_scales)

        # Arguments: is_training, reuse, scope, use batchnorm
        if FLAGS.training:
            eval_args = (False, True, 'ssd_fpn_512', FLAGS.use_bn)
            train_args = (True, False, 'ssd_fpn_512', FLAGS.use_bn)
        else:
            eval_args = (False, False, 'ssd_fpn_512', FLAGS.use_bn)

        def model(*args):
            def _model(img):
                return super(DSSD, self).ssd_network(img, *args, **net_kwargs)

            return _model

        def set_ground_thruth_anchors(labels, bboxes, anchors):
            from utils import set_gbbox
            gtlabels, gtbboxes, gtscores = [], [], []
            for idx, anchor in enumerate(anchors):
                # print('Layers %d' % idx)
                _label, _bboxes, _scores = set_gbbox(labels, bboxes, anchor)
                gtlabels.append(_label)
                gtbboxes.append(_bboxes)
                gtscores.append(_scores)
            return gtlabels, gtbboxes, gtscores

        def loss_fn(*args):
            total_loss = self.loss(*args)
            return total_loss

        self.encode_bboxes = set_ground_thruth_anchors
        self.LR = tf.placeholder(tf.float32)

        if FLAGS.training:
            self.train_model = model(*train_args)
        self.eval_model = model(*eval_args)

        self.losses = loss_fn
        self.global_step = tf.train.get_or_create_global_step()
        self.optimize = tf.train.MomentumOptimizer(learning_rate=self.LR, momentum=0.9, use_nesterov=True)
        self.get_params = lambda: tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES)
        self.model_params = None
        self.sess = tf.get_default_session()

    """
    Compute losses, args should be in order: 
    logits, localizations, glabels, glocalizations, gscores
    """

    def loss(self, *args, thresh=0.5, negratio=3., alpha=1., scope=''):
        from utils import flat_featuremaps, flatarrs_in_list
        with tf.name_scope(scope, 'losses'):
            # Flatten
            nbatch, ncls = args[0][0].get_shape().as_list()[0], args[0][0].get_shape().as_list()[-1]
            sizes = {'ncls': (ncls,), 'nloc': (4,), 'gcls': (), 'gloc': (4,), 'gsc': ()}
            logits, localizations, glabels, glocalizations, gscores = flat_featuremaps(*args, **sizes)

            # Compute predictions
            predictions = tf.nn.softmax(logits, axis=-1)

            # Compute positive and negative loss:
            smooth_l1 = lambda x: tf.where(tf.greater_equal(x, 1), tf.abs(x) - .5, .5 * tf.square(x))
            loc_loss = smooth_l1(localizations - glocalizations)
            conf_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=tf.one_hot(glabels, ncls))
            with tf.name_scope('positive'):
                # Positive confidences if score > thresh
                pmask = gscores > thresh
                fpmask = tf.cast(pmask, tf.float32)
                # Calc num positive boxes
                npositives = tf.reduce_sum(fpmask)
                N = tf.maximum(1., npositives)
                # boolean mask returns only the values where the mask is true
                ploss = tf.reduce_sum(tf.boolean_mask(conf_loss, pmask, name='positive_conf_loss')) / N
                plocloss = tf.reduce_sum(tf.boolean_mask(alpha * loc_loss, pmask, name='positive_loc_loss')) / N
            with tf.name_scope('negative'):
                # Calc num of negative boxes
                nmask = tf.logical_not(pmask)
                fnmask = tf.cast(nmask, tf.float32)
                nnegatives = tf.minimum(negratio * npositives, tf.reduce_sum(fnmask))
                # Check is there is any negative boxes
                use_nnegatives = nnegatives > 0
                use_nnegatives_float = tf.cast(use_nnegatives, tf.float32)
                # Use 8 neg boxes if there is no positive boxes
                nnegatives = nnegatives * use_nnegatives_float + (1 - use_nnegatives_float) * 8
                nnegatives = tf.cast(nnegatives, tf.int32)

                # Largest confidence loss
                max_hard_pred = tf.reduce_max(predictions[:, 1:], axis=-1)
                # Returns indices top nnegatives confidence losses
                vals, idxes = tf.nn.top_k(max_hard_pred * fnmask, k=nnegatives)

                # Gather the losses
                nloss = tf.gather(conf_loss, idxes)
                # Compute sum of neglosses
                nloss = tf.reduce_sum(nloss, name='negative_conf_loss') / tf.cast(nnegatives, tf.float32)

                total_loss = ploss + nloss
                total_loss += (alpha * plocloss)

            return total_loss

    def anchorboxes(self):
        from utils import generate_anchors
        layer_anchors = []
        for feat_shape, step, scale, ratio in zip(self.params.feat_shapes,
                                                  self.params.anchor_steps,
                                                  self.params.anchor_scales,
                                                  self.params.anchor_aspectratios):

            layer_anchors.append(generate_anchors(feat_shape, step, scale, ratio))
        return layer_anchors

    """locs has shape [(1, 37, 37, 4, 4), (1, 17, 17, 6, 4), (1, 9, 9, 6, 4), (1, 5, 5, 6, 4), (1, 3, 3, 4, 4), (1, 1, 1, 4, 4)],
    needs to be fatten in order to plot bboxes e.g. [mb, -1, 4, 4]"""

    def get_bboxes(self, predictions_layer, locs_layer, anchors_layer, prior_scaling=[0.1, 0.1, 0.2, 0.2], thresh=.1):

        def _get_bbox(preds, locs, anchors):
            y, x, h, w = anchors

            scores = tf.reshape(preds, [-1, *preds.get_shape().as_list()[-2:]])
            locs = tf.reshape(locs, [-1, *locs.get_shape().as_list()[-2:]])

            x = tf.reshape(x, [1, -1, 1])
            y = tf.reshape(y, [1, -1, 1])

            # prior variance on localizations
            cx = locs[..., 0] * w * prior_scaling[0] + x
            cy = locs[..., 1] * h * prior_scaling[1] + y
            w = w * tf.exp(locs[..., 2] * prior_scaling[2])
            h = h * tf.exp(locs[..., 3] * prior_scaling[3])
            # bboxes: ymin, xmin, xmax, ymax.
            ymin = tf.maximum(cy - h / 2., 0.)
            xmin = tf.maximum(cx - w / 2., 0.)
            ymax = tf.minimum(cy + h / 2., 1.)
            xmax = tf.minimum(cx + w / 2., 1.)

            bboxes = tf.squeeze(tf.stack([ymin, xmin, ymax, xmax], axis=-1), axis=0)

            # Back to original shape.
            scores = tf.reshape(scores, [-1, 21])
            # classes = tf.reshape(classes, [-1])
            bboxes = tf.reshape(bboxes, [-1, 4])

            return scores, bboxes

        # for each layers feature map add class and location of found objects
        nbatch = predictions_layer[0].get_shape().as_list()[0]
        batch_bboxes, batch_scores, batch_classes = [[] for i in range(nbatch)], [[] for i in range(nbatch)], [[] for i
                                                                                                               in range(
                nbatch)]

        for layer_id, (preds, locs, anchors) in enumerate(zip(predictions_layer, locs_layer, anchors_layer)):
            # preds = tf.Print(preds, [tf.shape(preds)])
            batch_preds, batch_locs = map(lambda x: tf.unstack(x, num=nbatch), (preds, locs))

            for batch_id in range(nbatch):
                p = batch_preds[batch_id]
                score, bbox = _get_bbox(p, batch_locs[batch_id], anchors)

                batch_bboxes[batch_id].append(bbox)
                batch_scores[batch_id].append(score)

        # # this for loop performes NMS on each batch and class separately, in order to do this we use a boolean mask
        #   so only elements for the current batch sample is taken into consideration.
        # #
        selected_scores = []
        selected_bboxes = []
        for i, (bb, sc) in enumerate(zip(batch_bboxes, batch_scores)):
            sc = tf.concat(sc, 0)
            bb = tf.concat(bb, 0)
            """NEW"""
            sel_sc = {}
            sel_bb = {}
            for c in range(1, FLAGS.ncls):
                class_scores = sc[:, c]
                select_mask = class_scores > thresh

                _bb = tf.boolean_mask(bb, select_mask)
                _sc = tf.boolean_mask(class_scores, select_mask)
                idx = tf.image.non_max_suppression(_bb, _sc, iou_threshold=.45, max_output_size=200)
                sel_sc[c] = tf.gather(_sc, idx)
                sel_bb[c] = tf.gather(_bb, idx)
            selected_scores.append(sel_sc)
            selected_bboxes.append(sel_bb)
            """END"""
        # Due to different images may have a different amount of object bboxes and scores
        #  can not be stacked and is returned as a list
        return selected_scores, selected_bboxes


class DSSDSaver(object):
    def __init__(self):
        self.saver = tf.train.Saver()
        self.sess = tf.get_default_session()

    def restore_model(self):
        print('{}'.format(tf.train.latest_checkpoint('./saved_model/')))
        self.saver.restore(self.sess, '{}'.format(tf.train.latest_checkpoint('./saved_model/')))

    def save_model(self, step):
        self.saver.save(self.sess, './saved_model/model', global_step=step)
