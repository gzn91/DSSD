from nets.dssd_321 import DSSDSaver
import tensorflow as tf
import numpy as np
import os
from preprocessing.augmentor import preprocess_for_export

FLAGS = tf.app.flags.FLAGS

def load_img(path='./data/test/imgs/'):
    import os
    import cv2
    import imageio as io
    mb_size = FLAGS.mb_size
    img_list = sorted(os.listdir(path))
    N = len(img_list)
    for img_id in range(0, N, mb_size):
        imgs = [cv2.cvtColor(cv2.imread(os.path.join(path, img_list[i])), cv2.COLOR_BGR2RGB)
                for i in np.arange(img_id, img_id + mb_size) % N]
        shapes = [img.shape[:-1] for img in imgs]
        imgs = [cv2.resize(img, (FLAGS.img_size, FLAGS.img_size), interpolation=cv2.INTER_AREA) for img in imgs]
        names = [img_list[i] for i in np.arange(img_id, img_id + mb_size) % N]
        print(names)
        yield (np.array(imgs), np.array(shapes), names)


# main training loop
def eval(input_fn, model_fn):
    sess = tf.get_default_session()

    # Create model
    ssd = model_fn()

    img_ph = tf.placeholder(tf.uint8, shape=(FLAGS.mb_size, None, None, 3))
    # shape = tf.cast(.2*shape_ph, tf.int32)

    imgs = preprocess_for_export(img_ph, FLAGS.img_size, FLAGS.mb_size)

    # Eval model
    model = ssd.eval_model

    # Graph outputs
    predictions, localizations, logits, end_points = model(imgs)

    # Get anchor refs
    anchors = ssd.anchorboxes()

    # Decode bboxes, returns lists containing the score and bboxes for each sample in batch
    scores, bboxes = get_bboxes(predictions, localizations, anchors)

    # Initialize graph
    sess.run(tf.global_variables_initializer())
    saver = DSSDSaver()

    # raise KeyboardInterrupt
    saver.restore_model()
    detections = []
    path = './data/test2/imgs/'
    num_imgs = len(os.listdir(path))
    num_iters = num_imgs//FLAGS.mb_size

    iterator = load_img(path)

    for i, (img_mb, shapes, filenames) in enumerate(iterator):
        fd_map = {img_ph: img_mb}
        score, bbox = sess.run([scores, bboxes], feed_dict=fd_map)
        print(f'at iteration {i} of {num_iters}')

        for j in range(len(filenames)):
            # with open('./data/imgs.txt', 'a') as f:
            #     f.write(filenames[i][:-4]+'\n')
            pred = {}
            pred['filename'] = filenames[j]
            pred['shape'] = shapes[j]
            for c in range(1, FLAGS.ncls):
                pred[f'scores_{c}'] = score[j][c]
                pred[f'bboxes_{c}'] = bbox[j][c]
            detections.append(pred)

    from datasets.cls_dict_gen import labels_to_name
    names = labels_to_name()
    for class_ind in range(1, FLAGS.ncls):
        # with open(os.path.join('./data/results/VOC2010/Main/', 'comp3_det_test_{}.txt'.format(names[class_ind])), 'wt') as f:
        with open(os.path.join('./data/predict/', 'results_{}.txt'.format(class_ind)), 'wt') as f:
            for image_ind, pred in enumerate(detections):
                filename = pred['filename']
                shape = pred['shape']
                scores = pred['scores_{}'.format(class_ind)]
                bboxes = pred['bboxes_{}'.format(class_ind)]
                bboxes[:, 0] = (bboxes[:, 0] * shape[0]).astype(np.int64, copy=False) + 1
                bboxes[:, 1] = (bboxes[:, 1] * shape[1]).astype(np.int64, copy=False) + 1
                bboxes[:, 2] = (bboxes[:, 2] * shape[0]).astype(np.int64, copy=False) + 1
                bboxes[:, 3] = (bboxes[:, 3] * shape[1]).astype(np.int64, copy=False) + 1

                valid_mask = np.logical_and((bboxes[:, 2] - bboxes[:, 0] > 0), (bboxes[:, 3] - bboxes[:, 1] > 0))

                for det_ind in range(valid_mask.shape[0]):
                    if not valid_mask[det_ind]:
                        continue
                    f.write('{:s} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n'.
                            format(filename[:-4], scores[det_ind],
                                   bboxes[det_ind, 1], bboxes[det_ind, 0],
                                   bboxes[det_ind, 3], bboxes[det_ind, 2]))


def get_bboxes(predictions_layer, locs_layer, anchors_layer, prior_scaling=[0.1, 0.1, 0.2, 0.2], thresh=.1):

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
        scores = tf.reshape(scores, [-1, FLAGS.ncls])
        # classes = tf.reshape(classes, [-1])
        bboxes = tf.reshape(bboxes, [-1, 4])

        return scores, bboxes

    # for each layers feature map add class and location of found objects
    nbatch = predictions_layer[0].get_shape().as_list()[0]
    batch_bboxes, batch_scores, batch_classes = [[] for i in range(nbatch)], [[] for i in range(nbatch)], [[] for i
                                                                                                           in range(
            nbatch)]

    for layer_id, (preds, locs, anchors) in enumerate(zip(predictions_layer, locs_layer, anchors_layer)):

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
    selected_classes = []
    selected_bboxes = []
    for i, (bb, sc) in enumerate(zip(batch_bboxes, batch_scores)):
        sc = tf.concat(sc, 0)
        bb = tf.concat(bb, 0)

        sel_sc = {}
        sel_bb = {}
        for c in range(1, FLAGS.ncls):
            class_scores = sc[:, c]
            select_mask = class_scores > thresh

            _bb = tf.boolean_mask(bb, select_mask)
            _sc = tf.boolean_mask(class_scores, select_mask)

            idx = tf.image.non_max_suppression(_bb, _sc, iou_threshold=.45, max_output_size=20)

            sel_sc[c] = tf.gather(_sc, idx)
            sel_bb[c] = tf.gather(_bb, idx)
        selected_scores.append(sel_sc)
        selected_bboxes.append(sel_bb)

    # Due to different images may have a different amount of object bboxes and scores
    #  can not be stacked and is returned as a list
    return selected_scores, selected_bboxes
