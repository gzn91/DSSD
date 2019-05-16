import base64
from nets.dssd_321 import DSSDSaver
from io import BytesIO
from PIL import Image
import os
import imageio
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from datasets.cls_dict_gen import labels_to_name
from preprocessing.augmentor import preprocess_for_export
from matplotlib.patches import Rectangle
from datetime import datetime

FLAGS = tf.flags.FLAGS


# main training loop
def eval(image, model_fn):
    sess = tf.get_default_session()

    # Create model
    ssd = model_fn()

    img_ph = tf.placeholder(tf.uint8, shape=(None, None, 3))

    img = preprocess_for_export(tf.expand_dims(img_ph, 0), FLAGS.img_size, 1)

    # Eval model
    eval_model = ssd.eval_model

    # Graph outputs
    predictions, localizations, logits, end_points = eval_model(img)
    # Get anchor refs
    anchors = ssd.anchorboxes()

    # Decode bboxes, returns lists containing the score and bboxes for each sample in batch
    scores, bboxes = ssd.get_bboxes(predictions,localizations, anchors)

    # Initialize graph
    sess.run(tf.global_variables_initializer())

    # Saver object to save and restore graph
    saver = DSSDSaver()

    saver.restore_model()
    global names
    names = labels_to_name()

    def get_infos(image, shape):
        score, bbox = sess.run([scores, bboxes], feed_dict={img_ph: image})

        for _score, _bbox in zip(score, bbox):
            draw_bbox(image, _score, _bbox)
        infos = []
        for k, v in score[0].items():
            for sc, bb in zip(score[0][k], bbox[0][k]):
                infos.append({'class': names[k], 'ymin': bb[0], 'xmin': bb[1], 'ymax': bb[2], 'xmax': bb[3]})

        return infos
    return get_infos


def iou_with_anchors(bboxes, labels):
    # Calculate the intersection x,y values
    iymin = tf.maximum(ymin, bbox[0])  # We here want the value furthest down
    ixmin = tf.maximum(xmin, bbox[1])  # We here want the value furthest to the right
    iymax = tf.minimum(ymax, bbox[2])  # We here want the value furthest up
    ixmax = tf.minimum(xmax, bbox[3])  # We here want the value furthest to the left
    ih = tf.maximum(iymax - iymin, 0)
    iw = tf.maximum(ixmax - ixmin, 0)
    # inter and union volumes
    ivol = ih * iw
    uvol = vol_anchors - ivol + (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    # (vol_anchors - ivol) is the volume that is outside of the bbox
    iou = ivol / uvol
    return iou


def draw_bbox(image, preds, bboxes):
    fig, ax = plt.subplots(1, figsize=(13, 7))
    scale_h, scale_w, _ = image.shape
    ax.imshow(image)

    for k, v in preds.items():
        for sc, bb in zip(preds[k], bboxes[k]):
            ymin, xmin, ymax, xmax = bb
            h = ymax-ymin
            w = xmax-xmin
            xmin *= scale_w
            xmax *= scale_w
            ymin *= scale_h
            ymax *= scale_h
            w *= scale_w
            h *= scale_h

            rec = Rectangle((xmin, ymin),width=w,height=h, fill=False, edgecolor='green', lw=3)
            ax.add_patch(rec)
            ax.annotate(f'{names[k]}: {(sc*100):.2f}%', (xmin, ymin), fontsize=15, color='white')
            # rec2 = Rectangle((xmin, ymin),width=100,height=100, fill=True, alpha=.5, facecolor='gray')
            # ax.add_patch(rec2)
            ax.axis('off')
    i = len(os.listdir('./eval_bboxed/'))
    plt.savefig(f'./eval_bboxed/out{i:03}.png')
    return

    pass




