import numpy as np, tensorflow as tf

FLAGS = tf.flags.FLAGS

# https://github.com/balancap/SSD-Tensorflow/blob/master/nets/ssd_vgg_300.py
# renamed some args
def generate_anchors(fmap_shape, step, scales, ratios, offset=.5):
    # Position grid, center of each anchor
    y, x = np.mgrid[0:fmap_shape[0], 0:fmap_shape[1]]
    y = (y.astype(np.float32) + offset) * step / FLAGS.img_size
    x = (x.astype(np.float32) + offset) * step / FLAGS.img_size

    # Expand dims to support easy broadcasting.
    y = np.expand_dims(y, axis=-1)
    x = np.expand_dims(x, axis=-1)

    # Compute relative height and width.
    # Tries to follow the original implementation of SSD for the order.
    num_anchors = len(scales) + len(ratios)
    h = np.zeros((num_anchors,), dtype=np.float32)
    w = np.zeros((num_anchors,), dtype=np.float32)
    # Add first anchor boxes with ratio=1.
    h[0] = scales[0]
    w[0] = scales[0]
    di = 1
    if len(scales) > 1:
        h[1] = np.sqrt(scales[0] * scales[1])
        w[1] = np.sqrt(scales[0] * scales[1])
        di += 1
    for i, r in enumerate(ratios):
        h[i + di] = scales[0] / np.sqrt(r)
        w[i + di] = scales[0] * np.sqrt(r)

    return y, x, h, w


def set_gbbox(labels, bboxes, anchors, prior_scaling=[0.1, 0.1, 0.2, 0.2]):

    # Anchors coordinates and volume.
    y, x, h, w = anchors
    ymin = y - h / 2.
    xmin = x - w / 2.
    ymax = y + h / 2.
    xmax = x + w / 2.

    vol_anchors = (xmax - xmin) * (ymax - ymin)

    """For each cell in grid and for each default anchor in that cell a possible ground truth label may be assigned"""
    shape = (y.shape[0], y.shape[1], h.size)

    fmap_scores = tf.zeros(shape, dtype=tf.float32, name='fmap_scores')
    fmap_labels = tf.zeros(shape, dtype=tf.int64, name='fmap_labels')
    fmap_ymin = tf.zeros(shape, dtype=tf.float32, name='fmap_ymin')
    fmap_xmin = tf.zeros(shape, dtype=tf.float32, name='fmap_xmin')
    fmap_ymax = tf.ones(shape, dtype=tf.float32, name='fmap_ymax')
    fmap_xmax = tf.ones(shape, dtype=tf.float32, name='fmap_xmax')

    idx = 0
    # Loop slice
    slices = (idx, fmap_labels, fmap_scores, fmap_ymin, fmap_xmin, fmap_ymax, fmap_xmax)

    def iou_with_anchors(bbox):
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

    def update_labels(*slices):
        """Update labels, and bboxes, if the IoU is greater than other anchors and is > .5"""
        i, fmap_labels, fmap_scores, \
        fmap_ymin, fmap_xmin, fmap_ymax, fmap_xmax = slices
        label = labels[i]
        bbox = bboxes[i]
        score = iou_with_anchors(bbox)
        # print(iou_score)
        """mask is a boolean mask with 1 assigned to the best anchor for that cell if any"""
        mask = tf.greater(score, fmap_scores)

        # Convert to float and int from bool for broadcasting
        imask = tf.cast(mask, tf.int64)
        fmask = tf.cast(mask, tf.float32)
        # Update labels and scores using mask.
        fmap_labels = tf.where(mask, imask * label, fmap_labels)
        fmap_scores = tf.where(mask, score, fmap_scores)
        fmap_ymin = tf.where(mask, fmask * bbox[0], fmap_ymin)
        fmap_xmin = tf.where(mask, fmask * bbox[1], fmap_xmin)
        fmap_ymax = tf.where(mask, fmask * bbox[2], fmap_ymax)
        fmap_xmax = tf.where(mask, fmask * bbox[3], fmap_xmax)

        return [i + 1, fmap_labels, fmap_scores,
                fmap_ymin, fmap_xmin, fmap_ymax, fmap_xmax]

    # N is num anchors, stops while loop when idx == num anchors
    N = tf.stack([tf.shape(labels)[0]])
    N = tf.reshape(N, ())

    def loop_cond(*slices):
        idx = slices[0]
        return tf.less_equal(idx, N - 1)

    # tf while loop, updates labels while loop_cond returns True
    slices = tf.while_loop(loop_cond, update_labels, slices)
    # split slices back to fmaps
    _, fmap_labels, fmap_scores, fmap_ymin, fmap_xmin, fmap_ymax, fmap_xmax = slices

    # Set BG class
    mask = tf.greater(fmap_scores, 0.5)
    fmap_labels, fmap_scores, fmap_ymin, fmap_xmin = \
        map(lambda x: tf.where(mask, x, tf.zeros_like(x)),
            (fmap_labels, fmap_scores, fmap_ymin, fmap_xmin))

    fmap_ymax, fmap_xmax = \
        map(lambda x: tf.where(mask, x, tf.ones_like(x)),
            (fmap_ymax, fmap_xmax))

    # Center coordinates
    fmap_cy = (fmap_ymax + fmap_ymin) / 2.
    fmap_cx = (fmap_xmax + fmap_xmin) / 2.
    fmap_h = fmap_ymax - fmap_ymin
    fmap_w = fmap_xmax - fmap_xmin

    # Set ground truth boxes x,y,h,w
    fmap_cy = (fmap_cy - y) / h / prior_scaling[0]
    fmap_cx = (fmap_cx - x) / w / prior_scaling[1]
    fmap_h = tf.log(fmap_h / h) / prior_scaling[2]
    fmap_w = tf.log(fmap_w / w) / prior_scaling[3]
    # stack on new axis e.g. 8x8xnum_anchorsx4
    fmap_localizations = tf.stack([fmap_cx, fmap_cy, fmap_w, fmap_h], axis=-1)

    return fmap_labels, fmap_localizations, fmap_scores


"""*fmaps is a list of lists of different feature maps where each list contains all the layers.
    returns a list with all the different feature maps flattened"""


def flat_featuremaps(*fmaps, **sizes):
    _flatarrs = [[] for _ in range(len(fmaps))]

    for idx, (_flatarr, size) in enumerate(zip(_flatarrs, sizes.values())):
        for arr in fmaps[idx]:
            _flatarr.append(flatarrs_in_list(arr, n=size))
        _flatarrs[idx] = tf.concat(_flatarr, axis=0)

    return _flatarrs


"""flattens arrays in args, to shape (-1, size) or (mb, -1, size), wher size is the unpacked size in sizes
corresponding to the args array. Returns list of flattened arrays in args"""


def flatarrs_in_list(*arrs, **sizes):
    _flatarr = []
    if 'use_mb' in sizes:
        nmbatch = arrs[0].get_shape().as_list()[0]

        for idx, (arr, size) in enumerate(zip(arrs, sizes.values())):
            _flatarr.append(tf.reshape(arr, [nmbatch, -1, *size]))
        _flatarr = tf.concat(_flatarr, axis=1)
        return _flatarr
    else:
        if not sizes:
            sizes = [_.get_shape().as_list()[-1] for _ in arrs]
        for idx, (arr, size) in enumerate(zip(arrs, sizes.values())):
            _flatarr.append(tf.reshape(arr, [-1, *size]))
        _flatarr = tf.concat(_flatarr, axis=0)
        return _flatarr


"""flatarrs flattens all arrays in *arrs to shape (mb,-1, size)"""


def flatarrs(*arrs, **sizes):
    _flatarr = []
    nmbatch = arrs[0].get_shape().as_list()[0]

    if not sizes:
        sizes = [_.get_shape().as_list()[-1] for _ in arrs]
    for idx, (arr, size) in enumerate(zip(arrs, sizes.values())):
        _flatarr.append(tf.reshape(arr, [nmbatch, -1, *size]))
        _flatarr[idx] = tf.concat(_flatarr[idx], axis=1)
    return _flatarr
