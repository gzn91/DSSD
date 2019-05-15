from nets.dssd_321 import DSSDSaver
import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np

FLAGS = tf.app.flags.FLAGS


# simple formatting for mb generation
def mb_generator(iterator, nlayers):
    def _mb_gen():
        gt = iterator.get_next()
        return gt[0], gt[1:1+nlayers], gt[1+nlayers:1+2*nlayers], gt[1+2*nlayers:1+3*nlayers], gt[-1]
    return _mb_gen


# main training loop
def train(input_fn, model_fn):
    sess = tf.get_default_session()

    # Create model
    ssd = model_fn()

    # Initialize iterator and create mb generator
    iterator = input_fn[0](ssd)
    iterator_eval = input_fn[1](ssd)
    # iterator = input_fn(ssd)
    iterator.initializer.run()
    iterator_eval.initializer.run()
    nlayers = len(ssd.params.feat_layers)
    nxt_mb = mb_generator(iterator, nlayers)
    nxt_mb_eval = mb_generator(iterator_eval, nlayers)

    # Defs of optimizer (Adam) and loss fn
    optimizer = ssd.optimize
    loss = ssd.losses

    # Traning model
    model = ssd.train_model
    # Eval model
    eval_model = ssd.eval_model

    # mb generation
    imgs, gbox_labels, gbox_scores, gbox_localizations, crop_image = nxt_mb()
    # tf.summary.image('training_input', imgs, max_outputs=6)
    imgs_eval, gbox_labels_eval, gbox_scores_eval, gbox_localizations_eval, crop_image_eval = nxt_mb_eval()
    # tf.summary.image('test_input', imgs_eval, max_outputs=6)

    # Graph outputs
    predictions, localizations, logits, end_points = model(imgs)
    predictions_eval, localizations_eval, logits_eval, _ = eval_model(imgs_eval)

    # Get trainable variables, (neural network weights)
    train_params = ssd.model_params = ssd.get_params()

    # Update batchnorm
    batch_norm_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    batch_norm_op = [param for param in batch_norm_op if 'resnet' not in param.name]

    # Get graph losses
    total_loss = loss(logits, localizations, gbox_labels, gbox_localizations, gbox_scores)
    total_loss_eval = loss(logits_eval, localizations_eval, gbox_labels_eval, gbox_localizations_eval, gbox_scores_eval)

    # Compute gradients
    train_params = [param for param in train_params if 'resnet' not in param.name]
    grads = tf.gradients(total_loss, train_params)
    # grads = tf.clip_by_value(grads, 1e-8, 5.0)
    grads_and_vars = list(zip(grads, train_params))

    # Get anchor refs
    anchors = ssd.anchorboxes()

    # Update neural network weights
    update_op = [optimizer.apply_gradients(grads_and_vars, global_step=ssd.global_step), batch_norm_op]
    # update_op = optimizer.minimize(total_loss)

    with tf.device('/cpu:0'):
        # Decode bboxes, returns lists containing the score and bboxes for each sample in batch, crops is cropping coords
        # and batch_aid is a helper vector to relate each crop with the correct batch.
        # _ = eval, __ = gt
        scores, bboxes = ssd.get_bboxes(predictions_eval,localizations_eval, anchors)
        _scores, _bboxes = ssd.get_bboxes(predictions, localizations, anchors)
        __scores, __bboxes = ssd.get_bboxes([tf.one_hot(_, depth=FLAGS.ncls) for _ in gbox_labels],
                                                              gbox_localizations, anchors)

        # Draw bboxes on image
        # bboxes is a list since it's not possible to stack tensors with different amount of num bboxes
        draw_bboxes = []
        _draw_bboxes = []
        __draw_bboxes = []

        for i in range(1, FLAGS.mb_size):
            bb = tf.concat([bboxes[i][k] for k in scores[i].keys()], axis=0)
            _bb = tf.concat([_bboxes[i][k] for k in _scores[i].keys()], axis=0)
            __bb = tf.concat([__bboxes[i][k] for k in __scores[i].keys()], axis=0)
            draw_bboxes.append(tf.image.draw_bounding_boxes(tf.expand_dims(tf.image.resize_images(imgs_eval[i], (300, 300)), axis=0),
                                                            tf.expand_dims(bb, axis=0)))
            _draw_bboxes.append(tf.image.draw_bounding_boxes(tf.expand_dims(tf.image.resize_images(imgs[i], (300, 300)), axis=0),
                                                            tf.expand_dims(_bb, axis=0)))
            __draw_bboxes.append(tf.image.draw_bounding_boxes(tf.expand_dims(tf.image.resize_images(imgs[i], (300, 300)), axis=0),
                                                            tf.expand_dims(__bb, axis=0)))

        draw_bboxes = tf.concat(draw_bboxes, 0)
        _draw_bboxes = tf.concat(_draw_bboxes, 0)
        __draw_bboxes = tf.concat(__draw_bboxes, 0)
        tf.summary.image('predictions', draw_bboxes, max_outputs=6)
        tf.summary.image('training_predictions', _draw_bboxes, max_outputs=6)
        tf.summary.image('training_data', __draw_bboxes, max_outputs=6)

    # Build summaries
    tf.summary.scalar('train_total_loss', total_loss)
    tf.summary.scalar('test_total_loss', total_loss_eval)
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

    # schedule
    lr_schedule = {50000: .1, 80000: .1, 100000: .1, 120000: .1}
    lrnow = FLAGS.learning_rate
    _lr = tf.Variable(FLAGS.learning_rate)
    _lrnow = _lr.assign(ssd.LR)

    # Initialize graph
    sess.run(tf.global_variables_initializer())

    variables2restore = slim.get_variables(scope='resnet_v2_101')

    init_fn = slim.assign_from_checkpoint_fn(
        './resnet/resnet_v2_101.ckpt',
        variables2restore,
        ignore_missing_vars=True)

    # Saver object to save and restore graph
    init_fn(sess)

    saver = DSSDSaver()
    i = 0

    if FLAGS.restore:
        saver.restore_model()
        i = sess.run(ssd.global_step)
        lrnow = sess.run(_lr)

    while i <= 150000:
        if i in lr_schedule:
            lrnow *= lr_schedule[i]
            sess.run(_lrnow, feed_dict={ssd.LR: lrnow})
            print('lr:', lrnow)
        if i % 200 == 0:
            sess.run(ssd.global_step.assign(i))
            summary, preds, glabels, ssd_loss = sess.run([merged, predictions_eval, gbox_labels_eval, total_loss_eval])
            print('EVALUATING AT STEP: ', i)
            print('val accuracy: {:.02f} % || ssd loss: {:.02f} ||'\
                  .format(accuracy_fn(preds, glabels) * 100, ssd_loss))
            writer.add_summary(summary, global_step=i)
            if i >= 5000 and i % 5000 == 0:
                print('lr:', lrnow)
                saver.save_model(i)
                print('SAVING MODEL')
        preds, glabels, ssd_loss, _, _, gscore = sess.run([predictions, gbox_labels, total_loss, update_op, batch_norm_op, gbox_scores], feed_dict={ssd.LR: lrnow})
        if i % 50 == 0:
            print('Max scores per layer:', [np.max(_) for _ in gscore])
            print('|| global step: {} || train accuracy: {:.02f} % || ssd loss: {:.02f} ||'\
                  .format(i, accuracy_fn(preds, glabels)*100, ssd_loss))
            print('')

        i += 1


# function for calculating the class accuracy for the current batch.
def accuracy_fn(preds, gts):
    corrs = 0
    count = 0
    for pred, gt in zip(preds, gts):
        pred = np.argmax(pred,-1)
        corrs += np.sum(pred[gt != 0] == gt[gt != 0])
        count += np.prod(gt[gt != 0].shape)
    return corrs/count
