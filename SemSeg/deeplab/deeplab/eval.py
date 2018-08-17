# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Evaluation script for the DeepLab model.

See model.py for more details and usage.
"""

from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import confusion_matrix
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import sets
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import weights_broadcast_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.tf_export import tf_export

import math
import six
import tensorflow as tf
from deeplab import common
from deeplab import model
from deeplab.datasets import segmentation_dataset
from deeplab.utils import input_generator
import numpy as np

slim = tf.contrib.slim

flags = tf.app.flags

FLAGS = flags.FLAGS

flags.DEFINE_string('master', '', 'BNS name of the tensorflow server')

# Settings for log directories.

flags.DEFINE_string('eval_logdir', None, 'Where to write the event logs.')

flags.DEFINE_string('checkpoint_dir', None, 'Directory of model checkpoints.')

# Settings for evaluating the model.

flags.DEFINE_integer('eval_batch_size', 1,
                                         'The number of images in each batch during evaluation.')

flags.DEFINE_multi_integer('eval_crop_size', [513, 513],
                                                     'Image crop size [height, width] for evaluation.')

flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                                         'How often (in seconds) to run evaluation.')

# For `xception_65`, use atrous_rates = [12, 24, 36] if output_stride = 8, or
# rates = [6, 12, 18] if output_stride = 16. For `mobilenet_v2`, use None. Note
# one could use different atrous_rates/output_stride during training/evaluation.
flags.DEFINE_multi_integer('atrous_rates', None,
                                                     'Atrous rates for atrous spatial pyramid pooling.')

flags.DEFINE_integer('output_stride', 16,
                                         'The ratio of input to output spatial resolution.')

# Change to [0.5, 0.75, 1.0, 1.25, 1.5, 1.75] for multi-scale test.
flags.DEFINE_multi_float('eval_scales', [1.0],
                                                 'The scales to resize images for evaluation.')

# Change to True for adding flipped images during test.
flags.DEFINE_bool('add_flipped_images', False,
                                    'Add flipped images for evaluation or not.')

# Dataset settings.

flags.DEFINE_string('dataset', 'pascal_voc_seg',
                                        'Name of the segmentation dataset.')

flags.DEFINE_string('eval_split', 'val',
                                        'Which split of the dataset used for evaluation')

flags.DEFINE_string('dataset_dir', None, 'Where the dataset reside.')

flags.DEFINE_integer('max_number_of_evaluations', 0,
                                         'Maximum number of eval iterations. Will loop '
                                         'indefinitely upon nonpositive values.')

def metric_variable(shape, dtype, validate_shape=True, name=None):
  """Create variable in `GraphKeys.(LOCAL|METRIC_VARIABLES`) collections."""

  return variable_scope.variable(
      lambda: array_ops.zeros(shape, dtype),
      trainable=False,
      collections=[
          ops.GraphKeys.LOCAL_VARIABLES, ops.GraphKeys.METRIC_VARIABLES
      ],
      validate_shape=validate_shape,
      name=name)

def _streaming_confusion_matrix(labels, predictions, num_classes, weights=None):
    """Calculate a streaming confusion matrix.

    Calculates a confusion matrix. For estimation over a stream of data,
    the function creates an  `update_op` operation.

    Args:
        labels: A `Tensor` of ground truth labels with shape [batch size] and of
        type `int32` or `int64`. The tensor will be flattened if its rank > 1.
        predictions: A `Tensor` of prediction results for semantic labels, whose
        shape is [batch size] and type `int32` or `int64`. The tensor will be
        flattened if its rank > 1.
        num_classes: The possible number of labels the prediction task can
        have. This value must be provided, since a confusion matrix of
        dimension = [num_classes, num_classes] will be allocated.
        weights: Optional `Tensor` whose rank is either 0, or the same rank as
        `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
        be either `1`, or the same as the corresponding `labels` dimension).

    Returns:
        total_cm: A `Tensor` representing the confusion matrix.
        update_op: An operation that increments the confusion matrix.
    """
    # Local variable to accumulate the predictions in the confusion matrix.
    total_cm = metric_variable(
        [num_classes, num_classes], dtypes.float64, name='total_confusion_matrix')

    # Cast the type to int64 required by confusion_matrix_ops.
    predictions = math_ops.to_int64(predictions)
    labels = math_ops.to_int64(labels)
    num_classes = math_ops.to_int64(num_classes)

    # Flatten the input if its rank > 1.
    if predictions.get_shape().ndims > 1:
        predictions = array_ops.reshape(predictions, [-1])

    if labels.get_shape().ndims > 1:
        labels = array_ops.reshape(labels, [-1])

    if (weights is not None) and (weights.get_shape().ndims > 1):
        weights = array_ops.reshape(weights, [-1])

    # Accumulate the prediction to current confusion matrix.
    current_cm = confusion_matrix.confusion_matrix(
        labels, predictions, num_classes, weights=weights, dtype=dtypes.float64)
    update_op = state_ops.assign_add(total_cm, current_cm)
    return total_cm, update_op

def my_mean_iou(labels,
                         predictions,
                         num_classes,
                         weights=None,
                         metrics_collections=None,
                         updates_collections=None,
                         name=None):
    """Calculate per-step mean Intersection-Over-Union (mIOU).
    """

    if context.executing_eagerly():
        raise RuntimeError('tf.metrics.mean_iou is not supported when eager execution is enabled.')

    with variable_scope.variable_scope(name, 'mean_iou', (predictions, labels, weights)):
        # Check if shape is compatible.
        predictions.get_shape().assert_is_compatible_with(labels.get_shape())

        total_cm, update_op = _streaming_confusion_matrix(labels, predictions, num_classes, weights)

        def compute_mean_iou(name):
            """Compute the mean intersection-over-union via the confusion matrix."""
            sum_over_row = math_ops.to_float(math_ops.reduce_sum(total_cm, 0))
            sum_over_col = math_ops.to_float(math_ops.reduce_sum(total_cm, 1))
            cm_diag = math_ops.to_float(array_ops.diag_part(total_cm))
            denominator = sum_over_row + sum_over_col - cm_diag

            # The mean is only computed over classes that appear in the
            # label or prediction tensor. If the denominator is 0, we need to
            # ignore the class.
            num_valid_entries = math_ops.reduce_sum(
                    math_ops.cast(
                            math_ops.not_equal(denominator, 0), dtype=dtypes.float32))

            # If the value of the denominator is 0, set it to 1 to avoid
            # zero division.
            denominator = array_ops.where(
                    math_ops.greater(denominator, 0), denominator,
                    array_ops.ones_like(denominator))
            iou = math_ops.div(cm_diag, denominator)

            # If the number of valid entries is 0 (no classes) we return 0.
            result = array_ops.where(
                    math_ops.greater(num_valid_entries, 0),
                    math_ops.reduce_sum(iou, name=name) / num_valid_entries, 0)
            return result

        mean_iou_v = compute_mean_iou('mean_iou')

        if metrics_collections:
            ops.add_to_collections(metrics_collections, mean_iou_v)

        if updates_collections:
            ops.add_to_collections(updates_collections, update_op)

        return mean_iou_v, update_op, total_cm

def main(unused_argv):
    tf.logging.set_verbosity(tf.logging.INFO)
    # Get dataset-dependent information.
    dataset = segmentation_dataset.get_dataset(
            FLAGS.dataset, FLAGS.eval_split, dataset_dir=FLAGS.dataset_dir)

    tf.gfile.MakeDirs(FLAGS.eval_logdir)
    tf.logging.info('Evaluating on %s set', FLAGS.eval_split)

    with tf.Graph().as_default():
        samples = input_generator.get(
                dataset,
                FLAGS.eval_crop_size,
                FLAGS.eval_batch_size,
                min_resize_value=FLAGS.min_resize_value,
                max_resize_value=FLAGS.max_resize_value,
                resize_factor=FLAGS.resize_factor,
                dataset_split=FLAGS.eval_split,
                is_training=False,
                model_variant=FLAGS.model_variant)

        model_options = common.ModelOptions(
                outputs_to_num_classes={common.OUTPUT_TYPE: dataset.num_classes},
                crop_size=FLAGS.eval_crop_size,
                atrous_rates=FLAGS.atrous_rates,
                output_stride=FLAGS.output_stride)

        if tuple(FLAGS.eval_scales) == (1.0,):
            tf.logging.info('Performing single-scale test.')
            predictions = model.predict_labels(samples[common.IMAGE], model_options,
                                                                                 image_pyramid=FLAGS.image_pyramid)
        else:
            tf.logging.info('Performing multi-scale test.')
            predictions = model.predict_labels_multi_scale(
                    samples[common.IMAGE],
                    model_options=model_options,
                    eval_scales=FLAGS.eval_scales,
                    add_flipped_images=FLAGS.add_flipped_images)
        predictions = predictions[common.OUTPUT_TYPE]
        predictions = tf.reshape(predictions, shape=[-1])
        labels = tf.reshape(samples[common.LABEL], shape=[-1])
        weights = tf.to_float(tf.not_equal(labels, dataset.ignore_label))

        # Set ignore_label regions to label 0, because metrics.mean_iou requires
        # range of labels = [0, dataset.num_classes). Note the ignore_label regions
        # are not evaluated since the corresponding regions contain weights = 0.
        labels = tf.where(
                tf.equal(labels, dataset.ignore_label), tf.zeros_like(labels), labels)

        predictions_tag = 'miou'
        for eval_scale in FLAGS.eval_scales:
            predictions_tag += '_' + str(eval_scale)
        if FLAGS.add_flipped_images:
            predictions_tag += '_flipped'

        # Define the evaluation metric.
        metric_map = {}
        # metric_map[predictions_tag] = my_mean_iou(
        #        predictions, labels, dataset.num_classes, weights=weights)

        a, b, total_cm = my_mean_iou(
                predictions, labels, dataset.num_classes, weights=weights)
        metric_map[predictions_tag] = (a, b)


        metrics_to_values, metrics_to_updates = (
                tf.contrib.metrics.aggregate_metric_map(metric_map))

        for metric_name, metric_value in six.iteritems(metrics_to_values):
            slim.summaries.add_scalar_summary(
                    metric_value, metric_name, print_summary=True)

        num_batches = int(
                math.ceil(dataset.num_samples / float(FLAGS.eval_batch_size)))

        tf.logging.info('Eval num images %d', dataset.num_samples)
        tf.logging.info('Eval batch size %d and num batch %d',
                                        FLAGS.eval_batch_size, num_batches)

        num_eval_iters = None
        if FLAGS.max_number_of_evaluations > 0:
            num_eval_iters = FLAGS.max_number_of_evaluations
        # slim.evaluation.evaluation_loop(
        #         master=FLAGS.master,
        #         checkpoint_dir=FLAGS.checkpoint_dir,
        #         logdir=FLAGS.eval_logdir,
        #         num_evals=num_batches,
        #         eval_op=list(metrics_to_updates.values()),
        #         max_number_of_evaluations=num_eval_iters,
        #         eval_interval_secs=FLAGS.eval_interval_secs)

        total_cm_val = slim.evaluation.evaluate_once(
                master=FLAGS.master,
                checkpoint_path=FLAGS.checkpoint_dir,
                logdir=FLAGS.eval_logdir,
                num_evals=num_batches,
                eval_op=list(metrics_to_updates.values()),
                final_op=total_cm)
        # np.savetxt('theirs.txt', total_cm_val, '%.18g')



if __name__ == '__main__':
    flags.mark_flag_as_required('checkpoint_dir')
    flags.mark_flag_as_required('eval_logdir')
    flags.mark_flag_as_required('dataset_dir')
    tf.app.run()
