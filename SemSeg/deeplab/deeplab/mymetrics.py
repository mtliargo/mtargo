import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import confusion_matrix
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope

def _remove_squeezable_dimensions(predictions, labels, weights):
  """Squeeze or expand last dim if needed.

  Squeezes last dim of `predictions` or `labels` if their rank differs by 1
  (using confusion_matrix.remove_squeezable_dimensions).
  Squeezes or expands last dim of `weights` if its rank differs by 1 from the
  new rank of `predictions`.

  If `weights` is scalar, it is kept scalar.

  This will use static shape if available. Otherwise, it will add graph
  operations, which could result in a performance hit.

  Args:
    predictions: Predicted values, a `Tensor` of arbitrary dimensions.
    labels: Optional label `Tensor` whose dimensions match `predictions`.
    weights: Optional weight scalar or `Tensor` whose dimensions match
      `predictions`.

  Returns:
    Tuple of `predictions`, `labels` and `weights`. Each of them possibly has
    the last dimension squeezed, `weights` could be extended by one dimension.
  """
  predictions = ops.convert_to_tensor(predictions)
  if labels is not None:
    labels, predictions = confusion_matrix.remove_squeezable_dimensions(
        labels, predictions)
    predictions.get_shape().assert_is_compatible_with(labels.get_shape())

  if weights is None:
    return predictions, labels, None

  weights = ops.convert_to_tensor(weights)
  weights_shape = weights.get_shape()
  weights_rank = weights_shape.ndims
  if weights_rank == 0:
    return predictions, labels, weights

  predictions_shape = predictions.get_shape()
  predictions_rank = predictions_shape.ndims
  if (predictions_rank is not None) and (weights_rank is not None):
    # Use static rank.
    if weights_rank - predictions_rank == 1:
      weights = array_ops.squeeze(weights, [-1])
    elif predictions_rank - weights_rank == 1:
      weights = array_ops.expand_dims(weights, [-1])
  else:
    # Use dynamic rank.
    weights_rank_tensor = array_ops.rank(weights)
    rank_diff = weights_rank_tensor - array_ops.rank(predictions)

    def _maybe_expand_weights():
      return control_flow_ops.cond(
          math_ops.equal(rank_diff, -1),
          lambda: array_ops.expand_dims(weights, [-1]), lambda: weights)

    # Don't attempt squeeze if it will fail based on static check.
    if ((weights_rank is not None) and
        (not weights_shape.dims[-1].is_compatible_with(1))):
      maybe_squeeze_weights = lambda: weights
    else:
      maybe_squeeze_weights = lambda: array_ops.squeeze(weights, [-1])

    def _maybe_adjust_weights():
      return control_flow_ops.cond(
          math_ops.equal(rank_diff, 1), maybe_squeeze_weights,
          _maybe_expand_weights)

    # If weights are scalar, do nothing. Otherwise, try to add or remove a
    # dimension to match predictions.
    weights = control_flow_ops.cond(
        math_ops.equal(weights_rank_tensor, 0), lambda: weights,
        _maybe_adjust_weights)
  return predictions, labels, weights

def _safe_div(numerator, denominator, name):
  """Divides two tensors element-wise, returning 0 if the denominator is <= 0.

  Args:
    numerator: A real `Tensor`.
    denominator: A real `Tensor`, with dtype matching `numerator`.
    name: Name for the returned op.

  Returns:
    0 if `denominator` <= 0, else `numerator` / `denominator`
  """
  t = math_ops.truediv(numerator, denominator)
  zero = array_ops.zeros_like(t, dtype=denominator.dtype)
  condition = math_ops.greater(denominator, zero)
  zero = math_ops.cast(zero, t.dtype)
  return array_ops.where(condition, t, zero, name=name)


# def scatter_add_tensor(ref, indices, updates, name=None):
#     """
#     Adds sparse updates to a variable reference.

#     This operation outputs ref after the update is done. This makes it easier to chain operations that need to use the
#     reset value.

#     Duplicate indices are handled correctly: if multiple indices reference the same location, their contributions add.

#     Requires updates.shape = indices.shape + ref.shape[1:].
#     :param ref: A Tensor. Must be one of the following types: float32, float64, int64, int32, uint8, uint16,
#         int16, int8, complex64, complex128, qint8, quint8, qint32, half.
#     :param indices: A Tensor. Must be one of the following types: int32, int64. A tensor of indices into the first
#         dimension of ref.
#     :param updates: A Tensor. Must have the same dtype as ref. A tensor of updated values to add to ref
#     :param name: A name for the operation (optional).
#     :return: Same as ref. Returned as a convenience for operations that want to use the updated values after the update
#         is done.
#     """
#     with tf.name_scope(name, 'scatter_add_tensor', [ref, indices, updates]) as scope:
#         ref = tf.convert_to_tensor(ref, name='ref')
#         indices = tf.convert_to_tensor(indices, name='indices')
#         updates = tf.convert_to_tensor(updates, name='updates')
#         ref_shape = tf.shape(ref, out_type=indices.dtype, name='ref_shape')
#         scattered_updates = tf.scatter_nd(indices, updates, ref_shape, name='scattered_updates')
#         with tf.control_dependencies([tf.assert_equal(ref_shape, tf.shape(scattered_updates, out_type=indices.dtype))]):
#             output = tf.add(ref, scattered_updates, name=scope)
#         return output

def compute_m_iou_accu(labels,
                         predictions,
                         num_classes,
                         weights=None,
                         name=None):
    """Calculate per-step mean Intersection-Over-Union (mIOU).
    """
    # Check if shape is compatible.
    predictions.get_shape().assert_is_compatible_with(labels.get_shape())

    predictions = math_ops.to_int64(predictions)
    labels = math_ops.to_int64(labels)
    num_classes = math_ops.to_int64(num_classes)
    current_cm = confusion_matrix.confusion_matrix(
        labels, predictions, num_classes, weights=weights, dtype=dtypes.float64)

    sum_over_row = math_ops.to_float(math_ops.reduce_sum(current_cm, 0))
    sum_over_col = math_ops.to_float(math_ops.reduce_sum(current_cm, 1))
    cm_diag = math_ops.to_float(array_ops.diag_part(current_cm))
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
    m_iou = array_ops.where(
            math_ops.greater(num_valid_entries, 0),
            math_ops.reduce_sum(iou) / num_valid_entries, 0)

    class_count = array_ops.where(
            math_ops.greater(sum_over_col, 0), sum_over_col,
            array_ops.ones_like(sum_over_col))

    accu_per_class = math_ops.div(cm_diag, class_count)
    m_accu = array_ops.where(
            math_ops.greater(num_valid_entries, 0),
            math_ops.reduce_sum(accu_per_class) / num_valid_entries, 0)

    return m_iou, m_accu

def compute_accu(labels,
                predictions,
                weights=None,
                name=None):
    """Calculates how often `predictions` matches `labels`.
    """

    predictions, labels, weights = _remove_squeezable_dimensions(
            predictions=predictions, labels=labels, weights=weights)
    predictions.get_shape().assert_is_compatible_with(labels.get_shape())
    if labels.dtype != predictions.dtype:
        predictions = math_ops.cast(predictions, labels.dtype)
    is_correct = math_ops.to_float(math_ops.equal(predictions, labels))
    if weights != None:
        is_correct = math_ops.multiply(is_correct, weights)
    accu = math_ops.reduce_mean(is_correct)
    return accu

# def compute_maccu(labels,
#                                                         predictions,
#                                                         num_classes,
#                                                         weights=None,
#                                                         name=None):
#     """Calculates the mean of the per-class accuracies.
#     """
#     with variable_scope.variable_scope(name, 'mean_accuracy',
#             (predictions, labels, weights)):
#         labels = math_ops.to_int64(labels)

#         # Flatten the input if its rank > 1.
#         if labels.get_shape().ndims > 1:
#             labels = array_ops.reshape(labels, [-1])

#         if predictions.get_shape().ndims > 1:
#             predictions = array_ops.reshape(predictions, [-1])

#         # Check if shape is compatible.
#         predictions.get_shape().assert_is_compatible_with(labels.get_shape())

#         total = array_ops.zeros([num_classes], dtypes.float32)
#         count = array_ops.zeros([num_classes], dtypes.float32)

#         ones = array_ops.ones([array_ops.size(labels)], dtypes.float32)

#         if labels.dtype != predictions.dtype:
#             predictions = math_ops.cast(predictions, labels.dtype)
#         is_correct = math_ops.to_float(math_ops.equal(predictions, labels))

#         if weights is not None:
#             if weights.get_shape().ndims > 1:
#                 weights = array_ops.reshape(weights, [-1])
#             weights = math_ops.to_float(weights)

#             is_correct *= weights
#             ones *= weights


#         total = tf.scatter_nd_add(tf.Variable(total), labels, ones)
#         count = tf.scatter_nd_add(tf.Variable(count), labels, is_correct)

#         # total = scatter_add_tensor(total, labels, ones)
#         # count = scatter_add_tensor(count, labels, is_correct)

#         per_class_accuracy = _safe_div(count, total, None)

#         mean_accuracy_v = math_ops.reduce_mean(
#                 per_class_accuracy, name='mean_accuracy')
        
#         return mean_accuracy_v