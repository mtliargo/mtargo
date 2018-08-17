import math
import os.path
import time
import numpy as np
import tensorflow as tf
from deeplab import common
from deeplab import model
from deeplab.datasets import segmentation_dataset
from deeplab.utils import input_generator
from deeplab.utils import save_annotation

slim = tf.contrib.slim

flags = tf.app.flags

FLAGS = flags.FLAGS

flags.DEFINE_string('master', '', 'BNS name of the tensorflow server')

# Settings for log directories.

flags.DEFINE_string('vis_logdir', None, 'Where to write the event logs.')

flags.DEFINE_string('checkpoint_dir', None, 'Directory of model checkpoints.')

# Settings for visualizing the model.

flags.DEFINE_integer('vis_batch_size', 1,
                     'The number of images in each batch during evaluation.')

flags.DEFINE_multi_integer('vis_crop_size', [513, 513],
                           'Crop size [height, width] for visualization.')

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

flags.DEFINE_string('vis_split', 'val',
                    'Which split of the dataset used for visualizing results')

flags.DEFINE_string('dataset_dir', None, 'Where the dataset reside.')

flags.DEFINE_enum('colormap_type', 'pascal', ['pascal', 'cityscapes'],
                  'Visualization colormap type.')

flags.DEFINE_boolean('also_save_raw_predictions', False,
                     'Also save raw predictions.')

flags.DEFINE_integer('max_number_of_iterations', 0,
                     'Maximum number of visualization iterations. Will loop '
                     'indefinitely upon nonpositive values.')


# The format to save prediction
_PREDICTION_FORMAT = '%08d'

# To evaluate Cityscapes results on the evaluation server, the labels used
# during training should be mapped to the labels for evaluation.
_CITYSCAPES_TRAIN_ID_TO_EVAL_ID = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22,
                                   23, 24, 25, 26, 27, 28, 31, 32, 33]


def _convert_train_id_to_eval_id(prediction, train_id_to_eval_id):
  """Converts the predicted label for evaluation.

  There are cases where the training labels are not equal to the evaluation
  labels. This function is used to perform the conversion so that we could
  evaluate the results on the evaluation server.

  Args:
    prediction: Semantic segmentation prediction.
    train_id_to_eval_id: A list mapping from train id to evaluation id.

  Returns:
    Semantic segmentation prediction whose labels have been changed.
  """
  converted_prediction = prediction.copy()
  for train_id, eval_id in enumerate(train_id_to_eval_id):
    converted_prediction[prediction == train_id] = eval_id

  return converted_prediction


def _process_batch(sess, filename_list, original_images, semantic_predictions, image_names,
                   image_heights, image_widths, image_id_offset, save_dir):
  (original_images,
   semantic_predictions,
   image_names,
   image_heights,
   image_widths,
   ) = sess.run([original_images, semantic_predictions, image_names, image_heights, image_widths])

  num_image = semantic_predictions.shape[0]
  for i in range(num_image):
    name = image_names[i].decode('utf-8')
    idx = filename_list.index(name)
    image_height = np.squeeze(image_heights[i])
    image_width = np.squeeze(image_widths[i])
    semantic_prediction = np.squeeze(semantic_predictions[i])
    crop_semantic_prediction = semantic_prediction[:image_height, :image_width]

    save_annotation.save_annotation_indexed(
        crop_semantic_prediction, save_dir,
        _PREDICTION_FORMAT % (idx + 1), colormap_type=FLAGS.colormap_type)

    
def main(unused_argv):
  tf.logging.set_verbosity(tf.logging.INFO)
  # Get dataset-dependent information.
  dataset = segmentation_dataset.get_dataset(
      FLAGS.dataset, FLAGS.vis_split, dataset_dir=FLAGS.dataset_dir)

  filename_list = open(os.path.join(FLAGS.dataset_dir, '../ReOrg/filename-val.txt'), encoding='utf-8').readlines()
  filename_list = [x.strip() for x in filename_list]

  # Prepare for visualization.
  tf.gfile.MakeDirs(FLAGS.vis_logdir)
  save_dir = os.path.join(FLAGS.vis_logdir, FLAGS.vis_split)
  tf.gfile.MakeDirs(save_dir)

  tf.logging.info('Visualizing on %s set', FLAGS.vis_split)

  g = tf.Graph()
  with g.as_default():
    samples = input_generator.get(dataset,
                                  FLAGS.vis_crop_size,
                                  FLAGS.vis_batch_size,
                                  min_resize_value=FLAGS.min_resize_value,
                                  max_resize_value=FLAGS.max_resize_value,
                                  resize_factor=FLAGS.resize_factor,
                                  dataset_split=FLAGS.vis_split,
                                  is_training=False,
                                  model_variant=FLAGS.model_variant)
                                  
    model_options = common.ModelOptions(
        outputs_to_num_classes={common.OUTPUT_TYPE: dataset.num_classes},
        crop_size=FLAGS.vis_crop_size,
        atrous_rates=FLAGS.atrous_rates,
        output_stride=FLAGS.output_stride)

    if tuple(FLAGS.eval_scales) == (1.0,):
      tf.logging.info('Performing single-scale test.')
      predictions = model.predict_labels(
          samples[common.IMAGE],
          model_options=model_options,
          image_pyramid=FLAGS.image_pyramid)
    else:
      tf.logging.info('Performing multi-scale test.')
      predictions = model.predict_labels_multi_scale(
          samples[common.IMAGE],
          model_options=model_options,
          eval_scales=FLAGS.eval_scales,
          add_flipped_images=FLAGS.add_flipped_images)
    predictions = predictions[common.OUTPUT_TYPE]

    # Set ignore_label regions to label 0, because metrics.mean_iou requires
    # range of labels = [0, dataset.num_classes). Note the ignore_label regions
    # are not evaluated since the corresponding regions contain weights = 0.

    if FLAGS.min_resize_value and FLAGS.max_resize_value:
      # Only support batch_size = 1, since we assume the dimensions of original
      # image after tf.squeeze is [height, width, 3].
      assert FLAGS.vis_batch_size == 1

      # Reverse the resizing and padding operations performed in preprocessing.
      # First, we slice the valid regions (i.e., remove padded region) and then
      # we reisze the predictions back.
      original_image = tf.squeeze(samples[common.ORIGINAL_IMAGE])
      original_image_shape = tf.shape(original_image)
      predictions = tf.slice(
          predictions,
          [0, 0, 0],
          [1, original_image_shape[0], original_image_shape[1]])
      resized_shape = tf.to_int32([tf.squeeze(samples[common.HEIGHT]),
                                   tf.squeeze(samples[common.WIDTH])])
      predictions = tf.squeeze(
          tf.image.resize_images(tf.expand_dims(predictions, 3),
                                 resized_shape,
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
                                 align_corners=True), 3)

    tf.train.get_or_create_global_step()
    saver = tf.train.Saver(slim.get_variables_to_restore())
    sv = tf.train.Supervisor(graph=g,
                             logdir=FLAGS.vis_logdir,
                             init_op=tf.global_variables_initializer(),
                             summary_op=None,
                             summary_writer=None,
                             global_step=None,
                             saver=saver)
    num_batches = int(math.ceil(
        dataset.num_samples / float(FLAGS.vis_batch_size)))

    last_checkpoint = FLAGS.checkpoint_dir

    start = time.time()
    tf.logging.info(
        'Starting visualization at ' + time.strftime('%Y-%m-%d-%H:%M:%S',
                                                    time.gmtime()))
    tf.logging.info('Visualizing with model %s', last_checkpoint)

    with sv.managed_session(FLAGS.master,
                            start_standard_services=False) as sess:
        sv.start_queue_runners(sess)
        sv.saver.restore(sess, last_checkpoint)

        image_id_offset = 1
        for batch in range(num_batches):
            tf.logging.info('Visualizing batch %d / %d', batch + 1, num_batches)
            _process_batch(sess=sess,
                            filename_list=filename_list,
                            original_images=samples[common.ORIGINAL_IMAGE],
                            semantic_predictions=predictions,
                            image_names=samples[common.IMAGE_NAME],
                            image_heights=samples[common.HEIGHT],
                            image_widths=samples[common.WIDTH],
                            image_id_offset=image_id_offset,
                            save_dir=save_dir)
            image_id_offset += FLAGS.vis_batch_size

        tf.logging.info(
            'Finished prediction at ' + time.strftime('%Y-%m-%d-%H:%M:%S',
                                                        time.gmtime()))


if __name__ == '__main__':
  flags.mark_flag_as_required('checkpoint_dir')
  flags.mark_flag_as_required('vis_logdir')
  flags.mark_flag_as_required('dataset_dir')
  tf.app.run()
