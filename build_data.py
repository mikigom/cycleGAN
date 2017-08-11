import tensorflow as tf
import random
import os
import StringIO
#from PIL import Image
import cv2


try:
  from os import scandir
except ImportError:
  # Python 2 polyfill module
  from scandir import scandir
    

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('X_input_dir', 'data/trainA',
                       'X input directory, default: data/trainA')
tf.flags.DEFINE_string('Y_input_dir', 'data/trainB',
                       'Y input directory, default: data/trainB')
tf.flags.DEFINE_string('X_output_file', 'data/trainA.tfrecords',
                       'X output tfrecords file, default: data/trainA.tfrecords')
tf.flags.DEFINE_string('Y_output_file', 'data/trainB.tfrecords',
                       'Y output tfrecords file, default: data/trainB.tfrecords')

def data_reader(input_dir, shuffle=True):
  """Read images from input_dir then shuffle them
  Args:
    input_dir: string, path of input dir, e.g., /path/to/dir
  Returns:
    file_paths: list of strings
  """
  file_paths = []

  for img_file in scandir(input_dir):
    if img_file.name.endswith('.jpeg') and img_file.is_file():
      file_paths.append(img_file.path)

  if shuffle:
    # Shuffle the ordering of all image files in order to guarantee
    # random ordering of the images with respect to label in the
    # saved TFRecord files. Make the randomization repeatable.
    shuffled_index = list(range(len(file_paths)))
    random.seed(12345)
    random.shuffle(shuffled_index)

    file_paths = [file_paths[i] for i in shuffled_index]

  return file_paths


def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(image_buffer):
  """Build an Example proto for an example.
  Args:
    image_buffer: string, JPEG encoding of RGB image
  Returns:
    Example proto
  """
  example = tf.train.Example(features=tf.train.Features(feature={
      'image/encoded_image': _bytes_feature((image_buffer))
    }))
  return example

class ImageCoder(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Create a single Session to run all image coding calls.
    self._sess = tf.Session()

    # Initializes function that converts PNG to JPEG data.
    self._png_data = tf.placeholder(dtype=tf.string)
    image = tf.image.decode_png(self._png_data, channels=3)
    self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)

    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def decode_jpeg(self, image_data):
    image = self._sess.run(self._decode_jpeg,
                           feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image



def data_writer(input_dir, output_file):
  """Write data to tfrecords
  """
  file_paths = data_reader(input_dir)

  # create tfrecords dir if not exists
  output_dir = os.path.dirname(output_file)
  try:
    os.makedirs(output_dir)
  except os.error as e:
    pass

  images_num = len(file_paths)
  coder = ImageCoder()

  # dump to tfrecords file
  writer = tf.python_io.TFRecordWriter(output_file)

  for i in range(len(file_paths)):
    file_path = file_paths[i]

    with tf.gfile.FastGFile(file_path, 'rb') as f:
      image_data = f.read()

    image_data = coder.decode_jpeg(image_data)
    image_data = cv2.copyMakeBorder(image_data, 88, 88, 48, 48, cv2.BORDER_REFLECT)

    cv2.imwrite('data/temp.jpg', cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR))

    #padding.save('data/temp.jpg')

    with tf.gfile.FastGFile('data/temp.jpg', 'rb') as g:
      image_buff = g.read()

    example = _convert_to_example(image_buff)
    writer.write(example.SerializeToString())

    if i % 500 == 0:
      print("Processed {}/{}.".format(i, images_num))
  print("Done.")
  writer.close()

def main(unused_argv):
  print("Convert X data to tfrecords...")
  data_writer(FLAGS.X_input_dir, FLAGS.X_output_file)
  print("Convert Y data to tfrecords...")
  data_writer(FLAGS.Y_input_dir, FLAGS.Y_output_file)

if __name__ == '__main__':
  tf.app.run()
