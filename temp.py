import tensorflow as tf
from tensorflow.python.platform import build_info as tf_build_info
a = tf_build_info
print(tf.config.list_physical_devices())
print("Num GPUs Available: ", len(tf.config.list_physical_devices()))
print(tf_build_info.cudnn_version_number)
#print("cudnn_version",tf_build_info.build_info['cudnn_version'])

print('')

print(tf.__version__)
