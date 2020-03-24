import cv2
from glob import glob
import tensorflow as tf
import numpy as np
import alexnet
from IPython.display import clear_output
clear_output()

dataset_ = "/content/drive/My Drive/ResearchWork/dataset/"
X = tf.placeholder(tf.float32, shape=(None, 224, 224, 3), name="input")

def inference():
    ref_dict = {}
    for img_file in glob(dataset_+"test/*.*"):
        img_name = img_file.split('/')[-1]
        img = cv2.imread(img_file)
        img = cv2.resize(img, (224, 224))
        ref_dict[img_name] = img
    images = np.asarray(list(ref_dict.values()))
    output = alexnet.alexnet_v2(X)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state("model")
        saver.restore(sess, "model/model-epoch40.ckpt")
        result = sess.run(output, feed_dict={X:images})
        result = tf.nn.softmax(result)
        inf_result = tf.argmax(result, 1)
        class_result = sess.run(result)
        out = sess.run(inf_result)
        for res in range(len(out)):
            if (out[res] == 0):
                print('It is Orca!', list(ref_dict.keys())[res])
                print(class_result[res])
            else:
                print('It is Non-Orca!', list(ref_dict.keys())[res])
                print(class_result[res])


if __name__ == "__main__":
    inference()
