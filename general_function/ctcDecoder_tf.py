# -*- coding:utf-8 -*-
# author : zhangwei

import tensorflow as tf

class ctc_decoder():
    # 用tf定义一个专门ctc解码的图和会话，就不会一直增加节点了，速度快了很多
    def __init__(self, batch_size, timestep, nclass):
        self.batch_size = batch_size
        self.timestep = timestep
        self.nclass = nclass
        self.graph_ctc = tf.Graph()
        with self.graph_ctc.as_default():
            self.y_pred_tensor = tf.placeholder(shape=[self.batch_size, self.timestep, self.nclass], dtype=tf.float32, name="y_pred_tensor")
            self._y_pred_tensor = tf.transpose(self.y_pred_tensor, perm=[1, 0, 2])  #  要把timestep 放在第一维
            self.input_length_tensor = tf.placeholder(shape=[self.batch_size, 1], dtype=tf.int32, name="input_length_tensor")
            self._input_length_tensor = tf.squeeze(self.input_length_tensor, axis=1) #  传进来的是 [batch_size,1] 所以要去掉一维
            self.ctc_decode, _ = tf.nn.ctc_greedy_decoder(self._y_pred_tensor, self._input_length_tensor)
            self.decoded_sequences = tf.sparse_tensor_to_dense(self.ctc_decode[0])

            self.ctc_sess = tf.Session(graph=self.graph_ctc)

    def ctc_decode_tf(self, args):
        y_pred, input_length = args
        decoded_sequences = self.ctc_sess.run(self.decoded_sequences,feed_dict={self.y_pred_tensor: y_pred, self.input_length_tensor: input_length})
        return decoded_sequences