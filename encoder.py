import tensorflow as tf
from tensorflow.python.ops import control_flow_ops, tensor_array_ops

def init_matrix(shape):
    return tf.random_normal(shape, stddev=0.1)

def init_vector(shape):
    return tf.zeros(shape)

class Encoder:
    def build():
        with tf.variable_scope('generator'):
            self.g_embeddings = tf.Variable(init_matrix([self.num_emb, self.input_dim]))
            self.g_params.append(self.g_embeddings)
            self.g_recurrent_unit = self.create_recurrent_unit(self.g_params)  # maps h_tm1 to h_t for generator
            self.g_output_unit = self.create_output_unit(self.g_params)  # maps h_t to o_t (output token logits)

        ta_emb_x = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=self.sequence_length)
        ta_emb_x = ta_emb_x.unstack(self.processed_x)

    def create_output_unit(self, params):
        self.Wo = tf.Variable(init_matrix([self.hidden_dim, self.num_emb]))
        self.bo = tf.Variable(init_matrix([self.num_emb]))
        params.extend([self.Wo, self.bo])

        def unit(hidden_memory_tuple):
            hidden_state, c_prev = tf.unstack(hidden_memory_tuple)
            # hidden_state : batch x hidden_dim
            logits = tf.matmul(hidden_state, self.Wo) + self.bo
            # output = tf.nn.softmax(logits)
            return logits

        return unit

class GRU(object):
    """
    Input: [seq_len * batch_size * input_dim]
    Output: [seq_len * batch_size * hidden_dim]
    """
    def __init__(self, input_dim, hidden_dim,
             sequence_length):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        self.recurrent_unit = self.create_recurrent_unit()

    def build(self, init_state, xs, context=None):

        # processed for batch
        # with tf.device("/cpu:0"):
            # self.processed_x = tf.transpose(tf.nn.embedding_lookup(self.g_embeddings, self.x), perm=[1, 0, 2])  # seq_length x batch_size x input_dim

        hiddens = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=self.sequence_length,
            dynamic_size=False, infer_shape=True)
        # hiddens = []

        def recurrence(i, h_tm1, hs):
            h_t = self.recurrent_unit(xs[i], h_tm1)  # hidden_memory_tuple
            # hiddens = hiddens.write(i, h_t)
            hs = hs.write(i, h_t)
            print ("HELLO", h_t)
            # hs.append(h_t)
            return i + 1, h_t, hs

        _, _, self.hiddens = control_flow_ops.while_loop(
            cond=lambda i, _1, _3: i < self.sequence_length,
            body=recurrence,
            loop_vars=(tf.constant(0, dtype=tf.int32),
                       # xs,
                       init_state,
                       hiddens))

        return self.hiddens.stack()
        # self.hiddens = tf.transpose(self.hiddens.stack(), perm=[1, 0, 2])  # batch_size x seq_length x vocab_size

    def create_recurrent_unit(self):
        with tf.name_scope("GRU"):
            self.W_rx = tf.Variable(init_matrix([self.hidden_dim, self.input_dim]), name="W_rx")
            self.W_zx = tf.Variable(init_matrix([self.hidden_dim, self.input_dim]), name="W_zx")
            self.W_hx = tf.Variable(init_matrix([self.hidden_dim, self.input_dim]), name="W_hx")
            self.U_rh = tf.Variable(init_matrix([self.hidden_dim, self.hidden_dim]), name="U_rh")
            self.U_zh = tf.Variable(init_matrix([self.hidden_dim, self.hidden_dim]), name="U_zh")
            self.U_hh = tf.Variable(init_matrix([self.hidden_dim, self.hidden_dim]), name="U_hh")
        # params.extend([
        #     self.W_rx, self.W_zx, self.W_hx,
        #     self.U_rh, self.U_zh, self.U_hh])

        def unit(x_t, h_tm1):
            # x_t = tf.reshape(x_t, [self.input_dim, 1])
            # h_tm1 = tf.reshape(h_tm1, [self.hidden_dim, 1])
            print x_t.get_shape().as_list()
            print h_tm1.get_shape().as_list()
            print self.W_rx.get_shape().as_list()
            print self.W_zx.get_shape().as_list()
            r = tf.sigmoid(tf.matmul(x_t, self.W_rx) + tf.matmul(h_tm1, self.U_rh))
            z = tf.sigmoid(tf.matmul(x_t, self.W_zx) + tf.matmul(h_tm1, self.U_zh))
            h_tilda = tf.tanh(tf.matmul(x_t, self.W_hx) + tf.matmul(r * h_tm1, self.U_hh))
            h_t = (1 - z) * h_tm1 + z * h_tilda
            return h_t
            # return tf.reshape(h_t, [self.hidden_dim])

        return unit
