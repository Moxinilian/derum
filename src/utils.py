import tensorflow as tf

def right_handed_conv(inp, filt):
    """
    Implements a convolution where the anchor is on the right of the filter.
    Values outside the bounds of the input tensors are assumed to be zero.

    Do not run without XLA if you want acceptable speeds.

    Inputs:
    - inp [batch, time (inp)]
    - filt [batch, filter_len]
    Output: [batch, time (inp)]
    """

    with tf.name_scope("right_handed_conv"):
        output = tf.zeros(inp.shape)
        to_concat = []
        for t in range(inp.shape[1]):
            to_sum = []
            for k in range(filt.shape[1]):
                if t-k >= 0:
                    to_sum.append(filt[:,k] * inp[:,t-k])
            to_concat.append(tf.reshape(tf.math.accumulate_n(to_sum), (inp.shape[0], 1)))

        return tf.concat(to_concat, axis=1)
