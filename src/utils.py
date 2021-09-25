import tensorflow as tf

def right_handed_conv(inp, filt):
    """
    Implements a convolution where the anchor is on the right of the filter.
    Values outside the bounds of the input tensors are assumed to be zero.

    Inputs:
    - inp [batch, time (inp)]
    - filt [batch, filter_len]
    Output: [batch, time (inp)]
    """

    with tf.name_scope("right_handed_conv"):
        def apply_conv(x):
            inp, filt = x
            filt = tf.concat([tf.zeros((filt.shape[0] + 1,)), filt], axis=0)
            filt = tf.reverse(filt, [0])

            inp_reshaped = tf.reshape(inp, (1, inp.shape[0], 1))
            filt_reshaped = tf.reshape(filt, (filt.shape[0], 1, 1))

            conv = tf.nn.conv1d(inp_reshaped, filt_reshaped, [1], "SAME")

            return tf.reshape(conv, inp.shape)

        output_spec = tf.TensorSpec((inp.shape[1],), dtype=inp.dtype)
        return tf.map_fn(apply_conv, (inp, filt), fn_output_signature=output_spec)
