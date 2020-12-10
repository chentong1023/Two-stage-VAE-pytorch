import torch

# Original deconv2d:
#def deconv2d(input_, output_shape, k_h, k_w, d_h, d_w, stddev=0.02, name="deconv2d"):
#    with tf.variable_scope(name):
#        w = tf.get_variable("w", [k_h, k_w, output_shape[-1], input_.get_shape()[-1]], initializer=tf.random_normal_initializer(stddev=stddev))
#        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape, strides=[1, d_h, d_w, 1])
#        biases = tf.get_variable("biases", [output_shape[-1]], initializer=tf.constant_initializer(0.0))
#    return tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
# 
def n_deconv2d(input__,o_shape, k_h, k_w, d_h, d_w, stddev=0.02, name="deconv2d"):
        output_shape=[o_shape[0],o_shape[3],o_shape[1],o_shape[2]]
        input_=input__.permute(0,3,1,2)
        array = input_.numpy()
        input_siz=array.shape[2]
        padding=(k_h-1) // 2
        output_padding=d_h-1
        calc_deconv = torch.nn.ConvTranspose2d(
            input_.shape[1],output_shape[1],
            kernel_size=[k_h, k_w],
            stride=[d_h,d_w],
            padding=padding,
            output_padding=output_padding
        )
        deconv = calc_deconv(input_,output_size=output_shape)
        deconv = deconv.permute(0,2,3,1)
        #TODO?:add proper bias on deconv
        return deconv
