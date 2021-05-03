#coding=utf-8

import tensorflow as tf

def cal(a, b, c):
    add_op = a + b
    print(add_op)
    mul_op = add_op * c

    init = tf.global_variables_initializer()
    sess = tf.Session()
    
    sess.run(init)
    mul_op_res = sess.run([mul_op])

    return mul_op_res

a = tf.constant(1.0)
b = tf.constant(2.0)
c = tf.constant(3.0)

print(cal(a, b, c))