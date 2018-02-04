# -*- coding: utf-8 -*-

import tensorflow as tf

x = tf.constant([2])
y = tf.constant([4])

z = tf.add(x,y)

with tf.Session() as sesssion:
  result = session.run(z)
  print(result)

Scalar = tf.constant([7])
Vector = tf.constant([5.2,6.1,2.8])
Matrix = tf.constant([[1,2,3],[9,8,7],[6,4,8]])
Tensor = tf.constant( [ [[1,2,3],[2,3,4],[3,4,5]] , [[4,5,6],[5,6,7],[6,7,8]] , [[7,8,9],[8,9,10],[9,10,11]] ] )
with tf.Session() as session:
    result = session.run(Scalar)
    print(result)
    result = session.run(Vector)
    print(result)
    result = session.run(Matrix)
    print(result)
    result = session.run(Tensor)
    print(result)

Matrix_one = tf.constant([[1,2,3],[2,3,4],[3,4,5]])
Matrix_two = tf.constant([[2,2,2],[2,2,2],[2,2,2]])

first_operation = tf.add(Matrix_one, Matrix_two)
second_operation = Matrix_one + Matrix_two

with tf.Session() as session:
    result = session.run(first_operation)
    print("Defined using tensorflow function: ")
    print(result)
    result = session.run(second_operation)
    print("Defined using normal expressions: ")
    print(result)

Matrix_one = tf.constant([[2,3],[3,4]])
Matrix_two = tf.constant([[2,3],[3,4]])

first_operation = tf.matmul(Matrix_one, Matrix_two)

with tf.Session() as session:
    result = session.run(first_operation)
    print("Defined using tensorflow function :")
    print(result)

state = tf.Variable(0)

one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)

init_op = tf.global_variables_initializer()

with tf.Session() as session:
  session.run(init_op)
  print(session.run(state))
  for _ in range(3):
    session.run(update)
    print(session.run(state))

a=tf.placeholder(tf.float32)

b=a**2

with tf.Session() as sess:
    result = sess.run(b,feed_dict={a:5})
    print(result)

dictionary={a: [ [ [1,2,3],[4,5,6],[7,8,9],[10,11,12] ] , [ [13,14,15],[16,17,18],[19,20,21],[22,23,24] ] ] }

with tf.Session() as sess:
    result = sess.run(b,feed_dict=dictionary)
    print(result)

a = tf.constant([5])
b = tf.constant([2])
c = tf.add(a,b)
d = tf.subtract(a,b)

with tf.Session() as session:
    result = session.run(c)
    print(result)
    result = session.run(d)
    print(result)
