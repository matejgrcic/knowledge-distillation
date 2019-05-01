import tensorflow as tf

def cross_entropy_loss(student_logits, teacher_logits):
        student_softmax = softmax(student_logits)
        teacher_softmax = softmax(teacher_logits)
        return - tf.reduce_mean(tf.log(student_softmax) * teacher_softmax)

def softmax(logits):
    return tf.transpose(tf.transpose(tf.exp(logits)) / tf.reduce_sum(tf.exp(logits), axis=1))
