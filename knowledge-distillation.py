import tensorflow as tf
from data import get_cifar10_data
import teacher_model
import student_model
from metrics import accuracy

train_x, train_y, valid_x, valid_y, test_x, test_y = get_cifar10_data()
number_of_classes = test_y.shape[1]

teacher_config = {}
teacher_config['max_epochs'] = 8
teacher_config['batch_size'] = 50
teacher_config['num_classes'] = number_of_classes

student_config = {}
student_config['max_epochs'] = 50
student_config['batch_size'] = 50
student_config['num_classes'] = number_of_classes
student_config['temperature'] = 1.5

teacher = teacher_model.DeepModel(train_x, number_of_classes)
student = student_model.DeepModel(train_x, number_of_classes, student_config['temperature'])

def callback_fn(session, config, model):
    model_accuracy = accuracy(valid_x, valid_y, config['batch_size'], config['num_classes'], session, model) 
    print(f"Validation set accuracy: {model_accuracy * 100.}")

session = tf.Session()
session.run(tf.global_variables_initializer())

teacher.train(train_x, train_y, session, teacher_config, callback_fn)
teacher_accuracy = accuracy(test_x, test_y, teacher_config['batch_size'], teacher_config['num_classes'], session, teacher)
print(f"Teacher network test set accuracy: {teacher_accuracy}")

student.train(train_x, session, student_config, teacher, callback_fn)
student_accuracy = accuracy(test_x, test_y, student_config['batch_size'], student_config['num_classes'], session, student)
print(f"Student network test set accuracy: {student_accuracy}")