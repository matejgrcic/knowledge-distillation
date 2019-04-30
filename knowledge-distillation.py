import tensorflow as tf
from data import get_cifar10_data
import teacher_model
import student_model
from metrics import accuracy

train_x, train_y, valid_x, valid_y, test_x, test_y = get_cifar10_data()
number_of_classes = test_y.shape[1]

teacher = teacher_model.DeepModel(train_x, number_of_classes)
student = student_model.DeepModel(train_x, number_of_classes)

teacher_config = {}
teacher_config['max_epochs'] = 2
teacher_config['batch_size'] = 50
teacher_config['num_classes'] = number_of_classes

student_config = {}
student_config['max_epochs'] = 4
student_config['batch_size'] = 50
student_config['num_classes'] = number_of_classes

session = tf.Session()
session.run(tf.global_variables_initializer())
teacher.train(train_x, train_y, session, teacher_config)
student.train(train_x, session, student_config['batch_size'] ,student_config['max_epochs'], teacher)
student_accuracy = accuracy(test_x, test_y, student_config['batch_size'], student_config['num_classes'], session, student)
print('Student network: test set accuracy:', student_accuracy)
