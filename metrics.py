import numpy as np

def accuracy(x, y, batch_size, num_classes, session, model):
    num_examples = x.shape[0]
    assert num_examples % batch_size == 0
    num_batches = num_examples // batch_size
    cnt_correct = 0
    for i in range(num_batches):
        batch_x = x[i*batch_size:(i+1)*batch_size, ...]
        batch_y = y[i*batch_size:(i+1)*batch_size, ...]

        [logits_val] = session.run([model.logits], feed_dict={ model.X: batch_x })
        yp = np.argmax(logits_val, 1)
        yt = np.argmax(batch_y, 1)
        cnt_correct += (yp == yt).sum()
    return cnt_correct / num_examples