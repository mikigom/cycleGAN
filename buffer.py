# https://github.com/carpedm20/simulated-unsupervised-tensorflow/blob/master/buffer.py
import numpy as np
import random

class Buffer(object):
    def __init__(self, buffer_size, batch_size):
        self.rng = random.SystemRandom()
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.cur_capa = 0
        self.data = []

    def push(self, batches):
        assert type(batches).__module__ == np.__name__
        batch_size = len(batches)
        if self.cur_capa + batch_size > self.buffer_size:
            for i in range(batch_size):
                self.data[random.randrange(0, self.cur_capa)] = batches[i]
        else:
            for new_data in batches:
                self.data.append(new_data)
                self.cur_capa += 1

    def sample(self, n):
        assert self.cur_capa >= n
        return np.asarray([self.rng.choice(self.data) for i in range(n)])

    def __str__(self):
        return str(self.data)

class config:
    buffer_size = 10
    batch_size = 2

if __name__ == '__main__':
    buffer = Buffer(config())
    a = np.array([[1], [2]])
    b = np.array([[3], [4]])
    buffer.push(a)
    print(buffer)
    buffer.push(b)
    print(buffer)
    print(buffer.sample(3))
    c = np.array([[5], [6]])
    d = np.array([[7], [8]])
    e = np.array([[9], [10]])
    f = np.array([[11], [12]])
    buffer.push(c)
    buffer.push(d)
    buffer.push(e)
    print(buffer)
    print(buffer.sample(5))
    buffer.push(f)
    print(buffer)
