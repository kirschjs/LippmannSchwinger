from contextlib import contextmanager
import time


@contextmanager
def benchmark(name):
    start = time.time()
    yield
    end = time.time()

    print('{} took {:.2f} ms\n'.format(name, (end - start) * 1000.0))
