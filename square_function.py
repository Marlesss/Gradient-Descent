import numpy as np


def generate_function(n, cond):
    min_singular = 1
    max_singular = min_singular * cond
    s = np.diag(-np.sort(-np.insert(np.append((max_singular - min_singular) * np.random.sample(n - 2)
                                              + min_singular, min_singular), 0, max_singular)))  # degscending sort
    q, r = np.linalg.qr(np.random.sample(n, n))
    q_t = q.transpose()
    a = np.matmul(q, s, q_t)  # maybe there are only 2 args
    b = np.random.sample(n)
    c = np.random.sample()

    def f(x: np.ndarray):
        return np.dot(x.transpose(), a, x) + np.dot(b, x) + c


    return f

generate_function(3, 2)
