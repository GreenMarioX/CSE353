import math

import numpy as np


# look at accompanying practice_linear_alg.pdf to know what problems to code up

def linalg_operations(A, b, c, x, which_problem):
    if which_problem == '2a':
        return np.dot(A, x)

    if which_problem == '2b':
        return 'not possible'

    if which_problem == '2c':
        return np.dot(A.T, b)

    if which_problem == '2d':
        return 'not possible'

    if which_problem == '2e':
        return np.dot(x.T, A.T)

    if which_problem == '2f':
        return np.linalg.norm(x, 2)

    if which_problem == '2g':
        return np.linalg.norm(b, 2) ** 3

    if which_problem == '2h':
        Ax_minus_b = np.dot(A, x) - b
        result = np.linalg.norm(Ax_minus_b, 2) ** 2
        return result


def linalg_gradients(A, b, c, x, which_problem):
    if which_problem == '4a':
        return 2 * x

    if which_problem == '4b':
        return 2 * np.dot(np.dot(A.T, A), x)

    if which_problem == '4c':
        return 3 * np.dot(np.dot(c.T, x) ** 2, c)

    if which_problem == '4d':
        return 3 * np.dot(np.dot(np.dot(np.dot(b.T, A), x) ** 2, b.T), A)

    if which_problem == '4e':
        m, n = A.shape
        grad = np.zeros(n)
        for i in range(m):
            if b[i] - np.dot(A[i], x) > 0:
                print(A[i])
                grad -= A[i]
        return grad

    if which_problem == '4f':
        return math.exp((- np.linalg.norm(x - 5) ** 2)) * -2 * (x - 5)

    if which_problem == '4g':
        z = np.dot(b, np.dot(A, x))
        sigmoid = 1 / (1 + np.exp(-z))
        return np.dot(A.T, b * (1 - sigmoid))


offset = 0.123 #float(sys.argv[2])
multiplier = 0.01 #float(sys.argv[3])
Aa = np.array([[1, 2], [3, 4], [5, 6]])
xa = np.array([10, 20])
ba = np.array([-5, -6, -7])
ca = np.array([np.pi, np.exp(1)])

answer = linalg_gradients(multiplier * Aa + offset, multiplier * ba + offset, multiplier * ca + offset, multiplier * xa + offset, "4e")
if type(answer) == str:
    print(answer, end=';')
else:
    print(np.round(answer * 1000000) + 1, end=';')