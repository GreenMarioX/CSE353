import numpy as np
from BasicLinAlg import linalg_gradients, linalg_operations
import sys

A = np.array([[1, 2], [3, 4], [5, 6]])
x = np.array([10, 20])
b = np.array([-5, -6, -7])
c = np.array([np.pi, np.exp(1)])

correct_answers_operations = {'2a': np.array([50, 110, 170]),
                              '2b': 'not possible',
                              '2c': np.array([-58, -76]),
                              '2d': 'not possible',
                              '2e': np.array([50, 110, 170]),
                              '2f': 22.360679774997898,
                              '2g': 1153.6897329871667,
                              '2h': 47810.0}

correct_answers_gradients = {'4a': np.array([20, 40]),
                             '4b': np.array([2460, 3120]),
                             '4c': np.array([69352.00779, 60007.23942593]),
                             '4d': np.array([-7.67340e+08, -1.00548e+09]),
                             '4e': np.array([0., 0]),
                             '4f': np.array([-2.66919022e-108, -8.00757065e-108]),
                             '4g': np.array([0, 0]) * np.nan}

import sys

testnum = '3'#sys.argv[1]

correct = True
if testnum == '1':
    for k, v in correct_answers_operations.items():
        a = linalg_operations(A, b, c, x, k)

        if type(a) == str:
            correct = correct & (a == v)
        else:

            correct = correct & (np.linalg.norm(a - v) < 1.e-6)
    assert (correct)

elif testnum == '2':
    for k, v in correct_answers_gradients.items():
        if np.isnan(np.sum(v)): continue
        correct = correct & (np.linalg.norm(linalg_gradients(A, b, c, x, k) - v) < 1.e-6)
    assert (correct)

elif testnum == '3':

    offset = 0.123 #float(sys.argv[2])
    multiplier = 0.01 #float(sys.argv[3])

    for probnum in ['2a', '2b', '2c', '2d', '2e', '2f', '2g', '2h']:
        answer = linalg_operations(multiplier * A + offset, multiplier * b + offset, multiplier * c + offset,
                                   multiplier * x + offset, probnum)
        print("\nprobnum", probnum)
        if type(answer) == str:
            print(answer, end=';')
        else:
            print(np.round(answer * 1000000) + 1, end=';')

    for probnum in ['4a', '4b', '4c', '4d', '4e', '4f', '4g']:
        answer = linalg_gradients(multiplier * A + offset, multiplier * b + offset, multiplier * c + offset,
                                  multiplier * x + offset, probnum)
        print("\nprobnum", probnum)
        if type(answer) == str:
            print(answer, end=';')
        else:
            print(np.round(answer * 1000000) + 1, end=';')