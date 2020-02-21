# -*- coding: utf-8 -*-
import copy
import time

import matplotlib.pyplot as plt
import numpy as np

from rand_em_gain import rand_em_gain


em_gain = 5700.0

ntimes = 100
limits = [0, 2]
step = 0.001

n_array = np.arange(limits[0], limits[1]+step, step)
n_array = n_array[1:]
t_array = np.zeros(len(n_array))
meanx = np.zeros(len(n_array))
x = np.zeros(ntimes)

num = 0
for n_in in n_array:
    t = time.time()
    for i in range(ntimes):
        x[i] = rand_em_gain(n_in, em_gain)
    tt = time.time() - t
    t_array[num] = tt

    meanx[num] = np.mean(x)
    num += 1

expected = n_array * em_gain

# find percent error ((actual - expected) / expected)
mean_minus_expected = ((meanx - expected) / expected) * 100

# get int values from n_array
ints = copy.copy(n_array)
ints_i = np.where((ints % 1) == 0)
ints = ints[ints_i].astype(int)

ints_meanx = meanx[ints_i]
ints_mean_minus_expected = mean_minus_expected[ints_i]
ints_t = t_array[ints_i]

# find mean of mean_minus_expected
one_i = ints_i[0][0]
avg = np.mean(mean_minus_expected[one_i:])


# plot x out
plt.figure()
plt.plot(n_array, meanx, linewidth=0.5)
plt.plot(ints, ints_meanx, linestyle='None', marker='o')

plt.xticks(ints)
plt.xlabel('n_in')
plt.ylabel('x_out')
plt.title('rand_em_gain: Output vs. Input\ngain = ' + str(int(em_gain)))
plt.grid(alpha=0.5, axis='x')

# plot (actual - expected) / expected
plt.figure()
plt.plot(n_array, mean_minus_expected, linewidth=0.5)
plt.plot(ints, ints_mean_minus_expected, linestyle='None', marker='o')

a = plt.plot(n_array[one_i:], np.ones(len(n_array[one_i:])) * avg, '--r')
plt.legend(a, ['mean (n_in > 1) = ' + str(round(avg, 3)) + '%'])

plt.xticks(ints)
plt.xlabel('n_in')
plt.ylabel('% error')
plt.title('rand_em_gain: Percent Error\ngain = ' + str(int(em_gain)))
plt.grid(alpha=0.5, axis='x')

# plot time
plt.figure()
plt.plot(n_array, t_array, linewidth=0.5)
plt.plot(ints, ints_t, linestyle='None', marker='o')

plt.xticks(ints)
plt.xlabel('n_in')
plt.ylabel('time (sec)')
plt.title('rand_em_gain: Time vs. Input')
plt.grid(alpha=0.5, axis='x')
