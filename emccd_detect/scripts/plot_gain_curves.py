from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt
import numpy as np


def em_gain_pdf(x, n, g):
    return x**(n-1) * np.exp(-x/g) / (g**n * np.math.factorial(n-1))


gain = 100
n_array = np.arange(1, 7)
x_array = np.arange(0, 2001, 10)
legend_str = []
plt.figure()
for i, n in enumerate(n_array):
    pdf = em_gain_pdf(x_array, n, gain)
    legend_str.append('n = ' + str(n))
    plt.plot(x_array/gain, pdf, linewidth=.5)

plt.xlim(xmin=0)
plt.ylim(ymin=0)
plt.ylabel('probability density')
plt.xlabel('output counts / gain')
plt.legend(legend_str)
plt.show()
