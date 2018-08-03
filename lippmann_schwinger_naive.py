import matplotlib.pyplot as plt
import numpy.linalg as lina
import numpy as np


def gauss(npts, job, a, b, x, w):
    m = i = j = t = t1 = pp = p1 = p2 = p3 = 0.
    eps = 3e-14
    m = (npts + 1) / 2
    for i in range(1, int(m) + 1):
        t = np.cos(np.pi * (float(i) - 0.25) / (float(npts) + 0.5))
        t1 = 1
        while ((abs(t - t1)) >= eps):
            p1 = 1.
            p2 = 0.
            for j in range(1, npts + 1):
                p3 = p2
                p2 = p1
                p1 = ((2. * float(j) - 1) * t * p2 -
                      (float(j) - 1.) * p3) / (float(j))
            pp = npts * (t * p1 - p2) / (t * t - 1.)
            t1 = t
            t = t1 - p1 / pp
        x[i - 1] = -t
        x[npts - i] = t
        w[i - 1] = 2. / ((1. - t * t) * pp * pp)
        w[npts - i] = w[i - 1]
    if (job == 0):
        for i in range(npts):
            x[i] = x[i] * (b - a) / 2 + (b + a) / 2
            w[i] = w[i] * (b - a) / 2
    if (job == 1):
        for i in range(npts):
            xi = x[i]
            x[i] = a * b * (1 + xi) / (b + a - (b - a) * xi)
            w[i] = w[i] * 2. * a * b * b / ((b + a - (b - a) * xi) *
                                            (b + a - (b - a) * xi))
    if (job == 2):
        for i in range(npts):
            xi = x[i]
            x[i] = (b * xi + b + a + a) / (1 - xi)
            w[i] = w[i] * 2. * (a + b) / ((1 - xi) * (1 - xi))


M = 27
b = 10
n = 26

scale = n / 2
lambd = 1.5

k = np.zeros(M, float)
x = np.zeros(M, float)
w = np.zeros(M, float)
Finv = np.zeros((M, M), float)
F = np.zeros((M, M), float)
D = np.zeros(M, float)
V = np.zeros(M, float)
Vvec = np.zeros((n + 1, n), float)

plt.subplot(111)
plt.xlabel(r'$kb$', fontsize=24)
plt.ylabel(r'$\sin^2(\delta)$', fontsize=24)
plt.xlim(0.0, 6)

gauss(n, 2, 0., scale, k, w)

k0 = 0.02

for m in range(1, 930):
    k[n] = k0
    for i in range(n):
        D[i] = 2 / np.pi * w[i] * k[i]**2 / (k[i]**2 - k0**2)
    D[n] = 0.
    for j in range(n):
        D[n] = D[n] + w[j] * k0**2 / (k[j]**2 - k0**2)
    D[n] = D[n] * (-2 / np.pi)
    for i in range(n + 1):
        for j in range(n + 1):
            pot = -lambd * np.sin(b * k[i]) * np.sin(b * k[j]) / (k[i] * k[j])
            F[i][j] = pot * D[j]
            if i == j:
                F[i][j] = F[i][j] + 1
        V[i] = pot
    for i in range(n + 1):
        Vvec[i][0] = V[i]
    Finv = lina.inv(F)
    R = np.dot(Finv, Vvec)
    RN1 = R[n][0]
    shift = np.arctan(-RN1 * k0)
    sin2 = np.sin(shift)**2
    plt.plot(k0 * b, sin2, 'o', ms=1)
    k0 = k0 + 0.2 * np.pi / 1000

plt.show()