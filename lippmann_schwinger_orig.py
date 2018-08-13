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


# particle and interaction parameters
lambd = 1.5
b = 10
mu = 0.5

# on-shell momentum interval of interest
k0min = 0.02
k0max = 0.6
k0steps = 2900
k0range = np.linspace(k0min, k0max, k0steps)

# number of grid points
M = 24

kgauss = np.zeros(M, float)
#x = np.zeros(M, float)
wgauss = np.zeros(M, float)
Finv = np.zeros((M, M), float)
F = np.zeros((M, M), float)
D = np.zeros(M, float)
V = np.zeros(M, float)
Vvec = np.zeros((M, M - 1), float)

plt.subplot(111)
plt.xlabel(r'$kb$', fontsize=24)
plt.ylabel(r'$\sin^2(\delta)$', fontsize=24)

# calculate grid points and weights via Gaussian quadrature
# the midpoint of the grid acts as a regulator because the
# location of the grid point of largest momentum depends on it
midpoint = (M - 1) / 2
gauss(M - 1, 2, 0., midpoint, kgauss, wgauss)

for k0 in k0range:

    # solution via Gauss quadrature
    kgauss[M - 1] = k0
    for i in range(M - 1):
        D[i] = 2 * (2 * mu) / np.pi * wgauss[i] * kgauss[i]**2 / (
            kgauss[i]**2 - k0**2)
    D[M - 1] = 0.
    for j in range(M - 1):
        D[M - 1] = D[M - 1] + wgauss[j] * k0**2 / (kgauss[j]**2 - k0**2)
    D[M - 1] = D[M - 1] * (-2 * (2 * mu) / np.pi)

    for i in range(M):
        for j in range(M):
            pot = -lambd * np.sin(b * kgauss[i]) * np.sin(b * kgauss[j]) / (
                kgauss[i] * kgauss[j])
            F[i][j] = pot * D[j]
            if i == j:
                F[i][j] = F[i][j] + 1
        V[i] = pot

    for i in range(M):
        Vvec[i][0] = V[i]
    Finv = lina.inv(F)
    R = np.dot(Finv, Vvec)
    RN1 = R[M - 1][0]
    shift = np.arctan(-RN1 * k0)
    sin2 = np.sin(shift)**2
    plt.plot(k0 * b, sin2, 'o', ms=1)
    k0 = k0 + 0.2 * np.pi / 1000

plt.show()