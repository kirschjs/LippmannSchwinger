import matplotlib.pyplot as plt
import numpy.linalg as lina
import numpy as np

from util import benchmark

eps = np.finfo(float).eps


def gauss(npts, job, a, b):
    x = np.zeros(npts + 1, float)
    w = np.zeros(npts + 1, float)
    m = i = j = t = t1 = pp = p1 = p2 = p3 = 0.
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
    x[npts] = 0.
    w[npts] = 0.
    return [x, w]


def interaction(lec, ki, kj, type=1):
    if type == 0:
        return lec[0] * np.exp(-(lec[1] / (ki - kj))**2)
    # test potential as used in "Computational Physics (Problem Solving with Python)" by
    # Rubin H. Landau, Manuel J. Páez, Cristian C. Bordeianu, Eq. 26.38 p.603
    # delta-shell: l*d(r-b)
    if type == 1:
        return lec[0] * np.sin(lec[1] * ki) * np.sin(lec[1] * kj) / (
            ki * kj + eps)


def calc_wave_function(r, Finv, k):
    return sum(
        [np.sin(k[i] * r) / (k[i] * r) * Finv[i] for i in range(len(k))])


def plot_wave_function(r, Finv, k):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel(r'$r [fm]$', fontsize=24)
    ax.set_ylabel(r'$u(r)$', fontsize=24)
    wfkt = [calc_wave_function(rr, Finv, k) for rr in r]
    ax.plot(r, wfkt, 'o', ms=3, alpha=.25, label=r'')
    plt.show()
    exit()


def plot_grids(grids):
    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.set_xlabel(r'$b$', fontsize=24)
    #plt.ylabel(r'$\sin^2(\delta)$', fontsize=24)
    ax.set_ylim(.9, len(grids) * 0.2 + 1)
    y0 = 1
    for grd in grids:
        xx = grd
        yy = [y0 for n in grd]
        y0 += 0.2
        ax.plot(xx, yy, 'o', ms=3, alpha=.25, label=r'')
        ax.plot(y0, np.mean(yy), 'or', ms=5, alpha=.25, label=r'')
    ax = fig.add_subplot(122)
    ax.set_xlabel(r'nbr. of grid points', fontsize=24)
    #    ax.plot(
    #        [len(gr) for gr in grids], [np.mean(y0) for y0 in grids],
    #        'or',
    #        ms=2,
    #        alpha=.25,
    #        label=r'mean')
    ax.plot(
        [len(gr) for gr in grids], [np.median(y0) for y0 in grids],
        'ob',
        ms=2,
        alpha=.25,
        label=r'median')
    leg = plt.legend(ncol=1, fontsize=12)
    plt.show()
    exit()


def V_grid_k0(kgrid, kenergy, LECs):

    # number of momentum-grid points
    nGrid = len(kgrid)
    # number of energies for which "stuff" is calculated
    nEnergy = len(kenergy)

    Vofk0 = np.zeros((nEnergy, nGrid, nGrid), float)

    Vgrid = np.zeros((nGrid, nGrid), float)
    for i in range(nGrid):
        for j in range(nGrid):

            Vgrid[i][j] = interaction(LECs[0], kgrid[i], kgrid[j], LECs[1])
    for n0 in range(nEnergy):
        vkk = np.array(
            [[interaction(LECs[0], kenergy[n0], kenergy[n0], LECs[1])]])
        Vvec0 = [
            interaction(LECs[0], ki, kenergy[n0], LECs[1]) for ki in kgrid
        ]
        Vvec0[-1] = vkk

        Vgrid[:, -1] = Vvec0
        Vgrid[-1, :] = Vvec0

        Vofk0[n0] = Vgrid

    return zwei_m_over_hbar_sq * Vofk0


# on-shell momentum interval of interest
k0min = 0.02
k0max = 0.6
k0steps = 1000
k0range = np.linspace(k0min, k0max, k0steps)

# particle and interaction parameters
# hbar**2/m_N = 41.47 MeV*fm^2 = 1/(2mu)
MeVfm = 197.3161329
mn = 938.91852
# throughout, we express all quantities in MeV, hence
zwei_m_over_hbar_sq = mn

# coordinate-space grid for the wave function
plowefu = 1
nplo = 200
rmin = 0.001
rmax = 120
rgrid = np.linspace(rmin, rmax, 500)

# number of grid points, last point is replaced by the momentum of the
# energy of interest, i.e., 1,...,M-1 points are given by some quadrature
# rule and 1, the M-th, is equal to k_0;

M = 140

Finv = np.zeros((M, M), float)
F = np.zeros((M, M), float)
D = np.zeros(M, float)
V = np.zeros(M, float)
results = []

# calculate grid points and weights via Gaussian quadrature
# the midpoint of the grid acts as a regulator because the
# location of the grid point of largest momentum depends on it
midpoint = (M - 1) / 2
textbookgaussgrid = gauss(M - 1, 2, 0., midpoint)
legendre = np.polynomial.legendre.leggauss(M)
cheb = np.polynomial.chebyshev.chebgauss(M)
laguerre = np.polynomial.laguerre.laggauss(M)
hermite = np.polynomial.hermite.hermgauss(M)
hermite_e = np.polynomial.hermite_e.hermegauss(M)

# coordinate-space delta shell = l/(2mu) sin(k'b)sin(kb)/(k'k)
#        alpha a          l: strength  b:displacement   alpha*delta(r-a)
LECs = [[1.5, 10], 1]

grids = [[textbookgaussgrid, LECs, 'Gauss'], [legendre, LECs, 'Legendre'],
         [hermite, LECs, 'Hermite'], [cheb, LECs, 'Chebyshev'],
         [laguerre, LECs, 'Laguerre'], [hermite_e, LECs, 'Hermite_e']]

# visualization of the Gauss mesh as a function of the number
# of points on it; lim_m->inf(<p>)=inf *BUT* lim_m->inf(median[p])<inf
#plot_grids([gauss(m, 2, 0., midpoint)[0] for m in range(10, 500, 15)])

with benchmark("Matrix inversion on Gaussian grid:"):

    # calculate phase shifts for each quadrature polynomial, i.e., momentum grid
    for grd in grids[:2]:

        k, w = grd[0][0], grd[0][1]
        pot = V_grid_k0(k, k0range, grd[1])
        tmpres = []
        kstep = 0

        for k0 in k0range:

            k[-1] = k0

            # D matrix '=' (free propagator)*k^2
            for i in range(M - 1):
                D[i] = 2 / np.pi * w[i] * k[i]**2 / (k0**2 - k[i]**2)
            D[-1] = 0.

            for j in range(M - 1):
                D[M - 1] = D[-1] + w[j] * k0**2 / (k0**2 - k[j]**2)
            D[-1] = D[-1] * (-2 / np.pi)

            F = np.identity(M) - D * pot[kstep, :, :]
            Finv = lina.inv(F)
            R = np.dot(Finv, pot[kstep, -1, :])
            RN1 = R[-1]

            # UNITS: R(on-shell)=-tan(d)/(2mk/h^2)
            shift = np.arctan(-RN1 * k0)  # * zwei_m_over_hbar_sq)

            #tmpres.append(180. / np.pi * shift)
            tmpres.append(np.sin(shift)**2)

            if ((kstep == nplo) & (plowefu == 1)):
                plot_wave_function(rgrid, Finv[:, -1], k)
            kstep += 1
        results.append([tmpres, grd[2]])

plt.subplot(111)
plt.xlabel(r'$k^2$', fontsize=24)
plt.ylabel(r'$\delta$', fontsize=24)

for res in results:
    plt.plot(k0range, res[0], '-', lw=2, alpha=.5, label=r'%s' % res[1])

# analytical solution; arccot(x) = arctan(1/x)
if LECs[1] == 1:
    analytical_res = np.array([(np.angle(
        np.sin(kk * LECs[0][1]) + kk /
        (zwei_m_over_hbar_sq * LECs[0][0]) * np.exp(1j * kk * LECs[0][1])) -
                                kk * LECs[0][1]) for kk in k0range])
    #analytical_res = np.array([(
    #    np.arctan(1. /
    #              (1. / np.tan(kk * LECs[0][1]) + zwei_m_over_hbar_sq * LECs[0]
    #               [0] * LECs[0][1] * kk * LECs[0][1])) - kk * LECs[0][1])
    #                           for kk in k0range])

    analytical_res = np.sin(analytical_res)**2
    plt.plot(
        k0range, analytical_res, '--r', lw=1, alpha=1, label=r'analytical')

leg = plt.legend(ncol=1, fontsize=12)
plt.show()