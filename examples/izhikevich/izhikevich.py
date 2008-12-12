
from numpy import *
from pylab import plot, show, find, rand, randn

# Created by Eugene M. Izhikevich, February 25, 2003
# Excitatory neurons   Inhibitory neurons
Ne = 800
Ni = 200

re = rand(Ne)
ri = rand(Ni)

a = r_[0.02 * ones(Ne), 0.02 + 0.08 * ri]
b = r_[0.2 * ones(Ne), 0.25 - 0.05 * ri]
c = r_[-65 + 15 * re**2, -65 * ones(Ni)]
d = r_[8 - 6 * re**2, 2 * ones(Ni)]
S = c_[0.5 * rand(Ne + Ni, Ne), -rand(Ne + Ni, Ni)]

v = -65 * ones(Ne + Ni)# Initial values of v
u = b*v                   # Initial values of u
firings = zeros((0,2))

for t in xrange(1000):# simulation of 1000 ms
    I = r_[5 * randn(Ne), 2 * randn(Ni)] # thalamic input
    fired = find(v >= 30)# indices of spikes
    if any(fired):
        firings = vstack((firings, c_[t + 0 * fired, fired]))
        v[fired] = c[fired]
        u[fired] = u[fired] + d[fired]
        I = I + S[:,fired].sum(1)
    v = v + 0.5 * (0.04 * v**2 + 5 * v + 140 - u + I)
    v = v + 0.5 * (0.04 * v**2 + 5 * v + 140 - u + I)
    u = u + a*(b*v - u)

plot(firings[:,0], firings[:,1], '.')
show()
