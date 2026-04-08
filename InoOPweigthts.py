#import pandas as pd
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/home/12581631/bhac_code/bhac/tools/python')
import read
# import seaborn as sns

from math import sin, exp, pi, log, sqrt
from scipy import integrate

e = 4.8032068e-10
c = 29979245800.000000
me = 9.1093897e-28
mp = 1.672621777e-24
h = 6.6260755e-27

mec2overh = me*c**2/h

rextract = 100.

# Values from uniform simulation grid
B = 3.3606681876646900
ne = 5978637930.8871096
theta_e = 18.361513033085089
nu_min = 1.0e8
nu_max = 1.0e16
theta_min = 0.001
theta_max = (pi / 2.0) - 0.00001


pd_destroyed=read.particles(1,file='output/data2000',filetype='destroyed')

pd_destroyed.data


def jnu_integrand(theta, nu):
    nus = 2.0/9.0 * ((e * B)/(2.0 * pi * me * c)) * theta_e * theta_e * sin(theta)
    X = nu/nus
    a = ((sqrt(X) + (2.0**(11.0/12.0) * X**(1.0/6.0)))**2.0 * exp(-X**(1.0/3.0)))
    b = (sqrt(2.0) * pi * e * e * ne * nus) / (3.0 * c * sp.special.kn(2, 1.0 / theta_e))
    return a * b * sin(theta) * (4.0 * pi)

N = 50
nu_space = np.logspace(np.log(nu_min), np.log(nu_max), N, base=np.e)
nulnu = np.zeros(N)

for i, nu in enumerate(nu_space):
	nulnu[i], _ = sp.integrate.quad(jnu_integrand, theta_min, theta_max, args=(nu))
	nulnu[i] = nu * nulnu[i]
      


nulnu_particles = np.zeros(N-1)
dlognu=(np.log(nu_max) - np.log(nu_min))/(N-1)
for i in range(N-1):
    numin=nu_space[i]; numax=nu_space[i+1]
    for particle in pd_destroyed.data:
        if (exp(particle['x'][0]) < rextract or exp(particle['x'][0]) > rextract*1.01): continue
        if (particle['q']*mec2overh >numin and particle['q']*mec2overh<=numax):
            nulnu_particles[i] = nulnu_particles[i] + h*particle['m']*particle['q']*mec2overh
nulnu_particles = nulnu_particles/dlognu
#nulnu_particles5=nulnu_particles


nulnu_particles_destroyed=nulnu_particles





##################### observational data


df_sgrA = pd.read_csv("/home/12581631/bhac_code/bhac/tests/particles/FMtorus3D/SGRA_FLUX_FREQ.csv")

# add [df_sgrA["obs"]==True] to filter only observed data
author = list(set(df_sgrA["ref"]))
#print(author)

cmap = plt.get_cmap("nipy_spectral")  # or "gist_ncar", "plasma"

colors = cmap(np.linspace(0, 1, len(author)))

markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'X']
g=0


#####################



# plotting 


#fig, ax = plt.subplots(2, 2, figsize=(27, 22))
"""
#f, ax = plt.subplots(figsize=(7, 5))
ax[0, 0].plot(nu_space, nulnu)

ax[0, 0].set_xlabel(r'$\nu$[Hz]')
ax[0, 0].set_ylabel(r'log($\nu L_{\nu}$ [erg $s^{-1}$])')
ax[0, 0].set_yscale('log')
ax[0, 0].set_xscale('log')

#ax.set_ylim(1e-18,1e-14)



#ax=plt.gca()
#ax.plot(nu_space[:-1], nulnu_particles0)
#ax.plot(nu_space[:-1], nulnu_particles1)
ax[0, 1].plot(nu_space[:-1], nulnu_particles_destroyed)
#ax.plot(nu_space[:-1], nulnu_particles_destroyed_hr)
#ax.plot(nu_space, nulnu,'--')
#ax.plot(fcheck[0], 3e-15*fcheck[1]*fcheck[0],'--')
#ax.plot(fcheck[0], 7e-17*fcheck[2]*fcheck[0],'--')
ax[0, 1].set_xscale('log')
ax[0, 1].set_yscale('log')
ax[0, 1].set_xlabel(r'$\nu$[Hz]')
ax[0, 1].set_ylabel(r'log($\nu L_{\nu}$ [erg $s^{-1}$])')
ax[0, 1].set_ylim(1e29,1e37)


#ax=plt.gca()
ax[1, 0].plot(nu_space[:-1], nulnu_particles_destroyed/nu_space[:-1]/1e23)
#ax.plot(nu_space[:-1], nulnu_particles_destroyed/nu_space[:-1]/1e23)
#ax.plot(nu_space[:-1], nulnu_particles_destroyed_hr/nu_space[:-1]/1e23)
#ax.plot(nu_space, nulnu,'--')
#ax.plot(fcheck[0], 3e-15*fcheck[1]*fcheck[0],'--')
#ax.plot(fcheck[0], 7e-17*fcheck[2]*fcheck[0],'--')
ax[1, 0].set_xscale('log')
ax[1, 0].set_yscale('log')
ax[1, 0].set_xlabel(r'$\nu$[Hz]')
ax[1, 0].set_ylabel(r'log($L_{\nu}$ [Jy])')
ax[1, 0].set_ylim(1e-3,1e2)
"""

for i, color in zip(sorted(author), colors):
    
    freq_sgrA = df_sgrA[df_sgrA["obs"]==True][df_sgrA["ref"]==i]["freq"]* 1e9 # convert to Hz
    flux_sgrA = df_sgrA[df_sgrA["obs"]==True][df_sgrA["ref"]==i]["flux"] * 1e23 # convert to cgs

    yerr_lower = df_sgrA[df_sgrA["obs"]==True][df_sgrA["ref"]==i]["lerr"] *freq_sgrA
    yerr_upper = df_sgrA[df_sgrA["obs"]==True][df_sgrA["ref"]==i]["uerr"] *freq_sgrA
    plt.errorbar(
    freq_sgrA, flux_sgrA*freq_sgrA,xerr=0,
    yerr=[yerr_lower, yerr_upper],fmt=markers[g % len(markers)],color=color,markersize=2,capsize=3,linestyle="none",label=i)
    g+=1

plt.xlabel(r"$\nu \ [Hz]$")
plt.ylabel(r"$\nu L_\nu \ [erg/s]$")
plt.plot(nu_space[:-1], nulnu_particles_destroyed, linestyle="--", color="grey", label="Simulation")
plt.yscale("log")
plt.xscale("log")
plt.ylim(1e29,1e37)

plt.savefig('fnrnof.png', dpi=300)

