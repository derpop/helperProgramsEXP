#!/usr/bin/env python3

import os
import yaml
import pyEXP
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.colors import LogNorm
plt.rcParams['figure.figsize'] = [12, 9]

# The Tutorials/Data directory contains the test data
#

bconfig = """
---
id: cylinder
parameters:
  acyl: 1.0
  hcyl: 0.1
  lmaxfid: 32
  mmax: 6
  nmaxfid: 32
  nmax: 12
  ncylodd: 3
  ncylnx: 128
  ncylny: 64
  rnum: 200
  pnum: 0
  tnum: 80
  ashift: 0
  vflag: 0
  logr: false
  cachename: .eof.cache.file
  self_consistent: true
  npca: 100
  pcadiag: true
  pcavtk: true
  tk_type: Hall
...
"""


basis = pyEXP.basis.Basis.factory(bconfig)


coef_file = 'outcoef.star disk.run0'
coefs = pyEXP.coefs.Coefs.factory(coef_file)

size = 0.01  # Spatial extent
npix = 50    # Number of pixels per dimension
pmin = [-size, -size, 0.0]
pmax = [size, size, 0.0]
grid = [npix, npix, 0]

# Retrieve all available time steps
times = coefs.Times()

# Initialize the field generator
fields = pyEXP.field.FieldGenerator(times, pmin, pmax, grid)

# Generate the density surfaces for all time steps
surfaces = fields.slices(basis, coefs)

# Create a directory to save the plots
output_dir = 'density_plots'
os.makedirs(output_dir, exist_ok=True)

print(surfaces[times[0]].keys())


total = len(times)

for i, time_step in enumerate(times):
        density_data = surfaces[time_step]['dens']
        x = np.linspace(pmin[0], pmax[0], density_data.shape[0])
        y = np.linspace(pmin[1], pmax[1], density_data.shape[1])
        xv, yv = np.meshgrid(x, y)

                        # Set minimum positive value for log scale (avoid log(0))
        min_nonzero = np.min(density_data[density_data > 0])
        norm = LogNorm(vmin=min_nonzero, vmax=np.max(density_data))
        plt.figure()

        vmin = np.percentile(density_data, 5)
        vmax = np.percentile(density_data, 95)
        norm = Normalize(vmin=np.percentile(density_data, 5),vmax=np.percentile(density_data, 95))

        plt.contourf(xv, yv, density_data.T, cmap='viridis', norm=norm)

        plt.colorbar(label='Density (lin scale)')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'Density at T={time_step:.2f}')

        filename = os.path.join(output_dir, f'density_{i:05d}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

        percent = (i + 1) / total * 100
        print(f'Saved plot for T={time_step:.2f} as {filename} ({percent:.2f}% complete')
