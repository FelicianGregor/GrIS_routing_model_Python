"""
spatial_moulins_plot.py
=======================
Line-by-line Python translation of spatial_moulins_plot.m: map of the ice
surface elevation with all surface-to-bed connections ("moulins": drained
lakes and full-ice-thickness crevasses) and the Leverett stations Lev0-Lev6.

Uses variables left behind by the main model script:
    moulin_xyz, X, Y, dem
In MATLAB they sit in the shared workspace; in Python pass them in as globals:

    import runpy
    ws = runpy.run_path('Devel_SNOW_DELAY_FULL_MOULINS_AND_LAKES_MODEL_updated.py')
    runpy.run_path('spatial_moulins_plot.py', init_globals=ws)

or, in Jupyter/IPython:  %run -i <main script>  then  %run -i spatial_moulins_plot.py

Notes on behaviour carried over from MATLAB (nothing changed):
  * moulin_xyz(:,3) is moulin(:,i) for the LAST model day; a connection
    persists once formed, so this is every connection made during the season.
  * Off-ice (NaN) cells are set to moulin = 0 from day 2 (ice_limits), so they
    never appear here, even though margin runoff routed into them IS counted
    in melt_to_bed (see note 1 at the end of the main script).
  * `dem` is the SINK-FILLED DEM (overwritten by fillsinks in the main loop).
  * flipud(contourf(...)) has no effect on the figure (it flips the discarded
    contour matrix).
  * `axis image` fixes the limits to the DEM extent before the scatter calls;
    stations outside the DEM are not shown.
  * MATLAB workspace quirk: moulin_site is not cleared before the loop, so
    rows from an older, longer moulin_site in the workspace would survive.
    A fresh Python run always starts empty (= clean MATLAB workspace).
  * If there are no connections, MATLAB stops with an error at
    scatter(moulin_site...); Python stops at the same point.

Differences that cannot be made identical (flagged):
  * contourf levels: MATLAB and Matplotlib choose their own "nice" levels
    (here about 10 requested).
  * Colour 'g': MATLAB 'g' is pure green [0 1 0]; Matplotlib's 'g' is dark
    green (0, 0.5, 0). The MATLAB colour is used explicitly. ('r' is
    identical in both.)
  * The '.' marker is drawn small in both, but at slightly different sizes.
  * Legend swatch for the contours is a proxy grey patch; styling follows
    Matplotlib.
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

# load results using pickle
with open("results/results_routing_model.pkl", "rb") as f:
    ws = pickle.load(f)
# extract results from ws and assign names as used in this script
print(ws)
moulin_xyz = ws["moulin_xyz"]
X = ws["X"]
Y = ws["Y"]
dem = ws["dem"]

moulin_location = moulin_xyz[:, 2]
moulin_eastings = moulin_xyz[:, 0]
moulin_northings = moulin_xyz[:, 1]

Lev_stations = np.zeros((7, 2))
Lev_stations[:, 0] = [537915, 542272, 551765, 569356, 586283, 613888, 649397]
Lev_stations[:, 1] = [7439488, 7442077, 7443529, 7445161, 7446872, 7450845, 7453648]

moulin_index = 1

moulin_site = []
for i in range(1, len(moulin_location) + 1):          # i = 1:length(moulin_location)
    if moulin_location[i-1] > 0:
        moulin_site.append([moulin_eastings[i-1], moulin_northings[i-1]])   # moulin_site(moulin_index,:)

        moulin_index = moulin_index + 1
moulin_site = np.array(moulin_site)

fig = plt.figure()                                     # figure;
ax = plt.gca()                                         # hold on
cs = ax.contourf(X[0, :], Y[:, 1], dem,                # contourf(X(1,:),Y(:,2),dem)
                 levels=MaxNLocator(nbins=10).tick_values(np.nanmin(dem), np.nanmax(dem)),
                 cmap='gray')                          # colormap gray
# flipud(...) of the returned contour matrix: no effect on the plot (see notes)
ax.set_aspect('equal')                                 # axis image
ax.set_xlim(np.min(X[0, :]), np.max(X[0, :]))
ax.set_ylim(np.min(Y[:, 1]), np.max(Y[:, 1]))
h_moulins = ax.scatter(moulin_site[:, 0], moulin_site[:, 1], marker='.',
                       color=(0, 1, 0))                          # scatter(...,'.g')  MATLAB green
h_stations = ax.scatter(Lev_stations[:, 0], Lev_stations[:, 1], marker='^',
                        facecolors='none', edgecolors='r')       # scatter(...,'^r')
ax.set_title('Spatial extent of surface-to-bed connections', fontsize=10, fontweight='bold')
ax.set_xlabel('Eastings (UTM)')
ax.set_ylabel('Northings (UTM)')
h_dem = Patch(facecolor=plt.get_cmap('gray')(0.5), edgecolor='k')   # proxy for the contour object
ax.legend([h_dem, h_moulins, h_stations],
          ['Ice surface elevation', 'Surface-to-bed connection', 'Lev0-Lev6 (west-east)'],
          loc='upper right')                           # 'Location','NorthEast'
fig.colorbar(cs, ax=ax)                                # colorbar;
# hold off
plt.show()
