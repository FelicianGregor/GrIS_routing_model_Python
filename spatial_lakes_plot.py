"""
spatial_lakes_plot.py
=====================
Line-by-line Python translation of spatial_lakes_plot.m: map of the ice
surface elevation with the drained supraglacial lakes and the Leverett
stations Lev0-Lev6.

Like the MATLAB script, this uses variables left behind by the main model
script (drainage_xyz, X, Y, dem). In MATLAB they sit in the shared
workspace; in Python pass the main script's results in as globals:

    import runpy
    ws = runpy.run_path('Devel_SNOW_DELAY_FULL_MOULINS_AND_LAKES_MODEL_updated.py')
    runpy.run_path('spatial_lakes_plot.py', init_globals=ws)

or, in Jupyter/IPython, run both in the same namespace:

    %run -i Devel_SNOW_DELAY_FULL_MOULINS_AND_LAKES_MODEL_updated.py
    %run -i spatial_lakes_plot.py

Notes on behaviour carried over from MATLAB (nothing changed):
  * drainage_xyz(:,3) is drained_lakes(:,i) for the LAST model day, i.e.
    lakes that have drained at any point in the season (the state persists).
  * `dem` is the main script's `dem`, which was overwritten by
    fillsinks(dem) inside the day loop -> the map shows the SINK-FILLED DEM,
    not the original surface.
  * flipud(contourf(...)) flips only the returned contour matrix, which is
    discarded, so it has no effect on the figure. Kept as a comment.
  * `axis image` (equal aspect, limits fitted to the data) is applied BEFORE
    the scatter calls and fixes the axis limits; stations outside the DEM
    extent would not be shown. Reproduced by setting the limits explicitly
    (which also switches off Matplotlib's autoscaling).
  * MATLAB workspace quirk: lake_site is not cleared before the loop, so if
    an older, longer lake_site exists in the workspace, its extra rows are
    kept. A fresh Python run always starts empty (= clean MATLAB workspace).
  * If no lake drained, MATLAB stops with an error at scatter(lake_site...);
    Python stops with an error at the same point.

Differences that cannot be made identical (flagged):
  * contourf level choice: MATLAB picks its own "nice" levels; Matplotlib
    picks its own (here asked for about 10). Colours/levels differ slightly.
  * Legend swatch for the filled contours: MATLAB shows one patch for the
    contour object; Matplotlib needs a proxy patch (mid-grey of the colormap).
  * Plot styling (fonts, default sizes) follows Matplotlib, not MATLAB.
"""
import pickle  # for the results to be loaded

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

# load results using pickle
with open("results/results_routing_model.pkl", "rb") as f:
    ws = pickle.load(f)
# extract results from ws and assign names as used in this script
print(ws)
drainage_xyz = ws["drainage_xyz"]
X = ws["X"]
Y = ws["Y"]
dem = ws["dem"]

lakes_locations = drainage_xyz[:, 2]
lakes_eastings = drainage_xyz[:, 0]
lakes_northings = drainage_xyz[:, 1]

Lev_stations = np.zeros((7, 2))
Lev_stations[:, 0] = [537915, 542272, 551765, 569356, 586283, 613888, 649397]
Lev_stations[:, 1] = [7439488, 7442077, 7443529, 7445161, 7446872, 7450845, 7453648]

lakes_index = 1

lake_site = []
for i in range(1, len(lakes_locations) + 1):          # i = 1:length(lakes_locations)
    if lakes_locations[i-1] > 0:
        lake_site.append([lakes_eastings[i-1], lakes_northings[i-1]])   # lake_site(lakes_index,:)

        lakes_index = lakes_index + 1
lake_site = np.array(lake_site)

fig = plt.figure()                                     # figure;
ax = plt.gca()                                         # hold on (Matplotlib keeps adding to ax)
cs = ax.contourf(X[0, :], Y[:, 1], dem,                # contourf(X(1,:),Y(:,2),dem)
                 levels=MaxNLocator(nbins=10).tick_values(np.nanmin(dem), np.nanmax(dem)),
                 cmap='gray')                          # colormap gray
# flipud(...) of the returned contour matrix: no effect on the plot (see notes)
ax.set_aspect('equal')                                 # axis image
ax.set_xlim(np.min(X[0, :]), np.max(X[0, :]))
ax.set_ylim(np.min(Y[:, 1]), np.max(Y[:, 1]))
h_lakes = ax.scatter(lake_site[:, 0], lake_site[:, 1], marker='o',
                     facecolors='none', edgecolors='b')          # scatter(...,'ob')
h_stations = ax.scatter(Lev_stations[:, 0], Lev_stations[:, 1], marker='^',
                        facecolors='none', edgecolors='r')       # scatter(...,'^r')
ax.set_title('Spatial extent of supraglacial lake drainages', fontsize=10, fontweight='bold')
ax.set_xlabel('Eastings (UTM)')
ax.set_ylabel('Northings (UTM)')
h_dem = Patch(facecolor=plt.get_cmap('gray')(0.5), edgecolor='k')   # proxy for the contour object
ax.legend([h_dem, h_lakes, h_stations],
          ['Ice surface elevation', 'Drained lakes', 'Lev0-Lev6 (west-east)'],
          loc='upper right')                           # 'Location','NorthEast'
fig.colorbar(cs, ax=ax)                                # colorbar;
# hold off
plt.show()
