# -*- coding: utf-8 -*-
"""
tstress_lakes_spatial_temporal.py
=================================
Line-by-line Python translation of tstress_lakes_spatial_temporal.m:
scatter plot of lake drainage events (day of year vs. surface elevation,
marker size scaled by tensile stress) and summary counts of drained lakes.

Uses variables left behind by the main model script:
    JD, DEM, drained_lakes, Rxx, LV_vector

Notes on behaviour carried over from MATLAB (nothing changed):
  * new_lake_locs = day-to-day difference of the drained state, so it is 1
    only on the day a lake drains (the same as lake_drainage_events).
  * elevation = DEM is the ORIGINAL surface vector (not the sink-filled dem);
    events in cells with elevation <= 0 or NaN are skipped (lake_elev > 0).
  * Marker size = tstress/10 with Rxx in Pa, i.e. 7500 points^2 for a lake
    at 75 kPa: very large markers, exactly as in MATLAB.
    UNCERTAINTY (flagged): a lake that drains under COMPRESSIVE stress
    (Rxx < 0) gets a negative marker size. MATLAB may refuse negative sizes;
    Matplotlib silently draws nothing for them. A warning is printed if this
    occurs (no change to the plot).
  * total_drained_lakes uses day 99 HARD-CODED (drainage_events(:,99)), not
    the last day: with fewer than 99 days MATLAB stops with an index error
    (Python too); with more days it still reports day 99.
  * daily_drainage (drainages per day) is computed but not used afterwards.
  * MATLAB workspace quirk: daily_drainage, new_lake_locs, lake_elev,
    lake_tstress and lakey are not cleared, so rows/columns from an older,
    larger run in the same workspace would survive (lakey in particular
    would keep stale events). A fresh Python run always starts empty
    (= clean MATLAB workspace). The script also overwrites the main
    script's loop variables i and j in MATLAB (irrelevant here).
  * If no lake drained, MATLAB stops with an error at scatter(lakey...);
    Python stops at the same point.
  * fprintf('%d') of a whole-number double prints an integer in MATLAB;
    Python's '%d' does the same (the values are counts, so always whole).
  * plt.show() is placed at the end so the text is printed first, as in
    MATLAB (where the figure does not block the script).
"""
import pickle
import warnings
import numpy as np
import matplotlib.pyplot as plt

# load results using pickle
with open("results/results_routing_model.pkl", "rb") as f:
    ws = pickle.load(f)
# extract results from ws and assign names as used in this script
JD = ws["JD"]
DEM = ws["DEM"]
drained_lakes = ws["drained_lakes"]
Rxx = ws["Rxx"]
LV_vector = ws["LV_vector"]

#assign to vars used for plotting
days = JD
elevation = DEM
drainage_events = drained_lakes
tstress = Rxx

lake_index = 1

# MATLAB grows these arrays inside the loop (zero-filled); pre-allocated here
daily_drainage = np.zeros((1, len(days)))
new_lake_locs = np.zeros((len(elevation), len(days)))
lake_elev = np.zeros((len(elevation), len(days)))
lake_tstress = np.zeros((len(elevation), len(days)))
lakey = []

for i in range(1, len(days) + 1):                     # i = 1:length(days)   (1-based)
    if i == 1:
        daily_drainage[:, i-1] = np.sum(drainage_events[:, i-1])
    else:
        daily_drainage[:, i-1] = np.sum(drainage_events[:, i-1]) - np.sum(drainage_events[:, i-2])

    for j in range(len(elevation)):
        if i == 1:
            new_lake_locs[j, i-1] = drainage_events[j, i-1]
        else:
            new_lake_locs[j, i-1] = drainage_events[j, i-1] - drainage_events[j, i-2]

        if new_lake_locs[j, i-1] == 1:
            lake_elev[j, i-1] = elevation[j]
            lake_tstress[j, i-1] = tstress[j]
        else:
            lake_elev[j, i-1] = 0
            # lake_melt_delivery(j,i) = 0;

        if lake_elev[j, i-1] > 0:
            lakey.append([lake_elev[j, i-1],               # lakey(lake_index,1)
                          days[i-1],                       # lakey(lake_index,2)
                          lake_tstress[j, i-1]])           # lakey(lake_index,3)

            lake_index = lake_index + 1
lakey = np.array(lakey)

fig = plt.figure()                                         # figure;
if np.any(lakey[:, 2] / 10 < 0):
    warnings.warn('Some lakes drained under compressive stress (Rxx < 0): their '
                  'marker size tstress/10 is negative and they are not drawn.')
plt.scatter(lakey[:, 1], lakey[:, 0], s=(lakey[:, 2] / 10),
            marker='o', facecolors='none', edgecolors='b')   # scatter(...,(lakey(:,3)/10),'ob')
# (lakey(:,2),lakey(:,1),100,'ob')
plt.title('Distribution of lake drainage events', fontsize=10, fontweight='bold')
plt.xlabel('Day of Year')
plt.ylabel('Elevation (m a.s.l.)')
# legend ('Surface-to-bed connection','Location','SouthEast')

total_drained_lakes = np.sum(drainage_events[:, 98])       # drainage_events(:,99)  -- day 99 hard-coded
lakes_total = np.sum(LV_vector > 0)                        # NaN > 0 is False, as in MATLAB
with np.errstate(divide='ignore', invalid='ignore'):       # MATLAB: x/0 -> Inf/NaN silently
    percent_lakes_drained = np.sum(total_drained_lakes / lakes_total) * 100

# print
print('Total drained lakes: %d' % total_drained_lakes)
print('Total lakes: %d' % lakes_total)
print('Percentage drained lakes: %.2f%%' % percent_lakes_drained)


# tot_melt = sum(daily_melt);
# total_generated_melt = sum(tot_melt);
# percent_melt_to_bed = (total_melt_to_bed/total_generated_melt)*100;

plt.show()
