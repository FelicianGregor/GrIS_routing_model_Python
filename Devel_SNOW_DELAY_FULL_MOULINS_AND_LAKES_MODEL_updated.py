"""
Devel_SNOW_DELAY_FULL_MOULINS_AND_LAKES_MODEL_updated.py
=========================================================
LINE-BY-LINE Python translation of the MATLAB script
Devel_SNOW_DELAY_FULL_MOULINS_AND_LAKES_MODEL_updated.m
(Caroline Clason, Stockholm University, 25 Sept 2013; sensitivity-analysis
edits by Felician Heim, St Andrews).

THIS IS A LITERAL TRANSLATION. The algorithm, loop structure, branch order,
operation order and all known quirks of the MATLAB code are preserved on
purpose, so that results can be compared one-to-one with MATLAB output.
Nothing has been "fixed". Proposed changes are listed (prioritized) as
comments at the END of this file.

NOTE: This code runs extremely slow, so likely (if everything works as intended) the code would need adjustments for making it more efficient

Translation conventions
-----------------------
* Day index `i` is kept 1-BASED exactly as in MATLAB (so tests such as
  `i == 1` and `i <= melt_delay+1` are unchanged). Column access
  X(j,i) is written X[j, i-1]; X(j,i-1) is written X[j, i-2].
* Cell indices (j, I) are 0-based (they are only used for indexing).
* MATLAB `A(:)` (column-major flatten) -> A.ravel(order='F').
  MATLAB `reshape(v, ...)`            -> v.reshape(..., order='F').
  Using NumPy's default C order anywhere would silently scramble the map.
* MATLAB round() -> matlab_round() (halves away from zero; Python rounds
  halves to even).
* NaN comparisons are False in both languages, so every if/elseif chain
  behaves identically when a NaN appears (including the cases where no
  branch fires and a variable silently keeps its previous value).

Random element (MATLAB-specific, see routeflats.py)
---------------------------------------------------
flowdir -> routeflats chooses randomly (randperm) among equally valid
downstream neighbours of flat cells, every day. To make a Python run
reproducible, seed it before running:
    import routeflats; routeflats.seed_routeflats(0)
MATLAB's random stream cannot be reproduced.

How to run (equivalent of the live-script sensitivity analysis)
---------------------------------------------------------------
MATLAB's `if ~exist('Kic','var')` lets a calling live script pre-set
parameters in the shared workspace. The Python equivalent is to run this
file with pre-set globals:

    import runpy
    ws = runpy.run_path('Devel_SNOW_DELAY_FULL_MOULINS_AND_LAKES_MODEL_updated.py',
                        init_globals={'Kic': 120000, 'TS': 70000, 'crev_width': 2})
    ws['melt_delivery_discharge']      # ws is the "workspace" dict

or simply `python Devel_SNOW_DELAY_FULL_MOULINS_AND_LAKES_MODEL_updated.py`
for the defaults. Input files are read from the current working directory,
as in MATLAB.
"""
import numpy as np
import pickle # for exporting variables for later use during plots
from matlab_compat import matlab_round, lin, sortrows, xlsread
from rasterread import rasterread
from fillsinks import fillsinks
from flowdir import flowdir
from flowaccwithlakes import flowaccwithlakes

# This model predicts full ice-thickness crevasses, lake filling and drainage
# and quantifies meltwater delivery to the bed. Crevasses and lakes transfer
# their total accumulated melt fill to the bed when connections are
# made and subsequent days of melt inflow are also routed to the bed.

# !!! This script has been updated from the original used in Clason et al (2012)
# to include delay in transit time of meltwater through cells covered by snow
# and to include refreezing/formation of superimposed ice

# Caroline Clason, Stockholm University, 25th September 2013

# **************************************************************************

# input variables
# Note, that I (Felician Heim, St Andrews University, MRes) change the script in a way so that in case in the live
# script for sensitivity analysis parameters have already been defined,
# this script wont overwrite them by assigning the vars again...

# change only the following parameters:
# K_ic = fracture toughness
# TS = tensile strength
# crev_width = width of crevasses in m

# --- MATLAB lines 24-49 ---------------------------------------------------
met_elev = 457          # elevation (m a.s.l.) of met station/temperature sensor
temp_lapse = 0.0055     # air temperature lapse rate (degC/m)
DDF_i = 7.79            # Degree-day factor for ice (mm w.e. d degC)
DDF_s = 5.81            # Degree-day factor for snow (mm w.e. d degC)
Pi = 918                # density of ice
Pw = 1000               # density of freshwater
P_s = 0.3               # density of new snow (g/cm3)
g = 9.81                # acceleration due to gravity
t = 24                  # time in hours (24 because daily time-step)

# if vars already exist (from live script in sensitivity analysis) then
# dont overwrite them by reassigning them
# (MATLAB-SPECIFIC: exist('Kic','var') -> check the module globals, which are
#  pre-set by runpy.run_path(..., init_globals={...}))
if 'Kic' not in globals(): Kic = 150000          # fracture toughness (Pa m^1/2)
if 'TS' not in globals(): TS = 75000             # tensile strength (Pa)
if 'crev_width' not in globals(): crev_width = 1  # width of crevasses in m

# print current value of vars modified during sensitivity analysis
print('Fracture toughness: %g' % Kic)
print('Tensile strength: %g' % TS)
print('Crevasse width: %g' % crev_width)

cell_width = 500        # width of each grid cell
crev_length = 500       # length of crevasses
percolation_time = 10   # length of delay for meltwater percolation through snow 1m (300mm w.e.) deep (days)
perc_grad = 0.0333      # gradient for change in percolation time through snowpack !!!N.B. must be changed if 'percolation_time' is changed!!!
pmax = 0.6              # pmax factor for formation of superimposed ice after Reeh (1991)

# *************************************************************************

# --- MATLAB lines 53-73 ---------------------------------------------------
# meteorological data
metdata = xlsread('input_data/Lev_met_input09.xlsx', 'A2:C100')  # reads met data
JD = metdata[:, 0]        # julian days
temp = metdata[:, 1]      # average temperature data (degC)
precip = metdata[:, 2]    # daily total precipitation data (mm)

# ice surface elevation data (m)
dem, X, Y, _ = rasterread('input_data/SP_DEM_LEV.asc')   # reads DEM ascii
DEM = dem.ravel(order='F')                    # matrix -> vector (column-major!)
surface_nans = np.isnan(DEM)
SIZ = len(DEM)
size_days = len(JD)
surf_elev = np.zeros(SIZ)

for loop in range(len(DEM)):   # replace nans in DEM with zeros
    if surface_nans[loop] == 1:
        surf_elev[loop] = 0
    else:
        surf_elev[loop] = DEM[loop]

# **************************************************************************

# --- MATLAB lines 77-116 --------------------------------------------------
# adjusts initial snowpack across the catchment with accumulation gradient

s_depth = 0          # starting snowpack depth at met station (mm w.e.)
snow_elev = 457      # elevation of snow depth measurement (m a.s.l.)
accum_grad = 0.2566  # snow accumulation gradient (mm w.e./m)

grad_elev = np.zeros(SIZ)
snow_depth = np.zeros(SIZ)
pmax_snow_depth = np.zeros(SIZ)

for a in range(len(surf_elev)):

    # adjusts snow depth for elevation using accumulation gradient
    if surf_elev[a] < snow_elev:
        grad_elev[a] = (snow_elev - surf_elev[a]) * accum_grad
    elif surf_elev[a] > snow_elev:
        grad_elev[a] = (surf_elev[a] - snow_elev) * accum_grad

    if surf_elev[a] == snow_elev:
        snow_depth[a] = s_depth
    elif surf_elev[a] < snow_elev:
        snow_depth[a] = s_depth - grad_elev[a]
    elif surf_elev[a] > snow_elev:
        snow_depth[a] = s_depth + grad_elev[a]

    if snow_depth[a] <= 0:
        snow_depth[a] = 0
    else:
        snow_depth[a] = snow_depth[a]

    # Pmax applied to winter snowpack at beginning of melt season to
    # determine refreezing potential in each cell following Schuler et al (2007)
    pmax_snow_depth[a] = snow_depth[a] * pmax

del grad_elev   # clear grad_elev

###############################
# potentail changes: 
# use a simple linear regression to predict the snow depth at any point, in second step use rule to make snow depth 0 when elevation lower than station
################################

# **************************************************************************

# --- MATLAB lines 120-144 -------------------------------------------------
# tensile stress data (kPa)
tensile, _, _, _ = rasterread('input_data/tensile_stresses_LEV.asc')
Rxx_pa = tensile * 1000            # convert kPa to Pa
Rxx = Rxx_pa.ravel(order='F')      # matrix -> vector
RR = Rxx_pa < TS                   # assign a runoff ratio of zero where melt meets cell >= tensile strength (1 if "no crevasse")
                                   # (NaN < TS is False -> RR = 0 on NaN cells, same as MATLAB)

# ice thickness data (m)
h, _, _, _ = rasterread('input_data/ice_thickness_LEV.asc')
H = h.ravel(order='F')             # matrix -> vector

# lake data
lakes, _, _, _ = rasterread('input_data/lake_volume_LEV.asc')               # reads lake volume (10^6 m^3)
LV = lakes * 1000000000 / (cell_width * cell_width)             # convert 10^6 m^3 to mm
LV_vector = LV.ravel(order='F')

# ADDED CHECK (no effect on valid input): MATLAB would fail later with an
# index/dimension error if the grids did not match; NumPy might broadcast.
assert tensile.shape == dem.shape == h.shape == lakes.shape, \
    'DEM, tensile stress, ice thickness and lake grids must have the same size'

# georeference data
eastings = X.ravel(order='F')
northings = Y.ravel(order='F')

# assigns 1s to areas outside the ice surface boundary
ice_limits = np.isnan(DEM)

# no melting across the catchment until after day 1 calculation
melt_start = 0

# **************************************************************************

# --- MATLAB lines 148-181 -------------------------------------------------
# pre-allocate space for model outputs
SIZE = LV.shape
LVactual = np.zeros(SIZE)
lapse_elev = np.zeros(SIZ)
temp_elev = np.zeros(SIZ)
pos_temp = np.zeros(SIZ)
snow = np.zeros(SIZ)
precip_snow = np.zeros(SIZ)
snow_total = np.zeros((SIZ, size_days))
perc_lapse = np.zeros(SIZ)
melt_delay = np.zeros(SIZ)
daily_melt = np.zeros(SIZ)
daily_melt_inc = np.zeros(SIZ)
daily_melt_inc_days = np.zeros((SIZ, size_days))
pmax_filling = np.zeros((SIZ, size_days))
cumu_melt = np.zeros((SIZ, size_days))
delay_timer = np.zeros((SIZ, size_days))
dmax = np.zeros(SIZ)
b_lake = np.zeros(SIZ)
B_lake = np.zeros(SIZ)
Ki_lake = np.zeros(SIZ)
crevasse_depth = np.zeros(SIZ)
full_crevasses = np.zeros((SIZ, size_days))
drained_lakes = np.zeros((SIZ, size_days))
lake_drainage_events = np.zeros((SIZ, size_days))
melt_to_bed = np.zeros((SIZ, size_days))
b = np.zeros((SIZ, size_days))
B = np.zeros(SIZ)
moulin = np.zeros((SIZ, size_days))
total_daily_melt_delivery = np.zeros(size_days)
melt_delivery_volume = np.zeros(size_days)
melt_delivery_discharge = np.zeros(size_days)
tot_melt = np.zeros(size_days)
tot_tot_melt = np.zeros(size_days)

# MATLAB grows melt_xyz implicitly on first assignment; pre-allocated here
melt_xyz = np.zeros((SIZ, 3))

# **************************************************************************

# model begins calculating melt, crevasse depths and lake filling/drainage
# for each time step from here

# loops through days   (i is 1-BASED, as in MATLAB)
for i in range(1, len(temp) + 1):
    # print the day we are at
    print('calculating melt, fracture depth and lake filling/drainage for day ' + str(i))

    # loops through each DEM cell
    for j in range(len(surf_elev)):

        # ******************************************************************

        # Degree-day melt model component   (MATLAB lines 199-234)

        # adjusts temperature for changing elevation using lapse rate
        # (if surf_elev == met_elev, lapse_elev keeps its previous value, as in MATLAB;
        #  it is not used in that case)
        if surf_elev[j] < met_elev:
            lapse_elev[j] = (met_elev - surf_elev[j]) * temp_lapse
        elif surf_elev[j] > met_elev:
            lapse_elev[j] = (surf_elev[j] - met_elev) * temp_lapse

        if surf_elev[j] == met_elev:
            temp_elev[j] = temp[i-1]
        elif surf_elev[j] < met_elev:
            temp_elev[j] = temp[i-1] + lapse_elev[j]
        elif surf_elev[j] > met_elev:
            temp_elev[j] = temp[i-1] - lapse_elev[j]

        # sets daily average temperatures to zero when below 0degC
        # (if temp_elev is NaN neither branch fires and pos_temp keeps the
        #  previous day's value -- identical in MATLAB and Python)
        if temp_elev[j] >= 0:
            pos_temp[j] = temp_elev[j]
        elif temp_elev[j] < 0:
            pos_temp[j] = 0

        if ice_limits[j] == 1:
            pos_temp[j] = 0

        # determines whether precipitation is falling as snow or is rain and neglected (when above 1degC)
        if temp_elev[j] <= 1:
            snow[j] = precip[i-1] * P_s    # converts snow in mm to mm w.e.
            precip_snow[j] = snow[j]
        else:
            snow[j] = 0
            precip_snow[j] = snow[j]

        # adds fallen snow to initial snowpack
        if i == 1:
            snow_total[j, i-1] = snow_depth[j] + precip_snow[j]
        else:
            snow_total[j, i-1] = snow_total[j, i-2] + precip_snow[j]

        # also need to add time delay dependent on distance "upstream" -
        # i.e. link with ice surface elevation... BUT since percolation is a
        # much slower process than flow through the basal saturated layer,
        # and thus makes up a much larger proportion of delayed flow,
        # perhaps can avoid this??

        # adjusts percolation time through snow of varying thickness   (MATLAB lines 249-268)
        # !!! MATLAB-SPECIFIC: `snow_total(j)` is LINEAR indexing into the
        # SIZ x size_days matrix. Because j <= SIZ it always reads
        # snow_total(j,1), i.e. the DAY-1 snowpack, on every day.
        # Reproduced literally with lin(); this equals snow_total[j, 0].
        if lin(snow_total, j) < 300:   # depth in mm w.e.
            perc_lapse[j] = (300 - lin(snow_total, j)) * perc_grad
        elif lin(snow_total, j) > 300:
            perc_lapse[j] = (lin(snow_total, j) - 300) * perc_grad

        if lin(snow_total, j) == 300:
            melt_delay[j] = percolation_time
        elif lin(snow_total, j) < 300:
            melt_delay[j] = matlab_round(percolation_time - perc_lapse[j])
        elif lin(snow_total, j) > 300:
            melt_delay[j] = matlab_round(percolation_time + perc_lapse[j])

        if melt_delay[j] <= 0:
            melt_delay[j] = 0
        else:
            melt_delay[j] = matlab_round(melt_delay[j])

        # calculates daily melt total, refreezing and meltwater transit delay in snowpack
        # (MATLAB lines 272-329)
        if i == 1:
            if snow_total[j, i-1] == 0:
                cumu_melt[j, i-1] = melt_start + (pos_temp[j] * DDF_i)        # apply DDF for ice when no snow cover
                pmax_filling[j, i-1] = pmax_snow_depth[j] - cumu_melt[j, i-1]  # start to fill up pmax space in snowpack with melt
                delay_timer[j, i-1] = 0                                        # no delay in meltwater transit time when cell has no snow cover
                daily_melt[j] = cumu_melt[j, i-1]
                daily_melt_inc[j] = daily_melt[j]                              # daily melt including that lost as superimposed ice/refreezing and in the meltwater routing delay
                daily_melt_inc_days[j, i-1] = daily_melt[j]                    # daily melt stored over full 99 days for later use in control simulation
            else:
                cumu_melt[j, i-1] = melt_start + (pos_temp[j] * DDF_s)        # apply DDF for snow when snowpack not depleted
                pmax_filling[j, i-1] = pmax_snow_depth[j] - cumu_melt[j, i-1]  # start to fill up pmax space in snowpack with melt
                delay_timer[j, i-1] = melt_delay[j]                            # initial delay time
                daily_melt[j] = 0                                              # set daily melt to zero when meltwater is delayed in a snow-covered cell to account for slower transit time
                daily_melt_inc[j] = cumu_melt[j, i-1]
                daily_melt_inc_days[j, i-1] = cumu_melt[j, i-1]                # daily melt stored over full 99 days for later use in control simulation
        elif i <= (melt_delay[j] + 1):   # no melt passed on in snow-covered cells for the first 'x' days of the run goverened by delay length
            if cumu_melt[j, i-2] > snow_total[j, i-1]:
                cumu_melt[j, i-1] = cumu_melt[j, i-2] + (pos_temp[j] * DDF_i)
                pmax_filling[j, i-1] = pmax_snow_depth[j] - cumu_melt[j, i-1]  # fill up pmax space in snowpack with melt
                delay_timer[j, i-1] = 0                                        # no longer a delay as snow has been melted
                daily_melt[j] = cumu_melt[j, i-1] - cumu_melt[j, i-2]
                daily_melt_inc[j] = daily_melt[j]
                daily_melt_inc_days[j, i-1] = daily_melt[j]                    # daily melt stored over full 99 days for later use in control simulation
            else:
                cumu_melt[j, i-1] = cumu_melt[j, i-2] + (pos_temp[j] * DDF_s)
                pmax_filling[j, i-1] = pmax_snow_depth[j] - cumu_melt[j, i-1]  # fill up pmax space in snowpack with melt
                delay_timer[j, i-1] = melt_delay[j]
                daily_melt[j] = 0
                daily_melt_inc[j] = cumu_melt[j, i-1] - cumu_melt[j, i-2]
                daily_melt_inc_days[j, i-1] = cumu_melt[j, i-1] - cumu_melt[j, i-2]  # daily melt stored over full 99 days for later use in control simulation
        else:
            if cumu_melt[j, i-2] > snow_total[j, i-1]:
                cumu_melt[j, i-1] = cumu_melt[j, i-2] + (pos_temp[j] * DDF_i)
                pmax_filling[j, i-1] = pmax_snow_depth[j] - cumu_melt[j, i-1]  # fill up pmax space in snowpack with melt
                delay_timer[j, i-1] = 0                                        # no longer a delay as snow has been melted
                daily_melt[j] = cumu_melt[j, i-1] - cumu_melt[j, i-2]
                daily_melt_inc[j] = daily_melt[j]
                daily_melt_inc_days[j, i-1] = daily_melt[j]                    # daily melt stored over full 99 days for later use in control simulation
            else:
                if pmax_filling[j, i-2] <= 0:
                    cumu_melt[j, i-1] = cumu_melt[j, i-2] + (pos_temp[j] * DDF_s)
                    pmax_filling[j, i-1] = pmax_snow_depth[j] - cumu_melt[j, i-1]  # fill up pmax space in snowpack with melt
                    delay_timer[j, i-1] = melt_delay[j]
                    # MATLAB: cumu_melt(j,i-delay_timer(j,i)) - cumu_melt(j,(i-delay_timer(j,i)-1))
                    # 1-based column (i - d) -> 0-based (i - d - 1); safe because i > d + 1 here
                    d_t = int(delay_timer[j, i-1])
                    daily_melt[j] = cumu_melt[j, (i - d_t) - 1] - cumu_melt[j, (i - d_t - 1) - 1]
                    daily_melt_inc[j] = cumu_melt[j, i-1] - cumu_melt[j, i-2]
                    daily_melt_inc_days[j, i-1] = cumu_melt[j, i-1] - cumu_melt[j, i-2]  # daily melt stored over full 99 days for later use in control simulation
                else:
                    cumu_melt[j, i-1] = cumu_melt[j, i-2] + (pos_temp[j] * DDF_s)
                    pmax_filling[j, i-1] = pmax_snow_depth[j] - cumu_melt[j, i-1]
                    delay_timer[j, i-1] = melt_delay[j]
                    daily_melt[j] = 0
                    daily_melt_inc[j] = cumu_melt[j, i-1] - cumu_melt[j, i-2]
                    daily_melt_inc_days[j, i-1] = cumu_melt[j, i-1] - cumu_melt[j, i-2]  # daily melt stored over full 99 days for later use in control simulation

    # --- MATLAB lines 333-343 ---------------------------------------------
    # creates an xyz file of E, N and melt for each day
    melt_xyz[:, 0] = eastings
    melt_xyz[:, 1] = northings
    melt_xyz[:, 2] = daily_melt

    # converts xyz melt values into grid format
    xyz = melt_xyz
    xyz = sortrows(xyz, [1, -2])                      # sort the rows by easting and then northing
    height = int(np.sum(xyz[:, 0] == xyz[0, 0]))       # grid height
    width = int(np.sum(xyz[:, 1] == xyz[0, 1]))        # grid width
    melt_grid = xyz[:, 2].reshape((height, width), order='F')

    # **********************************************************************

    # --- MATLAB lines 347-368 ---------------------------------------------
    # Melt routing and accumulation component, including filling of lakes

    # calculates weighted flow accumulation
    W0 = melt_grid                                       # daily melt input for weighting
    cs = abs(lin(Y, 1) - lin(Y, 0))                      # cell size   (MATLAB Y(2)-Y(1): linear index -> Y(2,1)-Y(1,1)); unused
    siz = dem.shape                                      # matrix dimensions; unused
    dem = fillsinks(dem)                                 # fills sinks in DEM
    Ms = flowdir(X, Y, dem, type='single')               # single flow direction algorithm
    As, LVactual = flowaccwithlakes(Ms, W0, RR, LV, LVactual)   # calculates flow accumulation and filling of lakes
    with np.errstate(divide='ignore', invalid='ignore'):        # MATLAB gives Inf/NaN silently for 0/0, x/0
        LV_percent_fill = 100 * (LVactual / LV)                 # fill percentage of lake maximum volume

    # rate of crevasse filling for input to fracture model (m/hr)
    q = (As / t) / 1000
    Q = q.ravel(order='F')

    # lake filling for drainage of lakes through fractures (m/hr)
    q_lake = (LVactual / t) / 1000
    Q_lake = q_lake.ravel(order='F')

    # converts runoff ratio matrix to vector(this will be modified as lakes
    # drain, adding to cells through which water is no longer routed)
    RR = np.asarray(RR).ravel(order='F').copy()

    # **********************************************************************

    # Fracture depth model and lake drainage component and melt delivery to the bed

    # (Where solving for Ki or Ki_lake on LHS, a simple LEFM model (after
    # Van der Veen, 2007) for calculating depths of water-filled crevasses
    # is applied. The first term on RHS is the stress intentsity factor (Ki1)
    # associated with a tensile stress, term two (Ki2) represents the
    # lithostatic stress in ice with density Pi, and the third term (Ki3)
    # represents the effect of water pressure).
    #
    # NOTE: x**1.5 of a NEGATIVE number gives a COMPLEX result in MATLAB but
    # NaN (with a RuntimeWarning) in NumPy. This can only happen with negative
    # ice thickness in the input; flagged, not expected for valid data.

    for I in range(len(surf_elev)):

        # ******************************************************************

        # lake drainage component   (MATLAB lines 385-442)

        # runs fracture model for lake drainage only for cells where lakes
        # are present
        if LV_vector[I] > 0:

            # fracture depth is equal to ice thickness
            dmax[I] = H[I]

            # melt fill in lakes converted to melt head (m)
            b_lake[I] = Q_lake[I] * t

            # adjusts fill for cell size and crevasse dimensions
            B_lake[I] = (b_lake[I] * (cell_width * cell_width)) / (crev_width * crev_length)

            # this is solved when B_lake is suffient for Ki > Kic for depth equal to ice thickness
            Ki_lake[I] = (1.12 * (Rxx[I] * (np.sqrt(np.pi * dmax[I])))) - (0.683 * (Pi * (g * (dmax[I] ** 1.5)))) + (0.683 * (Pw * (g * (B_lake[I] ** 1.5))))

            if i == 1:
                # no drainage if Ki not greater than Kic on first day
                if Ki_lake[I] < Kic:
                    crevasse_depth[I] = 0
                    drained_lakes[I, i-1] = 0
                    lake_drainage_events[I, i-1] = 0
                    melt_to_bed[I, i-1] = 0
                    RR[I] = Rxx[I] < TS
                # accumulated lake fill transfered to bed and lake cell added to runoff ratio grid if lake drains
                elif Ki_lake[I] >= Kic:
                    crevasse_depth[I] = dmax[I]
                    drained_lakes[I, i-1] = 1
                    lake_drainage_events[I, i-1] = 1
                    melt_to_bed[I, i-1] = ((Q_lake[I]) * t) * 1000   # (converted back to mm w.e)
                    RR[I] = drained_lakes[I, i-1] == 0
            else:
                # registers that a lake has previously drained and continues to route daily melt input to the bed
                if drained_lakes[I, i-2] == 1:
                    crevasse_depth[I] = dmax[I]
                    drained_lakes[I, i-1] = 1
                    lake_drainage_events[I, i-1] = 0
                    melt_to_bed[I, i-1] = ((Q[I]) * t) * 1000        # (converted back to mm w.e)
                    RR[I] = drained_lakes[I, i-1] == 0
                # no drainage if Ki still not greater than Kic
                elif Ki_lake[I] < Kic:
                    crevasse_depth[I] = 0
                    drained_lakes[I, i-1] = 0
                    lake_drainage_events[I, i-1] = 0
                    melt_to_bed[I, i-1] = 0
                    RR[I] = Rxx[I] < TS
                # accumulated lake fill transfered to bed and lake cell added to runoff ratio grid when lake drains
                elif Ki_lake[I] >= Kic:
                    crevasse_depth[I] = dmax[I]
                    drained_lakes[I, i-1] = 1
                    lake_drainage_events[I, i-1] = 1
                    melt_to_bed[I, i-1] = ((Q_lake[I]) * t) * 1000   # (converted back to mm w.e)
                    RR[I] = drained_lakes[I, i-1] == 0

        # where lakes are not present fracture model runs as normal
        else:   # LV_vector(I) == 0   (also NaN, since NaN > 0 is False)

            dmax[I] = 0
            b_lake[I] = 0
            B_lake[I] = 0
            Ki_lake[I] = 0
            drained_lakes[I, i-1] = 0
            RR[I] = Rxx[I] < TS

            # **************************************************************

            # component for water-filled crevasse depth calculation and melt transfer
            # (MATLAB lines 456-520)

            # calculates crevasse depths only if tensile stress >= tensile
            # strength (initial surface fracture present)
            if Rxx[I] >= TS:

                d_start = 1e-7       # initial fracture depth value above zero to ensure initiation
                d_increment = 1      # increment of crevasse depth
                d = d_start          # starting crevasse depth
                Ki = 1 + Kic         # starts at a value greater than Kic so while loop is run at least once

                # continues iterating through values of depth (thus, crevasse
                # propagates as long as Ki > fracture toughness)until Ki is
                # equal to or less than the fracture toughness
                while Ki > Kic:

                    # total depth of water in 1x1 after 24 hours (converted from rate of filling 'Q')
                    B[I] = Q[I] * t

                    if i == 1:
                        # depth of water in crevasse on day 1 (adjusted for
                        # cell size and crevasse dimensions)
                        b[I, i-1] = (B[I] * (cell_width * cell_width)) / (crev_width * crev_length)
                    else:
                        # cumulative depth of water head
                        b[I, i-1] = b[I, i-2] + ((B[I] * (cell_width * cell_width)) / (crev_width * crev_length))

                    # continually solved for increasing values of depth until Ki <= Kic
                    Ki = (1.12 * (Rxx[I] * (np.sqrt(np.pi * d)))) - (0.683 * (Pi * (g * (d ** 1.5)))) + (0.683 * (Pw * (g * (b[I, i-1] ** 1.5))))

                    d = d + d_increment

                # (as in MATLAB, d has already been incremented once more after
                #  the last Ki evaluation, i.e. depth is reported one step deeper)
                crevasse_depth[I] = d

            # crevasse depth is zero for all cells with tensile stress < tensile strength
            else:   # Rxx(I) < TS
                crevasse_depth[I] = 0

            # melt is delivered to the bed when crevasse depth >= ice thickness
            if i == 1:
                if crevasse_depth[I] < H[I]:
                    full_crevasses[I, i-1] = 0
                    melt_to_bed[I, i-1] = 0
                else:   # crevasse_depth(I) >= H(I)
                    # !!! also taken when H(I) is NaN (NaN comparison is False),
                    # as in MATLAB: every non-lake cell with NaN ice thickness is
                    # flagged as a full-depth crevasse on day 1 and stays so.
                    # Off-ice cells receive margin runoff via routeflats, which
                    # is therefore counted as bed delivery (see note 1 at the end).
                    full_crevasses[I, i-1] = 1
                    # if crevasse reaches bed on day 1, first day of melt fill is routed to the bed
                    melt_to_bed[I, i-1] = ((Q[I]) * t) * 1000   # (converted back to mm w.e)
            else:
                if full_crevasses[I, i-2] == 1:
                    full_crevasses[I, i-1] = 1
                    # if crevasse reached bed on a previous day, melt is transfered to bed on all subsequent days
                    melt_to_bed[I, i-1] = ((Q[I]) * t) * 1000
                elif crevasse_depth[I] >= H[I]:
                    full_crevasses[I, i-1] = 1
                    # when crevasse first reaches bed, all accumulated melt content drains
                    melt_to_bed[I, i-1] = (b[I, i-1] * ((crev_width * crev_length) / (cell_width * cell_width))) * 1000
                else:   # crevasse_depth(I) < H(I)
                    full_crevasses[I, i-1] = 0
                    melt_to_bed[I, i-1] = 0

        # ******************************************************************

        # 'moulin' denotes any surface-to-bed connection (whether through
        # lake drainage or a full ice thickness crevasse). A connection is
        # formed when melt is first delivered to the bed and persists
        # throughout the remainder of the modelled ablation season.
        # (MATLAB lines 531-546; note ice_limits is only applied from day 2)

        if i == 1:
            if melt_to_bed[I, i-1] == 0:
                moulin[I, i-1] = 0
            else:
                moulin[I, i-1] = 1
        else:
            if moulin[I, i-2] == 1:
                moulin[I, i-1] = 1
            elif melt_to_bed[I, i-1] == 0:
                moulin[I, i-1] = 0
            else:
                moulin[I, i-1] = 1
            if ice_limits[I] == 1:
                moulin[I, i-1] = 0

    # **********************************************************************

    # --- MATLAB lines 551-568 ---------------------------------------------
    # reshapes runoff ratios to grid format (updated with any drained lakes)
    RR = RR.reshape(SIZE, order='F')

    # **CHECK THIS IS CALCULATED CORRECTLY!**
    # total melt generated at surface each day (mm w.e./day) N.B. excludes
    # melt lost to refreezing/formation of superimposed ice via pmax and
    # stuck in meltwater routing
    tot_melt[i-1] = np.sum(daily_melt)

    # Total generated melt including that lost to refreezing and melt routing
    tot_tot_melt[i-1] = np.sum(daily_melt_inc)

    # total meltwater (mm w.e./day) delivered to the bed through connections each day
    total_daily_melt_delivery[i-1] = np.sum(melt_to_bed[:, i-1])

    # convert melt to bed from specific discharge (mm w.e./day) to actual discharge (cumecs)
    melt_delivery_volume[i-1] = (total_daily_melt_delivery[i-1] / 1000) * (cell_width * cell_width)   # converts from mm to m^3
    melt_delivery_discharge[i-1] = (melt_delivery_volume[i-1] / 86400)                                 # convert to m^3/s

# --- MATLAB lines 572-587 -------------------------------------------------
# total melt generated at surface over whole time period (excluding
# superimposed ice if pmax is implemented and routing delay)
total_generated_melt = np.sum(tot_melt)

# total melt generated at surface over whole time period (including
# superimposed ice if pmax is implemented and routing delay)
total_generated_melt_inc = np.sum(tot_tot_melt)

# total meltwater delivered to the ice-bed interface over the whole time period
total_melt_to_bed = np.sum(total_daily_melt_delivery)

with np.errstate(divide='ignore', invalid='ignore'):   # MATLAB: x/0 -> Inf/NaN silently
    # percentage of surface melt reaching the bed over whole time period
    percent_melt_to_bed = (total_melt_to_bed / total_generated_melt_inc) * 100

    # percentage of surafce melt stored supraglacially through refreezing and percolation
    percent_melt_stored_supra = ((total_generated_melt_inc - total_generated_melt) / total_generated_melt_inc) * 100


# **************************************************************************

# --- MATLAB lines 592-617 -------------------------------------------------
# data conversion for plotting and grid display
# (MATLAB-SPECIFIC: the loop variable i keeps its LAST value after the loop;
#  Python does the same. i is 1-based -> column i-1.)

# creates an xyz file of E, N and moulin (all connections) formation
moulin_xyz = np.zeros((SIZ, 3))
moulin_xyz[:, 0] = eastings
moulin_xyz[:, 1] = northings
moulin_xyz[:, 2] = moulin[:, i-1]

# converts xyz locations of moulins (all connections) into grid format
xyzm = moulin_xyz
xyzm = sortrows(xyzm, [1, -2])                    # sort the rows by easting and then northing
heightm = int(np.sum(xyzm[:, 0] == xyzm[0, 0]))    # grid height
widthm = int(np.sum(xyzm[:, 1] == xyzm[0, 1]))     # grid width
moulin_grid = xyzm[:, 2].reshape((heightm, widthm), order='F')   # changes to grid format

# creates an xyz file of E, N and lake drainage
drainage_xyz = np.zeros((SIZ, 3))
drainage_xyz[:, 0] = eastings
drainage_xyz[:, 1] = northings
drainage_xyz[:, 2] = drained_lakes[:, i-1]

# converts xyz locations of total drainage events into grid format
# (NB: this is the drained STATE on the last day, not a count of events)
xyzm = drainage_xyz
xyzm = sortrows(xyzm, [1, -2])                    # sort the rows by easting and then northing
heightm = int(np.sum(xyzm[:, 0] == xyzm[0, 0]))    # grid height
widthm = int(np.sum(xyzm[:, 1] == xyzm[0, 1]))     # grid width
drainage_grid = xyzm[:, 2].reshape((heightm, widthm), order='F')  # changes to grid format

# Save the results needed for plotting:
ws = {"drainage_xyz": drainage_xyz, 
        "moulin_xyz": moulin_xyz,
        "dem": dem, 
        "X": X, 
        "Y": Y, 
        "JD": JD, 
        "DEM": DEM, 
        "drained_lakes": drained_lakes, 
        "Rxx": Rxx, 
        "LV_vector": LV_vector} # save to dictionary


# save to results folder
with open("results/results_routing_model.pkl", "wb") as f:
    pickle.dump(ws, f)


# ##########################################################################
# ##########################################################################
#
#  PROPOSED STRUCTURAL CHANGES  (NOT implemented above -- listed in order of
#  priority; numerical correctness first, then clarity, then speed)
#
# ##########################################################################
#
#  1. CHECK HOW MARGIN RUNOFF IS COUNTED (correctness, highest priority)
#     routeflats lowers NaN (off-ice) cells to min(dem)-1, so flat ice cells
#     at the margin are routed INTO NaN cells (MATLAB and Python alike).
#     Those cells have NaN (or zero) ice thickness, so on day 1
#     `crevasse_depth < H` is False and they become permanent "full
#     crevasses": all margin runoff is then counted in melt_to_bed /
#     melt_delivery_discharge, although it leaves the ice SURFACE at the
#     margin. (moulin is zeroed there from day 2 via ice_limits, but
#     melt_to_bed is still summed.) In the synthetic test 30% of all "bed
#     delivery" came from off-ice cells.
#     -> In the MATLAB output check  sum(melt_to_bed(ice_limits,:),'all')
#        relative to total_melt_to_bed. If it is non-zero, decide
#        scientifically whether margin runoff should be reported separately
#        (e.g. as "supraglacial runoff at the margin") and add an
#        off-by-default switch for it (see 4f).
#
#  2. RANDOMNESS IN FLAT ROUTING (correctness / reproducibility)
#     routeflats.m picks among several valid downstream neighbours of a
#     flat's perimeter cell with randperm(8). Because flowdir is recomputed
#     every day, MATLAB runs are not exactly reproducible, and Python cannot
#     reproduce MATLAB's random stream. Python uses a seeded generator
#     (routeflats.seed_routeflats(seed)).
#     -> Quantify the spread: run the model for several seeds (and, in
#        MATLAB, for several rng() seeds) and compare melt_delivery_discharge.
#        If the spread is negligible, a fixed neighbour order is acceptable
#        (an algorithm change: must be documented).
#
#  3. VALIDATION HARNESS AGAINST MATLAB OUTPUT (correctness)
#     Save from one MATLAB run (default parameters): fillsinks(dem), the
#     flowdir result [r,c] = find(Ms) for day 1, daily_melt (days 1, 10, 50,
#     99), As, LVactual, melt_to_bed, crevasse_depth, moulin,
#     melt_delivery_discharge. Compare with the Python run:
#       - exact equality expected for fillsinks, snow/melt variables and
#         flags; flowdir identical except at flat-perimeter cells where
#         MATLAB's random choice differs (point 2);
#       - to compare routing-dependent outputs exactly, load the MATLAB Ms
#         into Python for that test only;
#       - |rel. diff| < 1e-12 expected for As, LVactual, melt_to_bed given
#         the same Ms (summation-order round-off only);
#       - any flipped Ki > Kic / Ki < Kic decision must be investigated
#         (possible only at round-off level through the x**1.5 power function).
#     Also confirm the MATLAB version used for the original results:
#     unique() returned the LAST occurrence before R2013a and the FIRST
#     occurrence since (Python: first), which changes tie-breaking in
#     routeflats.
#
#  4. EXPLICIT MASS-BALANCE DIAGNOSTIC (correctness / mass conservation)
#     Add (as extra outputs, without changing any result) a daily budget:
#        generated melt (sum daily_melt_inc)
#        = delivered to bed (split into ice cells / off-ice cells, point 1)
#          + stored in undrained lakes + stored in not-yet-full crevasses
#          (b converted to mm) + water ending in ice cells with no outflow
#          + refrozen / delayed in snow + residual.
#     Known leaks / phantoms this would expose (all present in MATLAB):
#       a) melt "in transit" in the snowpack is never released when a cell
#          becomes snow-free (it is counted as stored, not delivered);
#       b) overspill from an UNDRAINED lake in a cell with Rxx >= TS
#          (RR = 0) is absorbed by the routing but not added to any crevasse
#          store or to melt_to_bed;
#       c) after a lake drains, LVactual is not reset, so the lake store
#          refills on paper while the same inflow is also sent to the bed
#          via As (no double delivery, but the store is a phantom);
#       d) water reaching NaN-LV cells is stored without limit
#          (MATLAB min() ignores NaN) -- in LVactual, while the same water is
#          also counted in As and hence in melt_to_bed (point 1);
#       e) margin runoff counted as bed delivery (point 1).
#
#  5. OPTIONAL, OFF-BY-DEFAULT FLAGS FOR SUSPECTED BUGS (science decisions)
#     Keep current behaviour as default; add switches to test sensitivity:
#       a) percolation delay from the CURRENT snowpack snow_total(j,i)
#          instead of the day-1 value (linear-indexing quirk);
#       b) crevasse depth = last depth with Ki > Kic (remove the extra
#          +1 m step) or a finer depth increment;
#       c) apply ice_limits to moulin on day 1 too;
#       d) use >= instead of > in the snow-gone test so bare-ice cells use
#          DDF_i on their first melt day;
#       e) reset LVactual to 0 when a lake drains (see 4c);
#       f) exclude off-ice / NaN-thickness cells from full_crevasses and
#          report their inflow as margin runoff (see point 1).
#
#  6. COMPUTE fillsinks / flowdir ONCE, BEFORE THE DAY LOOP (speed)
#     The DEM never changes. NOTE: because of the random choice in
#     routeflats (point 2) this is NOT result-identical to MATLAB: MATLAB
#     draws a new flat routing every day, a single computation fixes one
#     realisation. Decide after point 2. fillsinks alone can be moved out
#     of the loop with identical results.
#
#  7. REPLACE THE exist()/globals MECHANISM BY A FUNCTION (clarity)
#     run_model(Kic=150000, TS=75000, crev_width=1, data_dir='.', seed=0)
#     returning a results dict / xarray Dataset. Makes the sensitivity
#     analysis a plain loop and removes hidden state between runs.
#
#  8. VECTORISE THE MELT COMPONENT OVER CELLS (speed; exact equivalent)
#     All per-cell branches can be written with boolean masks (np.where),
#     keeping matlab_round, the day-1 snowpack quirk and the branch order.
#     Must be checked to be bit-identical to the loop version before use.
#
#  9. VECTORISE THE CREVASSE-DEPTH SEARCH (speed; exact equivalent)
#     Evaluate Ki on the SAME depth sequence d = 1e-7 + k*1 for all crevassed
#     cells at once (or solve for the first k analytically and verify).
#     The literal while-loop is the slowest part of the model when water
#     heads b become large (b is multiplied by 500 and accumulates).
#
# 10. REPLACE THE xyz/sortrows GRIDDING BY reshape(..., order='F')
#     (clarity/speed). Equivalent for the north-up grids that rasterread
#     returns; keep an assertion that both give the same grid.
#
# 11. INPUT VALIDATION AND OUTPUT FILES (robustness)
#     Check for NaN in temp/precip, negative or NaN ice thickness inside the
#     ice mask, and grid alignment (same header) of all rasters; write the
#     outputs (grids as GeoTIFF/ASCII, time series as CSV/NetCDF).
#
# ##########################################################################
