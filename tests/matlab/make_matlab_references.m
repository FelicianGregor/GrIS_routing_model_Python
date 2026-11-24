% make_matlab_references.m
% -------------------------------------------------------------------------
% Produces the MATLAB reference files used by the Python test suite.
% Run ONCE in MATLAB, with the ORIGINAL MATLAB model folder as the current
% folder (the folder in which Devel_SNOW_DELAY_FULL_MOULINS_AND_LAKES_MODEL_updated.m,
% routeflats.m, flowdir.m, fillsinks.m, flowaccwithlakes.m, flowdistance.m,
% ixneighbors.m and the input files live, i.e. where you made the standard run).
%
%   >> run('<path to python repo>/tests/matlab/make_matlab_references.m')
%
% Nothing in the MATLAB model folder is modified. A TEMPORARY copy of
% routeflats.m with  randperm(8) -> 1:8  is written to tempdir and put first
% on the path while this script runs; it is removed again at the end.
%
% Outputs (written to OUT_DIR, set below):
%   1) workspace_routing_model_matlab_fixedorder.mat
%        full model workspace, default parameters, FIXED neighbour order
%        -> copy into  <python repo>/MATLAB_reference_output/
%   2) helpers_matlab.mat
%        small synthetic test cases for the helper functions (fixed order)
%        -> copy into  <python repo>/tests/fixtures/
%   3) (optional, RUN_SEEDS = true) workspace_routing_model_matlab_seed<s>.mat
%        standard (random) routing with rng(s), for the seed-spread study
% -------------------------------------------------------------------------

OUT_DIR   = fullfile(pwd, 'matlab_references_for_python');
RUN_SEEDS = false;          % set true for the optional seed study (Level 4)
SEEDS     = [1 2 3];
MODEL     = 'Devel_SNOW_DELAY_FULL_MOULINS_AND_LAKES_MODEL_updated';

if ~exist(OUT_DIR, 'dir'); mkdir(OUT_DIR); end
fprintf('MATLAB version: %s\n', version);

% ---- 1. temporary fixed-order copy of routeflats.m ----------------------
src = which('routeflats');
assert(~isempty(src), 'routeflats.m not found on the path');
txt = fileread(src);
n = numel(strfind(txt, 'randperm(8)'));
assert(n == 1, ['expected exactly one "randperm(8)" in routeflats.m, found %d. ' ...
                'Please check the file and adapt this script.'], n);
fixdir = fullfile(tempdir, 'routeflats_fixedorder');
if ~exist(fixdir, 'dir'); mkdir(fixdir); end
fid = fopen(fullfile(fixdir, 'routeflats.m'), 'w');
fwrite(fid, strrep(txt, 'randperm(8)', '1:8'));
fclose(fid);
addpath(fixdir, '-begin');
clear routeflats; rehash;
assert(strcmp(fileparts(which('routeflats')), fixdir), 'fixed-order routeflats not first on path');
cleanupObj = onCleanup(@() local_cleanup(fixdir));

% ---- 2. full model run with fixed order ---------------------------------
local_run_model(MODEL, fullfile(OUT_DIR, 'workspace_routing_model_matlab_fixedorder.mat'), []);

% ---- 3. helper fixtures (fixed order) -----------------------------------
local_make_helper_fixtures(fullfile(OUT_DIR, 'helpers_matlab.mat'));

% ---- 4. optional seed runs with the ORIGINAL (random) routeflats --------
if RUN_SEEDS
    rmpath(fixdir); clear routeflats; rehash;
    for s = SEEDS
        local_run_model(MODEL, fullfile(OUT_DIR, sprintf('workspace_routing_model_matlab_seed%d.mat', s)), s);
    end
end

fprintf('\nDone. Copy the files from\n  %s\ninto the Python repo as described at the top of this script.\n', OUT_DIR);


% =========================================================================
function local_run_model(model, outfile, seed)
    % Runs the model script in THIS function's workspace (so no variables
    % from the caller - e.g. Kic - leak in) and saves that workspace.
    setappdata(0, 'ref_outfile', outfile);
    setappdata(0, 'ref_fixed', isempty(seed));
    if ~isempty(seed); rng(seed); end
    run(model);
    close all;
    ROUTEFLATS_FIXED_ORDER_MATLAB = getappdata(0, 'ref_fixed'); %#ok<NASGU>
    MATLAB_VERSION_USED = version;                               %#ok<NASGU>
    if ~getappdata(0, 'ref_fixed'); clear ROUTEFLATS_FIXED_ORDER_MATLAB; end
    save(getappdata(0, 'ref_outfile'));
    fprintf('saved %s\n', getappdata(0, 'ref_outfile'));
end

function local_make_helper_fixtures(outfile)
    S = struct();
    cases = local_cases();
    for k = 1:numel(cases)
        c = cases{k};
        p = sprintf('case%d_', k);
        [ic, icd] = ixneighbors(c.dem);
        filled = fillsinks(c.dem);
        Ms = flowdir(c.X, c.Y, filled, 'type', 'single');
        [Xu, Yu] = meshgrid(1:size(c.dem, 2), 1:size(c.dem, 1));
        D = flowdistance(Ms, Xu, Yu);
        [A, LVactual] = flowaccwithlakes(Ms, c.W0, c.RR, c.LV, zeros(size(c.dem)));
        S.([p 'name'])     = c.name;
        S.([p 'dem'])      = c.dem;
        S.([p 'X'])        = c.X;
        S.([p 'Y'])        = c.Y;
        S.([p 'W0'])       = c.W0;
        S.([p 'RR'])       = c.RR;
        S.([p 'LV'])       = c.LV;
        S.([p 'ic'])       = ic;
        S.([p 'icd'])      = icd;
        S.([p 'filled'])   = filled;
        S.([p 'Ms'])       = Ms;
        S.([p 'D'])        = D;
        S.([p 'A'])        = A;
        S.([p 'LVactual']) = LVactual;
    end
    save(outfile, '-struct', 'S');
    fprintf('saved %s (%d cases)\n', outfile, numel(cases));
end

function cases = local_cases()
    rng(42);
    cases = {};
    % 1: tilted plane with a pit
    [X, Y] = meshgrid(0:6, 6:-1:0);
    dem = 10 + X + 0.1*Y;  dem(4,4) = dem(4,4) - 5;
    cases{end+1} = local_case('pit', X, Y, dem);
    % 2: plane with a 3x3 flat plateau
    [X, Y] = meshgrid(0:8, 8:-1:0);
    dem = 10 + X + 0.01*Y;  dem(4:6,4:6) = dem(5,5);
    cases{end+1} = local_case('flat', X, Y, dem);
    % 3: off-ice (NaN) margin on the west side
    [X, Y] = meshgrid(0:7, 7:-1:0);
    dem = 10 + X + 0.05*Y;  dem(:,1:2) = NaN;
    cases{end+1} = local_case('nan_margin', X, Y, dem);
    % 4: enclosed NaN hole
    [X, Y] = meshgrid(0:6, 6:-1:0);
    dem = 10 + X;  dem(4,4) = NaN;
    cases{end+1} = local_case('enclosed_nan', X, Y, dem);
    % 5: rough integer surface -> many ties, flats and pits (stress test)
    [X, Y] = meshgrid(0:24, 19:-1:0);
    dem = round(3*rand(20,25)) + 0.2*X;  dem(:,1:3) = NaN;  dem(8:10, 12:15) = 7;
    cases{end+1} = local_case('rough', X, Y, dem);
    % 6: non-square cells of 500 m (as in the model grid)
    [X, Y] = meshgrid(500*(0:9), 500*(11:-1:0));
    dem = 1000 + 0.01*X + 0.002*Y + 5*rand(12,10);
    cases{end+1} = local_case('cells500m', X, Y, dem);
end

function c = local_case(name, X, Y, dem)
    c.name = name; c.X = X; c.Y = Y; c.dem = dem;
    c.W0 = rand(size(dem)) * 10;
    c.W0(isnan(dem)) = 0;
    c.RR = rand(size(dem)) > 0.1;
    c.LV = zeros(size(dem));
    c.LV(rand(size(dem)) > 0.9) = 5;
end

function local_cleanup(fixdir)
    if any(strcmp(strsplit(path, pathsep), fixdir)); rmpath(fixdir); end
    clear routeflats; rehash;
end
