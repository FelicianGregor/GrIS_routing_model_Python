"""
rasterread.py
-------------
Python translation of TopoToolbox (v1) rasterread.m
(W. Schwanghart, 15 March 2009): read/import an ESRI ASCII grid.

    dem, X, Y, header = rasterread(file)

dem      : (nrows, ncols) array, NODATA -> NaN
X, Y     : meshgrid coordinate matrices of CELL CENTRES,
           Y decreasing down the rows (north at the top, row 0)
header   : dict with ncols, nrows, xllcorner, yllcorner, cellsize, nodata

MATLAB-specific behaviour reproduced / flagged:
  * Header: exactly the first 6 lines are read. Each line is split at the
    first SPACE (strtok with ' ' only, so TAB-separated headers fail in
    MATLAB too) and matched case-insensitively on the first 4 characters
    to ncols / nrows / xllcorner / yllcorner / cellsize / nodata.
    -> 'xllcenter' would also match 'xllcorner' (first 4 chars 'xllc') and
       be treated as a corner coordinate, exactly as in MATLAB.
  * Data: fscanf(fid,'%lg',[ncols nrows]) fills column-major, then the
    transpose gives a row-major grid = file order. Here: reshape in C order.
  * NODATA test is exact equality (dem == nodata), as in MATLAB.
  * Coordinates: MATLAB colon operator. UNCERTAINTY (flagged): MATLAB's
    colon computes elements symmetrically from both ends; here a + k*d is
    used. With integer-valued header entries (typical) the results are
    bit-identical; otherwise they could differ in the last bit.
"""
import numpy as np


_NAMES = ['ncols', 'nrows', 'xllcorner', 'yllcorner', 'cellsize', 'nodata']


def _str2double(s):
    """MATLAB str2double: NaN on failure, surrounding whitespace allowed."""
    try:
        return float(s.strip())
    except (ValueError, AttributeError):
        return np.nan


def rasterread(file):
    if not isinstance(file, str):
        raise TypeError('file is a string indicating the ASCII file name and position')

    header = dict.fromkeys(_NAMES)
    with open(file, 'r') as fid:
        for _ in range(len(_NAMES)):
            headertext = fid.readline().rstrip('\r\n')        # fgetl
            # strtok(headertext,' '): skip leading delimiters, token to next ' '
            stripped = headertext.lstrip(' ')
            k = stripped.find(' ')
            token = stripped if k < 0 else stripped[:k]
            headernum = '' if k < 0 else stripped[k:]
            if len(token) < 4:
                # MATLAB would error on x(1:4) index exceeding dimensions
                raise ValueError('rasterread: malformed header line: %r' % headertext)
            matches = [n for n in _NAMES if n[:4].lower() == token[:4].lower()]
            if not matches:
                raise ValueError('rasterread: unknown header entry: %r' % headertext)
            header[matches[0]] = _str2double(headernum)
        rest = fid.read()

    ncols = int(header['ncols'])
    nrows = int(header['nrows'])
    n = ncols * nrows
    tokens = rest.split()
    if len(tokens) < n:
        # MATLAB fscanf would return a zero-padded / truncated array here;
        # that is almost certainly a corrupt file, so stop instead.
        raise ValueError('rasterread: file contains %d values, expected %d'
                         % (len(tokens), n))
    values = np.array([float(v) for v in tokens[:n]], dtype=float)

    dem = values.reshape((nrows, ncols))                    # == fscanf(...,[ncols nrows])'
    dem[dem == header['nodata']] = np.nan

    cs = header['cellsize']
    # y = (yll+0.5cs)+((nrows-1)cs) : -cs : yll+0.5cs   (top to bottom)
    y0 = (header['yllcorner'] + (0.5 * cs)) + ((header['nrows'] - 1) * cs)
    y = y0 + np.arange(nrows) * (-cs)
    # x = xll+0.5cs : cs : (xll+0.5cs)+((ncols-1)cs)
    x0 = header['xllcorner'] + (0.5 * cs)
    x = x0 + np.arange(ncols) * cs

    X, Y = np.meshgrid(x, y)                                # MATLAB meshgrid(x,y)
    return dem, X, Y, header
