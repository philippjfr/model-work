"""
This module provides the code to identify pinwheel locations in an OR map, find the
hypercolumn distance and compute the pinwheel density. The approach uses the fact
that pinwheels are located at the intersections between the contours of the imaginary
and complex components in a polar orientation map."""

import cmath, math
import numpy as np
import matplotlib.pyplot as plt
import utils.image as image
from itertools import groupby

try:     from scipy.optimize import curve_fit
except:  curve_fit = None


#======================#
# Image transformation #
#======================#

def normalize_polar_channel(polar_channel):
    '''This functions normalizes an OR map (polar_channel) taking into account the
    region of interest (ROI). The ROI is specified by values set to 99. Note that
    this functionality is implemented to reproduce the experimental approach and has
    not been tested (not required for Topographica simulations)'''

    def grad(r):
        (r_x,r_y) = np.gradient(r)
        (r_xx,r_xy)=np.gradient(r_x);
        (r_yx,r_yy)=np.gradient(r_y);
        return r_xx**2+ r_yy**2 + 2*r_xy**2

    roi = np.ones(polar_channel.shape) # Set ROI to 0 to ignore values of -99.
    roi[roi == -99] = 0                # In Matlab: roi(find(z==-99))=0

    fst_grad = grad(roi)
    snd_grad = grad(fst_grad)

    snd_grad[snd_grad != 0] = 1  # Find non-zero elements in second grad and sets to unity
    roi[snd_grad == 1] = 0       # These elements now mask out ROI region (set to zero)

    ind = (polar_channel != 99)                    # Find the unmasked coordinates
    normalisation = np.mean(np.abs(polar_channel)) # The complex abs of unmasked
    return polar_channel / normalisation           # Only normalize with unmasked

#=========================#
# Pinwheel identification #
#=========================#

def remove_path_duplicates(vertices):
    "Removes successive duplicates along a path of vertices "
    zero_diff_bools = np.all(np.diff(vertices, axis=0) == 0, axis=1)
    duplicate_indices, = np.nonzero(zero_diff_bools)
    return np.delete(vertices, duplicate_indices, axis=0)

def polarmap_contours(polarmap):  # Example docstring
   """
   Identifies the real and imaginary contours in a polar map.  Returns the real and
   imaginary contours as 2D vertex arrays together with the pairs of contours known
   to intersect. The coordinate system is normalized so x and y coordinates range
   from zero to one.

   Contour plotting requires origin='upper' for consistency with image coordinate
   system.
   """

   # Convert to polar and normalise
   normalized_polar = normalize_polar_channel(polarmap)
   figure_handle = plt.figure()
   # Real component
   re_contours_plot = plt.contour(normalized_polar.real, 0, origin='upper')
   re_path_collections = re_contours_plot.collections[0]
   re_contour_paths = re_path_collections.get_paths()
   # Imaginary component
   im_contours_plot = plt.contour(normalized_polar.imag, 0, origin='upper')
   im_path_collections = im_contours_plot.collections[0]
   im_contour_paths = im_path_collections.get_paths()
   plt.close(figure_handle)

   intersections = [ (re_ind, im_ind)
                     for (re_ind, re_path) in enumerate(re_contour_paths)
                     for (im_ind, im_path) in enumerate(im_contour_paths)
                    if im_path.intersects_path(re_path)]

   (ydim, xdim) = polarmap.shape
   # Contour vertices  0.5 pixel inset. Eg. (0,0)-(48,48)=>(0.5, 0.5)-(47.5, 47.5)
   # Returned values will not therefore reach limits of 0.0 and 1.0
   re_contours =  [remove_path_duplicates(re_path.vertices) / [ydim, xdim] \
                       for re_path in re_contour_paths]
   im_contours =  [remove_path_duplicates(im_path.vertices) / [ydim, xdim] \
                       for im_path in im_contour_paths]
   return (re_contours,  im_contours, intersections)

def find_intersections(contour1, contour2):
    '''
    Vectorized code to find intersections between contours. All successive
    duplicate vertices along the input contours must be removed to avoid
    division-by-zero errors. Zero division errors for vertical/horizontal
    lines can be ignored with np.seterr(divide='warn', invalid='warn').
    '''
    amin = lambda x1, x2: np.where(x1<x2, x1, x2)       # Elementwise min selection
    amax = lambda x1, x2: np.where(x1>x2, x1, x2)       # Elementwise max selection
    aall = lambda abools: np.dstack(abools).all(axis=2) # dstacks, checks True depthwise
    # Uses delta (using np.diff) to find successive slopes along path
    slope = lambda line: (lambda d: d[:,1]/d[:,0])(np.diff(line, axis=0))
    # Meshgrids between both paths (x and y). One element sliced off end/beginning
    x11, x21 = np.meshgrid(contour1[:-1, 0], contour2[:-1, 0])
    x12, x22 = np.meshgrid(contour1[1:, 0], contour2[1:, 0])
    y11, y21 = np.meshgrid(contour1[:-1, 1], contour2[:-1, 1])
    y12, y22 = np.meshgrid(contour1[1:, 1], contour2[1:, 1])
    # Meshgrid of all slopes for both paths
    m1, m2 = np.meshgrid(slope(contour1), slope(contour2))
    m2inv = 1/m2 # m1inv was not used.
    yi = (m1*(x21-x11-m2inv*y21) + y11)/(1 - m1*m2inv)
    xi = (yi - y21)*m2inv + x21 # (xi, yi) is intersection candidate
    # Bounding box type conditions for intersection candidates
    xconds = (amin(x11, x12) < xi, xi <= amax(x11, x12),
              amin(x21, x22) < xi, xi <= amax(x21, x22) )
    yconds = (amin(y11, y12) < yi, yi <= amax(y11, y12),
              amin(y21, y22) < yi, yi <= amax(y21, y22) )
    return xi[aall(xconds)], yi[aall(yconds)]


def identify_pinwheels(re_contours,  im_contours, intersections):
    '''
    Locates the pinwheels from the intersection of the real and imaginary
    contours of of polar OR map.
    '''
    pinwheels = []
    for (re_ind, im_ind) in intersections:
        re_contour = re_contours[re_ind]
        im_contour = im_contours[im_ind]
        x, y = find_intersections(re_contour, im_contour)
        pinwheels += zip(x,y)
    return pinwheels

#===========================================#
# Hypercolumn distance and pinwheel density #
#===========================================#

def wavenumber_spectrum(spectrum, average_fn = np.mean):
    '''
    Bins the power values in the 2D FFT power spectrum as a function of
    wavenumber. Requires square FFT spectra (odd dimension) to work. If the input OR
    map is not square, it may be possible to use PIL to resample and resize the FFT
    so that it is square.
    '''
    dim, _dim = spectrum.shape
    assert dim == _dim, "This approach only supports square FFT spectra"
    assert dim % 2,     "Odd dimensions necessary for properly centered FFT plot"
    # OR_power_spectrum returns black (low values) for high powers - needs inverting
    spectrum = 1 - spectrum
    pixel_bins = range(0, (dim / 2) + 1)
    lower = -(dim / 2); upper = (dim / 2)+1
    # Grid of coordinates relative to central DC component (0,0)
    x,y = np.mgrid[lower:upper, lower:upper]
    flat_pixel_distances= ((x**2 + y**2)**0.5).flatten()
    flat_spectrum = spectrum.flatten()
    # Indices in pixel_bins to which the distances belong (points to max boundary)
    bin_allocation = np.digitize(flat_pixel_distances, pixel_bins)
    # The bin allocation zipped with actual fft power values
    spectrum_bins = zip(bin_allocation, flat_spectrum)
    grouped_bins = groupby(sorted(spectrum_bins), lambda x: x[0])
    hist_values = [([sval for (_,sval) in it], bin) for (bin, it) in grouped_bins]
    (power_values, bin_boundaries) = zip(*hist_values)
    averaged_powers = [average_fn(power) for power in power_values]
    assert len(bin_boundaries) == len(pixel_bins)
    return averaged_powers

def units_per_hypercolumn(shape,  kmax):
    ''' In the 2D polar FFT, each pixel position corresponds to wavenumber from the
    origin and represents the power in that direction. The center is the DC
    component, the first pixel offset is one wave period, two is two wave periods
    etc. The hypercolumn distance (in units) is therefore estimated from kmax by
    dividing the map size in units by kmax. Eg. a map 20 units wide with kmax of 4
    has a hypercolumn distance of 5, so you can expect 5 units between blobs.'''
    assert shape[0]==shape[1], "Can only compute HC distance for square OR maps"
    if kmax == 0: return shape[0] # DC component only, assume whole map is a hypercolumn.
    return shape[0] / float(kmax)


def pinwheel_density(roi, units_per_hc, pinwheel_count):
    ''' Computes the pinwheel density knowing the units per hypercolumn, the number
    of pinwheels and the area of the roi from which the pinwheels were extracted.
    The roi parameter can be a tuple specifying the rectangular width and height
    of the roi or the numpy array of the roi itself.'''

    if isinstance(roi, np.ndarray): (ydim, xdim) = roi.shape
    else:                           (ydim, xdim) = roi

    pinwheel_area = xdim*ydim
    hc_area = units_per_hc**2
    area_factor = float(pinwheel_area) / hc_area
    return pinwheel_count / area_factor

def pinwheel_selectivites(pinwheels, selectivity):
    """
    Compresses sheet2matrixidx, sheet2matrix and boundingbox code.
    Vectorizing with numpy did not cause significant speedup (600+
    pinwheels).
    """
    (row_density, col_density) = selectivity.shape
    maxcol = col_density -1; maxrow = row_density -1
    (left, bottom, right, top) = (0.0, 0.0, 1.0, 1.0)
    pinwheel_selectivities = []
    for (x,y) in pinwheels:
        col_val = (x-left) * col_density
        row_val = (top-y)  * row_density
        # Discretize and clip
        row_ind =int(np.clip(np.floor(row_val),0.0, maxrow))
        col_ind = int(np.clip(np.floor(col_val), 0.0, maxcol))
        pinwheel_selectivities.append(selectivity[row_ind][col_ind])
    return pinwheel_selectivities

def pinwheel_selectivites2(pinwheels, selectivity):
    # Vectorize the above. Index in one go and not in a loop.
    pass


def KaschubeFit(k, a0=0.35, a1=3.8, a2=1.3, a3=0.15, a4=-0.003, a5=0):
    """
    Fitting function used by Kaschube for finding the hypercolumn
    distance from the Fourier power spectrum. Default values
    correspond to a good starting point for GCAL maps. These values
    should match the init_fit defaults of pinwheel_analysis below.

    a0 => Gaussian height
    a1 => Peak position
    a2 => Gaussian spread (ie. variance)
    a3 => Baseline value (w/o falloff)
    a4 => Linear falloff
    a5 => Quadratic falloff
    """
    if (a0 <= 0) or (a1 <= 0) or (a2 <=0) or (a3 <=0):  return 10000 # Penalise bad values.
    exponent = - ((k - a1)**2) / (2 * a2**2)
    return a0 * np.exp(exponent) + a3 + a4*k + a5*np.power(k,2)


def pinwheel_analysis(preference, pinwheel_count, roi=None, zero_DC = False, ignore_DC=False,
                      fit_kmax=True):
    '''
    Pinwheel analysis that returns a dictionary containing the
    analysed values. Optional curve fitting (using the same function
    as detailed in Kaschube's paper - supplementary materials)
    '''
    fft_spectrum = image.OR_power_spectrum(preference)
    wavenumber_power = wavenumber_spectrum(fft_spectrum)

    kmax_argmax =  float(np.argmax(wavenumber_power[1:])+1)
    kmax_delta = 0.0 # If fitting, returns the difference with simple argmax

    if fit_kmax and (curve_fit is None):
        raise Exception('Requires curve_fit from scipy.optimize.')

    if fit_kmax:
        try:
            if zero_DC:
                wavenumber_power[0] = 0
            elif ignore_DC:
                wavenumber_power = wavenumber_power[1:]
                #ks += 1
            ks = np.array(range(len(wavenumber_power)))
            baseline = np.mean(wavenumber_power)
            height = wavenumber_power[int(kmax_argmax)] - baseline
            init_fit = [height, kmax_argmax, 4.0, baseline, 0, 0]
            fit,_ = curve_fit(KaschubeFit, ks, np.array(wavenumber_power), init_fit, maxfev=10000)
            if ignore_DC:
                fit[1] += 1
            arg_order = ['a0', 'a1', 'a2', 'a3', 'a4', 'a5']
            fit_dict = dict(zip(arg_order, fit))
            fitted_kmax = fit_dict['a1']
            kmax_delta = fitted_kmax - kmax_argmax
        except:
            print 'Fitting failed. Falling back to argmax...'
            fitted_kmax = kmax_argmax
            kmax_delta = None
            fit_dict = None

    kmax = fitted_kmax if fit_kmax else kmax_argmax
    if kmax <=0:
        print 'Negative kmax fit, falling back to argmax'
        kmax = kmax_argmax
        kmax_delta = None

    units_per_hc = units_per_hypercolumn(preference.shape,  kmax)
    if roi is None: roi = fft_spectrum.shape
    pw_density = pinwheel_density(roi, units_per_hc, pinwheel_count)
    return {'pinwheel_count':pinwheel_count,
            'wavenumber_power':wavenumber_power,
            'kmax':kmax,
            'kmax_argmax':kmax_argmax,
            'kmax_delta':kmax_delta,
            'units_per_hc':units_per_hc,
            'fit':(KaschubeFit, fit_dict) if fit_kmax else (None,None),
            'init_fit': init_fit if fit_kmax else None,
            'pw_density':pw_density}


def similarity_index(prefA, prefB, unit_metric=True):
    difference = abs(prefA - prefB)
    greaterHalf = (difference >= 0.5)
    difference[greaterHalf] = 1.0 - difference[greaterHalf] # Ensure difference is symmetric distance.
    similarity = 1 - np.mean(difference * 2.0)              # Difference [0,0.5] so 2x normalizes...
    if unit_metric: similarity = 2*(similarity-0.5)         # Subtracted from 1.0 as low difference => high stability
    return similarity                                       # If unit metric then lowest value is zero for uncorrelated.


def stability_index(preferences, last_map=None, unit_metric=True):
    '''
    Equation (11): Note that normalisation and modulus of pi (1/pi, mod pi) terms
    are implicit as orientation preference already expressed in interval [0,1.0].
    Preferences much be sorted by simulation time for correct results.

    If refmap is None, the final preference listed will be used for comparison.
    '''
    last_map = last_map if last_map is not None else preferences[-1]
    stabilities = []
    for preference in preferences:
        stability = similarity_index(last_map, preference, unit_metric = unit_metric)
        stabilities.append(stability)      # Subtracted from 1.0 as low difference => high stability
    return stabilities
