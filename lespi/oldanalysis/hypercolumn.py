from numpy.fft.fftpack import fft2
from numpy.fft.helper import fftshift
import cmath, math
import colorsys
from PIL import Image
import topo
import numpy as np
from itertools import groupby

def extract_channels(image, asHSV=False, astype='double'):
    """
    The astype parameter can be 'float', 'double' but PNG is natively
    defined as unit8. This function uses PIL and numpy but this functionality
    can also alternatively be implemented with pylab.imread and
    matplotlib.colors.rgb_to_hsv. Hue ranges from 0 to 2pi.
    """
    int_types = ['uint8', 'uint16', 'uint32', 'uint64',
                 'int8', 'int16', 'int32', 'int64']
    if not isinstance(image, np.ndarray):
        print "Convert to RGBA"
        imageRGBA = image.convert('RGBA')
        image = np.asarray(imageRGBA).astype(astype)
    data = np.rollaxis(image, axis=-1)
    r, g, b, a = data
    if not asHSV:                 return r, g, b, a
    elif astype not in int_types: return rgb_to_hsv(r, g, b)+(a,)
    else:                         raise Exception('Can only return HSV if floating point type') 

def OR_power_spectrum(image, toarray=True, peak_val=1.0):
    """ Taken from current topographica code. Applies FFT power
        spectrum to hue channel of OR maps (ie orientation). Accepts
        RGB images or arrays."""

    # Duplicates the line of code in command/pylabplot.py and tkgui.
    # Unfortunately, there is no sensible way to reuse the existing code
    if not toarray: peak_val=255
    if not image.shape[0] % 2:
        hue = image[:-1,:-1]
    else:
        hue = image
    fft_spectrum = abs(fftshift(fft2(hue-0.5, s=None, axes=(-2,-1))))
    fft_spectrum = 1 - fft_spectrum # Inverted spectrum by convention
    zero_min_spectrum = fft_spectrum - fft_spectrum.min()
    spectrum_range = fft_spectrum.max() - fft_spectrum.min()
    normalized_spectrum = (peak_val * zero_min_spectrum) / spectrum_range

    if not toarray:
        return Image.fromarray(normalized_spectrum.astype('uint8'), mode='L')
    return normalized_spectrum

def OR_analysis(spectrum, roi=None):
    spectrum = 1 - spectrum
    values, bins = wavenumber_spectrum(spectrum)
    power1D = np.array([np.mean(vals) for vals in values])
    kmax =  bins[power1D.argmax()] - 1
    units_per_hc = units_per_hypercolumn(spectrum.shape,  kmax)
    return units_per_hc,power1D,kmax

def wavenumber_spectrum(spectrum):
    """ Bins the power values in the 2D FFT power spectrum as a function of
    wavenumber. Requires square FFT spectra to make sense. If the input OR map is not
    square, it may be possible to use PIL to resample and resize the FFT so that it
    is square."""

    dim, _dim = spectrum.shape
    assert dim == _dim, "This approach only supports square FFT spectra"
    assert dim % 2,     "Odd dimensions necessary for properly centered FFT plot"

    pixel_bins = range(0, (dim / 2) + 1)
    lower = -(dim / 2); upper = (dim / 2)+1
    x,y = np.mgrid[lower:upper, lower:upper]
    flat_pixel_distances= ((x**2 + y**2)**0.5).flatten()
    flat_spectrum = spectrum.flatten()
    bin_allocation = np.digitize(flat_pixel_distances, pixel_bins)

    spectrum_bins = zip(bin_allocation, flat_spectrum)
    grouped_bins = groupby(sorted(spectrum_bins), lambda x: x[0])
    hist_values = [([sval for (_,sval) in it], bin) for (bin, it) in grouped_bins]
    return zip(*hist_values)

def units_per_hypercolumn(shape,  kmax):
    ''' In the 2D polar FFT, each pixel position corresponds to wavenumber from the
    origin and represents the power in that direction. The center is the DC
    component, the first pixel offset is one wave period, two is two wave periods
    etc. The hypercolumn distance (in units) is therefore estimated from kmax by
    dividing the map size in units by kmax. Eg. a map 20 units wide with kmax of 4
    has a hypercolumn distance of 5, so you can expect 5 units between blobs.'''
    assert shape[0]==shape[1], "Can only compute HC distance for square OR maps"
    if kmax == 0:
        return 0
    else:
        return shape[0] / kmax
