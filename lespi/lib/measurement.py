"""
Measurement

Analysis functions that measure orientation maps and analyse them
within each running Topographica simulation. These functions return
dictionaries which allow the data to be collated by RunBatchCommand in
the Topographica Lancet extension (topo/misc/lancext.py).
"""

from topo.analysis.featureresponses import MeasureResponseCommand

import numpy as np
import math

import topo
from topo.transferfn.misc import PatternCombine
import analysis

# Set up appropriate defaults for analysis
import topo.analysis.featureresponses
import topo.command.pylabplot


def activity_plotgroup():
    from topo.command.analysis import save_plotgroup
    save_plotgroup("Activity")

def OR_measurement(frequencies=[1.6], num_phase=8, num_orientation=16, outputs=['V1Exc']):
    """ Default scale is 0.3, not 1.0 (100% contrast)"""
    results = {}
    measurement = topo.command.analysis.measure_sine_pref(frequencies=frequencies,
                                                          num_phase=num_phase,
                                                          num_orientation=num_orientation,
                                                          scale=1.0,
                                                          outputs=outputs)
    for sheet in outputs:
        results[sheet+'_OrientationPreference'] = measurement[sheet]['OrientationPreference'].top
        results[sheet+'_OrientationSelectivity'] = measurement[sheet]['OrientationSelectivity'].top

    return results

def ROI(disable=False):
    """
    In the paper, the cortical density is 98 with area 1.5, resulting
    in a 147x147 matrix. A central area cannot have exactly 1.0x1.0
    sheet coordinates - the roi slice returned is slightly larger.
    """
    if disable: return slice(None,None)
    # sheet2matrixidx returns off-center slice with even dimensions.
    (start_idx, stop_idx) = topo.sim.V1Exc.sheet2matrixidx(-1.0,1.0)
    return slice(start_idx, stop_idx+1) # Centered with odd dimensions.

def pinwheel_analysis():
    """
    Computes the pinwheels analysis. Fits the hypercolumn distance to
    get the estimated value of kmax to compute the pinwheel density.
    """
    #roi = ROI() # Central ROI (1.0 x 1.0 in sheet coordinates)
    cortex_density=48
    preference = topo.sim['VSDLayer'].views.maps.OrientationPreference.top.data
    x,y = preference.shape
    preference = preference if x%2 else preference[:-1,:-1]
    polar_map = analysis.polar_preference(preference)
    contour_info = analysis.polarmap_contours(polar_map)
    (re_contours, im_contours, _ ) = contour_info
    pinwheels = analysis.identify_pinwheels(*contour_info)
    metadata = analysis.hypercolumn_distance(preference,cortex_density)
    metadata.update(rho=(len(pinwheels) / metadata['kmax'] ** 2))
    return {'pinwheels': pinwheels,
            're_contours': re_contours,
            'im_contours': im_contours,
            'metadata': metadata}

def save_projection(outputs=['V1Exc']):
    topo.command.analysis.update_projection()
    cfs = dict((sheet,topo.sim[sheet].views.cfs) for sheet in outputs)
    return cfs


def measure_position(outputs=['V1Exc']):
    try:
        results = {}
        measurement = topo.command.pylabplot.measure_position_pref(divisions=20,
                                                                   x_range=(-0.5, 0.5),
                                                                   y_range=(-0.5, 0.5),
                                                                   size=0.1, scale=1.0)

        for sheet in outputs:        
            results[sheet+'_XPreference'] = measurement[sheet]['XPreference']
            results[sheet+'_YPreference'] = measurement[sheet]['YPreference']

        return results
    except:
        pass

def measure_or_tuning(times=[0,100,5000,10000], outputs=['V1Exc'], frequencies=[1.6], num_phase=8, num_orientation=20):
    if topo.sim.time() == times[-1]:
        try:
            contrasts = [{"contrast": 5}, {"contrast": 10},
                         {"contrast": 30}, {"contrast": 70}]
            topo.command.pylabplot.measure_or_tuning_fullfield(outputs=outputs,
                                                               frequencies=frequencies,
                                                               num_orientation=num_orientation,
                                                               num_phase=num_phase,
                                                               curve_parameters=contrasts)
            return dict(('or_tuning_{sheet}'.format(sheet=s),
                         topo.sim[s].views.curves.Orientation.top) for s in outputs
                         if topo.sim[s].views.curves.has_key('Orientation'))
        except:
            pass

def measure_size(times=[0,100,5000,10000], num_phase=8):
    if topo.sim.time() == times[-1]:
        try:
            results = {}
            measurement = topo.command.pylabplot.measure_size_response(num_sizes=16, max_size=3.0,
                                                                       num_phase=num_phase,
                                                                       outputs=['V1Exc'])
            return {'Size':measurement}
        except:
            pass

def measure_octc(times=[0,100,5000,10000], num_phase=8, num_orientation=16):
    if topo.sim.time() == times[-1]:
        try:
            results = {}
            contrasts = [{"contrastsurround": 5},{"contrastsurround": 10},
                         {"contrastsurround": 30},{"contrastsurround": 70}]
            measurement = topo.command.pylabplot.measure_orientation_contrast(outputs=['V1Exc'],
                                                                              num_phase=num_phase,
                                                                              num_orientation=num_orientation,
                                                                              contrast_center=70,
                                                                              sizecenter=0.6,
                                                                              sizesurround=1.2,
                                                                              thickness=0.6,
                                                                              curve_parameters=contrasts)
            return {'OCTC':measurement}
        except:
            pass

def measure_rfs(times=[0,100,5000,10000]):
    if topo.sim.time() == times[-1]:
        try:
            results = {}
            measurement = topo.command.analysis.measure_rfs(outputs=['V1Exc'],inputs=['Retina'],
                                                            presentations=500)
            results['RFs'] = measurement['V1Exc']['Retina']
            return results
        except:
            pass


def clear_views(outputs):
    for output in outputs:
        try:
            del topo.sim[output].views.maps['Activity']
        except:
            pass
        
