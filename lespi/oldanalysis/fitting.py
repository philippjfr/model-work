from scipy import special as ss
import topo, math, param
import numpy as np
from functools import partial

def idog_conv(sc):
    return math.sqrt(sc*2)

def fr2sp(fr):
    """
    """
    return (math.sqrt(2)/(2*math.pi*fr))

##########################################################
###                 Modelling Functions                ###
##########################################################

# DoG Spatial Frequency Tuning Function

def lgn_dogmodel(f,R_0,K_c,K_s,a,b,C=1.0):
    """
    DoG model response function to sine grating disk stimulus
    with varying spatial frequency (f).
    Ref: Sceniak et al. 2006
    Fitting parameters: R_0 - Steady-state response
                        K_c - Center strength
                        a   - Center spatial constant
                        K_s - Surround Strength
                        b   - Surround spatial constant
    """
    # Fitting penalties for negative coefficients
    if (a <= 0) or (b <= 0) or (K_c <=0) or (K_s <=0) or (R_0 < 0):
        return 10000
    if not isinstance(f,float):
        R = np.zeros(len(f))
        for i,fr in enumerate(f):
            R_c = C * K_c*(1.0-np.exp(-(fr/2.0*a)**2.0))
            R_s = C * K_s*(1.0-np.exp(-(fr/2.0*b)**2.0))
            R[i] = R_0 + R_c - R_s
    else:
        R_c = C * K_c*(1.0-np.exp(-(f/2.0*a)**2.0))
        R_s = C * K_s*(1.0-np.exp(-(f/2.0*b)**2.0))
        R = R_0 + R_c - R_s

    return R


def v1_normmodel(d,beta,K_c,K_s,a,b,C=1.0):
    """
    Normalization model describing response of V1 neurons
    to sine grating disk stimuli of varying sizes.
    Ref: Sceniak et al. 2001
    Fitting parameters: K_c - Center strength
                        a   - Center spatial constant
                        K_s - Surround Strength
                        b   - Surround spatial constant
    """
    r = d/2.0
    if (a <= 0) or (b <= 0) or (K_c <=0) or (K_s <=0):   return 10000
    L_c = 0.5 * a * math.sqrt(math.pi) * ss.erf(2*r/a)
    L_s = 0.5 * b * math.sqrt(math.pi) * ss.erf(2*r/b)
    R = ((C*K_c*L_c)/(1+C*K_s*L_s))**beta
    return R


def v1_idogmodel(d,R_0,K_c,K_s,a,b):
    """
    Basic integrated difference of Gaussian response function
    for area summation curves.
    Ref: DeAngelis et al. 1994
    Fitting parameters: K_c - Center strength
                        a   - Center spatial constant
                        K_s - Surround Strength
                        b   - Surround spatial constant
                        R_0 - Steady-state response
    """

    r = d/2.0
    if (a <= 0) or (b <= 0) or (K_c <=0) or (K_s <=0) or (R_0 < 0): return 10000
    if (idog_conv(a) > 2) or (idog_conv(b) > 2):                    return 10000
    R_c = 0.5 * a * math.sqrt(math.pi) * ss.erf(2*r/a)
    R_s = 0.5 * b * math.sqrt(math.pi) * ss.erf(2*r/b)

    return R_0 + K_c*R_c + K_s*R_s

def lgn_idogmodel(d,R_0,K_c,K_s,a,b):
    """
    iDoG model response function to sine grating disk stimulus
    with optimal spatial frequency and varying disk radius (r).
    Ref: Sceniak et al. 2006
    Fitting parameters: R_0 - Steady-state response
                        K_c - Center strength
                        a   - Center spatial constant
                        K_s - Surround Strength
                        b   - Surround spatial constant
    """
    r = d/2.0
    if (K_c <= 0) or (K_s <= 0) or (a <= 0) or (b <= 0):
        return 10000
    if (idog_conv(a) > 2) or (idog_conv(b) > 2):
        return 10000
    if (K_c > 500) or (K_s > 100):
        return 10000
    R_e = K_c * (a/2 - ((a/2) * np.exp(-(r**2/a))))
    R_i = K_s * (b/2 - ((b/2) * np.exp(-(r**2/b))))
    return R_0 + R_e - R_i
