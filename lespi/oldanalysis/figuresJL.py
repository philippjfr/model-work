import colorsys
import math
import os
import json

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image

import utils.image as image

rgb_to_hsv = np.vectorize(colorsys.rgb_to_hsv)
hsv_to_rgb = np.vectorize(colorsys.hsv_to_rgb)


def overlaid_OR_map(pref, pwdata,style='wo'):
    fig=pinwheel_overlay(pwdata,style=style)
    plt.imshow(pref, cmap=plt.cm.hsv, extent=[0,1,0,1], aspect=1, interpolation='nearest')
    return fig

#===============#
# Metric figure #
#===============#

def HC_area(units_per_hc, map_unit_width, show_axes=False, offset = (0.1, 0.195), edgecolor='w'):

    """
    Overlay that makes a box showing the hypercolumn area.
    """
    HC_dist = float(units_per_hc) / map_unit_width
    fig = plt.figure(frameon=False)
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    (offsetX, offsetY) = offset
    ax.add_patch(plt.Rectangle((offsetX, offsetY),HC_dist, HC_dist,
                               facecolor=(0,0,0,0), edgecolor=edgecolor, lw=5))
    if not show_axes:
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        [spine.set_visible(False) for spine in ax.spines.values()]
    plt.ylim((0,1))
    plt.xlim((0,1))
    plt.close(fig)
    return fig

def density_scatterplot(GCAL_area_pwds, L_area_pwds, Kaschube_JSON, central_hc_area, show_axes=False,
                        lw=3, s=60, half_width=0.16, regression=True, avgfn = np.median):
    """
    Scatterplot of the pinwheel density over all the seeds (GCAL, area=1.5)
    """
    fig = plt.figure(figsize=(8,6))
    ax = plt.subplot(111)
    max_pwds, min_pwds = [],[]

    plt.axhline(y=math.pi, linestyle='dotted')
    d = json.load(open(Kaschube_JSON,'r'))

    fcolor = (0,0,0,0)
    (color_f, color_s, color_g) = ('#27833a','#7fcbf1', '#f9ab27')
    plt.scatter(d['F']['x'], d['F']['y'], marker='D', edgecolor=color_f, facecolor=fcolor, s=s, lw=lw)
    plt.scatter(d['S']['x'], d['S']['y'], marker='D', edgecolor=color_s, facecolor=fcolor, s=s, lw=lw)
    plt.scatter(d['G']['x'], d['G']['y'], marker='D', edgecolor=color_g, facecolor=fcolor, s=s, lw=lw)

    F_medx, F_medy = np.median(d['F']['x']), np.median(d['F']['y'])
    S_medx, S_medy = np.median(d['S']['x']), np.median(d['S']['y'])
    G_medx, G_medy = np.median(d['G']['x']), np.median(d['G']['y'])

    plt.hlines(F_medy, F_medx-half_width, F_medx+half_width,colors=color_f, linestyles='solid', lw=lw)
    plt.hlines(S_medy, S_medx-half_width, S_medx+half_width,colors=color_s, linestyles='solid', lw=lw)
    plt.hlines(G_medy, G_medx-half_width, G_medx+half_width,colors=color_g, linestyles='solid', lw=lw)

    for (area_pwds, color, marker) in [(GCAL_area_pwds,'r','o'), (L_area_pwds,'b','o')]:
        hc_area_ordered = sorted(area_pwds)
        pwds = [pwd for (_, pwd) in hc_area_ordered]
        hc_areas = [hc for (hc, pwd) in hc_area_ordered]

        scaled_hc_areas = np.array([(central_hc_area/ np.mean(hc_areas))*hc_area for hc_area in hc_areas])
        plt.scatter(scaled_hc_areas, np.array(pwds), marker=marker,
                    edgecolor=color, facecolor=(0,0,0,0), s=s,lw=lw)

        plt.hlines(avgfn(pwds), np.median(scaled_hc_areas)-half_width,
                   np.median(scaled_hc_areas)+half_width,
                   colors=color, linestyles='solid', lw=lw)

        if regression:
            from scipy import stats
            slope, intercept, r_value, p_value, std_err = stats.linregress(scaled_hc_areas,pwds)
            samples = np.linspace(0,1.1,100)
            plt.plot(samples, slope*samples + intercept, 'r--')
            print 'R value: %f' % r_value

    if not show_axes:
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        [spine.set_visible(False) for spine in ax.spines.values()]

    plt.ylim((0, 12))#15
    plt.xlim((0, 1.1))
    return fig



def kernel_plot(unit_metric, show_axes=False, lw=4):
    """
    Plots the kernel for squashing the pinwheel density into unit range.
    """
    pwds = np.linspace(0,50,99)
    fig = plt.figure()
    ax = plt.subplot(111)
    ys = [unit_metric(pwd) for pwd in pwds]
    plt.vlines(math.pi,0.0, max(ys), color='k', lw=2,  linestyles='dotted')
    plt.plot(pwds, ys, color='k', lw=lw)
    if not show_axes:
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        [spine.set_visible(False) for spine in ax.spines.values()]

    return fig


def FFT_histogram(pw_analysis, ylimit=0.8,
                  bar_color=(0.0, 0.35,0.0),
                  fit_color='r', marker_color=(0.0,0.0,0.5), argmax_color=None):
   '''
   Intended for generating FFT histograms.
   '''
   values = pw_analysis['wavenumber_power']
   (fit_fn, fit_dict) = pw_analysis['fit']
   bins = range(len(values))
   fig = plt.figure(frameon=False)
   fig.patch.set_alpha(0.0)
   ax = fig.add_subplot(111)
   ax.patch.set_alpha(0.0)
   # Hide the plot spines
   [spine.set_visible(False) for spine in ax.spines.values()]
   ax.get_xaxis().set_visible(False)
   ax.get_yaxis().set_visible(False)

   light_blue =(0.75, 0.75, 1.0)
   bars = plt.bar(bins, values, facecolor='w', edgecolor=bar_color,  width=1.0, lw=2)

   argmax = np.argmax(values[1:])+1
   if argmax_color is not None:
      bars[argmax].set_facecolor(argmax_color)

   if fit_dict is not None:
      xs = np.linspace(1, len(values), 100) # From 0 for DC
      fit_curve = list(fit_fn(x, **fit_dict) for x in xs)
      plt.plot(xs, fit_curve, color=fit_color, lw=8, alpha=0.7)
      kmax = fit_dict['a1']


   if kmax >=1.5:
       plt.arrow(kmax, max(fit_curve)+ 0.38 , 0.0, -0.3, fc=marker_color, ec=marker_color,lw=6,
                 length_includes_head=True, head_width=4.0, head_length=0.1)

   plt.xlim((0.0,len(values)))
   plt.ylim((0.0, ylimit))
   return fig


def pinwheel_overlay(pwdata, contours=True, pinwheels=True, style='wo',linewidth=1):
   '''
   Plots the real and imaginary pinwheel contours and the pinwheel
   locations. Designed to be overlayed over the respective OR
   map. Note that the pwdata argument  is a PwData instance.
   '''
   fig = plt.figure(frameon=False)
   fig.patch.set_alpha(0.0)
   ax = plt.subplot(111, aspect='equal', frameon=True)
   ax.patch.set_alpha(0.0)
   plt.hold(True)
   if contours:
      for recontour in pwdata.recontours:
         plt.plot(recontour[:,0], recontour[:,1],'w',linewidth=linewidth)
      for imcontour in pwdata.imcontours:
         plt.plot(imcontour[:,0], imcontour[:,1],'k', linewidth=linewidth)
   if pinwheels:
      Xs, Ys = zip(*pwdata.pinwheels)
      plt.plot(np.array(Xs), np.array(Ys), style)

   plt.xlim((0.0,1.0));         plt.ylim((0.0,1.0))
   ax.xaxis.set_ticks([]);      ax.yaxis.set_ticks([])
   ax.xaxis.set_ticklabels([]); ax.yaxis.set_ticklabels([])
   return fig

def scale_bar_overlay(pwinfo, map_unit_width, aspect=0.05, color=(1.0,1.0,1.0)):

   units_per_hc = pwinfo['units_per_hc']

   width = float(units_per_hc) / map_unit_width
   fig = plt.figure()
   (r,g,b) = color
   ax = fig.add_subplot(111, aspect=aspect)
   ax.patch.set_alpha(0.0)
   ax.get_xaxis().set_visible(False)
   ax.get_yaxis().set_visible(False)
   [spine.set_visible(False) for spine in ax.spines.values()]
   if (0.5>width>0)and (pwinfo['kmax_delta'] is not None):
       ax.add_patch(plt.Rectangle((0,0), width=width, height=1.0, facecolor=(r,g,b,1.0)))
   return fig


#===================#
# Stability figures #
#===================#




def stream(xs, ys, ystd, samples, color=(1,0,0), linecolor=(1,0,0),  show_std=False):
   """
   If samples is not None, the 0.96*standard error (95% confidence) lines are
   plotted using the given sample count.
   """
   stderr = ystd / (samples**0.5)
   conf95 = 1.95996 * stderr # http://en.wikipedia.org/wiki/1.96 (1.95996 39845 40054 23552)

   plt.fill_between(xs, ys+conf95, ys-conf95, interpolate=True, color=color)
   plt.plot(xs,ys,color=linecolor)

   if show_std:
      plt.plot(xs, ys-ystd, color=(r,g,b), linestyle='--')
      plt.plot(xs, ys+ystd, color=(r,g,b), linestyle='--')


def contrast_streams(selectivity_data, stability_data, metric_data, samples, vlines=[],
                     show_vlines=True, show_hlines=False):
   '''
   The plots showing selectivity and stability against scale (contrast)
   '''
   selX, selY, selErr = zip(*selectivity_data)
   stabX, stabY, stabErr = zip(*stability_data)
   metricX, metricY,metricErr = zip(*metric_data)
   print "Maximum selectivity plotted is: %f" % max(selY)
   fig = plt.figure(frameon=False)
   ax = fig.add_subplot(111, frameon=False)
   stream(np.array(selX), np.array(selY), np.array(selErr), samples, color='#ffcfcf',linecolor=(1,0,0))
   stream(np.array(stabX), np.array(stabY), np.array(stabErr), samples, color='#ccffcc',linecolor=(0,1,0))
   stream(np.array(metricX),np.array(metricY),np.array(metricErr),samples,color='#ccccff',linecolor=(0,0,1))
   [spine.set_visible(False) for spine in ax.spines.values()]
   ax.get_xaxis().set_visible(False)
   ax.get_yaxis().set_visible(False)
   plt.xlim((0.0,max(selX)))
   plt.ylim((-0.1,1.1))
   if show_hlines:
      plt.axhline(y=1, linewidth=4, color='k')
      plt.axhline(y=0, linewidth=4, color='k')
   for vl in vlines:
      if show_vlines: plt.axvline(x=vl, linewidth=1, color='r', linestyle='--')
   return fig


def stability_lines(contrasts, stabilities):
    tmeans = np.array([np.mean(stab, axis=1) for stab in stabilities])
    fig = plt.figure()
    plt.plot(contrasts, tmeans)
    grand_mean =  np.mean(tmeans, axis=1)
    plt.plot(contrasts, grand_mean, 'k--', lw=2)
    stderr = np.std(tmeans, axis=1) / (10**0.5)
    conf95 = 1.95996 * stderr
    plt.fill_between(contrasts, grand_mean+conf95, grand_mean-conf95, interpolate=True, color=(0.0,1.0,0.0,0.2))
    plt.ylim((0,1.0))
    return fig



def SIbarplot(stabilities, selectivities, aspect=7.0, selectivity_norm=1.0, swap=True):

   med_selectivities = [np.median(sel) for sel in selectivities]
   #print "Maximum white selectivity is value %f" % max(med_selectivities)

   # FIXME
   selectivity_norm = 0.946970#0.714500924485
   norm_selectivities = [sel / selectivity_norm for sel in med_selectivities]
   #print max(norm_selectivities),"<NORMED"

   bar_count = len(stabilities)
   fig = plt.figure(frameon=False)
   fig.patch.set_alpha(0.0)
   ax = fig.add_subplot(111, aspect=aspect)
   ax.patch.set_alpha(0.0)
   ax.get_xaxis().set_visible(False)
   ax.get_yaxis().set_visible(False)
   bins = range(bar_count)

   heights = norm_selectivities if swap else stabilities
   greys = stabilities if swap else norm_selectivities

   bars = plt.bar(bins, heights, width=0.8)

   for ind, greyval in enumerate(greys):
      if greyval >= 1.0:
         #print "WARNING: Clipping grey value below 1.0"
         greyval=1.0
      if greyval <= 0.0:
         #print "WARNING: Clipping grey value above 0.0"
         greyval=0.0

      bars[ind].set_facecolor((greyval,greyval,greyval))
   plt.xlim((0.0,bar_count))
   plt.ylim((0.0,1.0))
   return fig


def SI_plot(times, noiseless_SIs, jitter_SIs, additive_SIs):
   SIfig = plt.figure()
   ax = plt.subplot(111)
   plt.plot(times, noiseless_SIs, 'kx', markersize=15)
   plt.plot(times, jitter_SIs, 'ro', markersize=10, lw=0)
   plt.plot(times, additive_SIs, 'ko', markersize=10, lw=0)
   plt.ylim(0.5, 1.0)
   plt.xlim(2000, 22000)
   plt.axvline(6000, ymax=(2 / 5.0), linestyle='--', color='k')
   plt.xticks(times)
   ax.set_xticklabels([])
   ax.set_yticklabels([])
   return SIfig

def TC_curve_plot(normalize):
   matplotlib.rcParams['axes.color_cycle'] = ['r', 'y', 'b', 'm', 'c', 'g', '#ff9500', '#bafc56']

   def normalised_plot(x_values, y_values, label, lw):
      ax = plt.gca()
      fig = plt.gcf()
      plt.title('')
      fig.set_size_inches(8,6 )

      if normalize:
         if max(y_values) == 0: return
         else:
            plt.plot(x_values, [ yval / max(y_values) for yval in y_values], label=label, lw=lw, color='k')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
      else:
         plt.plot(x_values, y_values, label=label, lw=lw)
         plt.xlabel('')
         plt.ylabel('')
         plt.ylim((0.0,0.8))
         ax.get_xaxis().set_visible(False)
         ax.get_yaxis().set_visible(False)
         plt.draw()
   return normalised_plot



#===============#
# Raster images #
#===============#

def CFIm(cfs, coords, coord, shape):
   '''
   Extracts single CF as a PIL image (normalized)
   '''
   assert coord in coords
   normalize = lambda arr: (arr - arr.min()) / (arr.max() - arr.min())
   ind = np.argmax(np.all(coords==coord,1))
   return image.resize(image.greyscale_image(normalize(cfs[:,:,ind])), shape)

def CF_block(cfs, coords, orientation='H', size=26, border=5): # Move to image.py??
   '''
   Returns a PIL image containing a horizontal or vertical block of V1 CFs.
   '''
   assert orientation in ['H', 'V']
   cf_inds = [8, 13,17, 22, 26, 31, 35, 40]
   CF_XY = [(ind, 21) for ind in cf_inds] # Vertical coordinates
   CF_XY = [(y, x) for (x,y) in CF_XY] if (orientation == 'H') else CF_XY

   cf_ims = [CFIm(cfs, coords, [x,y],(size, size)) for (x,y) in CF_XY]
   (bgx, bgy) = (26+(border*2),size*8+(9*border))
   (bgx, bgy) = (bgy, bgx) if (orientation == 'H') else (bgx, bgy)
   cf_block = Image.new('RGBA', (bgx, bgy), (255,255,255))
   paste_coords = [(border, size*i + ((i+1)*border)) for i in range(len(cf_ims))]
   paste_coords = [(y,x) for (x,y) in paste_coords] if (orientation == 'H') else paste_coords
   [cf_block.paste(cf_im, pcoord) for (cf_im, pcoord) in zip(cf_ims, paste_coords)]
   return cf_block


#====================#
# Old Style (unused) #
#====================#


def contrast_graph(scales, median_selectivities, mean_stabilities):
   '''
   The plots showing selectivity and stability against scale (contrast)
   '''
   fig = plt.figure(frameon=False)
   ax = fig.add_subplot(111, frameon=False)
   plt.hold()
   plt.plot(scales, median_selectivities,'k--',  linewidth=3.0)
   plt.hold()
   plt.plot(scales, mean_stabilities,'k', linewidth=3.0)
   [spine.set_visible(False) for spine in ax.spines.values()]
   ax.get_xaxis().set_visible(False)
   ax.get_yaxis().set_visible(False)
   return fig

def HSVBars(stabilities, selectivities, aspect=0.7):
   '''
   21 bars: time=0, then 20 bars sampled every 1000 iterations. Inputs should be
   numpy arrays.
   '''
   med_selectivities = [np.median(sel) for sel in selectivities]
   bar_number = len(med_selectivities)
   stability_hues =  np.abs(np.array(stabilities)-1.0) / 6.0 # Segment of colour wheel
   RGBs = zip(*hsv_to_rgb(stability_hues, med_selectivities, np.ones(bar_number)))

   fig = plt.figure(frameon=False)
   ax = fig.add_subplot(111, aspect=aspect)
   for (ind, (r,g,b)) in enumerate(RGBs):
      ax.add_patch(plt.Rectangle((ind,0), width=0.8, height=10, facecolor=(r,g,b,1.0)))
   ax.xaxis.set_visible(False); ax.yaxis.set_visible(False)
   plt.xlim([0,bar_number]); plt.ylim([0,10])
   return fig

def HSVKey(dim=11):
   (h,s) = np.mgrid[0:dim,0:dim]
   h = h / (6.0*dim); s =  s / float(dim)
   v = np.ones((dim,dim))
   r,g,b = hsv_to_rgb(h,s,v)
   c = np.dstack([r,g,b])
   c = np.flipud(c) ; c = np.rot90(c)
   fig = plt.figure(frameon=False)
   ax = fig.add_subplot(111)
   ax.xaxis.set_visible(False); ax.yaxis.set_visible(False)
   plt.imshow(c, interpolation='nearest')
   return plt
