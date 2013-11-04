from scipy.optimize import curve_fit
from imagen import Disk, RawRectangle
from topo.analysis.featureresponses import CoordinatedPatternGenerator
from topo.misc.distribution import DSF_MaxValue, DSF_WeightedAverage
from topo.command.analysis import save_plotgroup, measure_sine_pref, measure_rfs
from topo.command.pylabplot import measure_position_pref

import matplotlib.pyplot as plt
import numpy as np
import param
import math
import pickle
import operator

# Import Analysis Functions
from hypercolumn import *
from fitting import *
from image_utils import *
from utils import image
import analysisJL

def analysis_group(group):
    group_b = [100,1000,2000,2500,3000,4000,5000,10000,20000]
    group_a = [i for i in range(0,20001) if (i % 1 == 0)]

    if (topo.sim.time() in group_a) and (group == 'a'):
        return False
    if (topo.sim.time() in group_b) and (group == 'b'):
        return False

    return True

class measure_spatial_curves(param.Parameterized):

    sf_measure = param.Boolean(default=True)

    size_measure = param.Boolean(default=True)

    sheet = param.String(default=None)

    contrasts = param.List(default=[])

    num_phase = param.Number(default=9)

    sf_size = param.Number(default=1.0)

    unit_list = param.List(default=[])

    sf_max = param.Number(default=5.0)

    sf_steps = param.Integer(default=21)

    size_max = param.Number(default=2.0)

    size_steps = param.Integer(default=21)

    sf_fallback = param.Number(default=1.8)

    pickle = param.Boolean(default=True)

    def __init__(self,**params):
        super(measure_spatial_curves,self).__init__(**params)

        self.c_dict = []
        for contrast in self.contrasts:
            self.c_dict.append({"contrast":contrast})

        self.dims = (len(self.contrasts),len(self.unit_list))
        self.sf_fit = (0,0.1429,0.1428,0.45,0.16)
        self.size_fit = (0,25,10,0.05,0.43)

        self.ASTC = np.zeros((self.size_steps,)+self.dims)
        self.size_fitted_curve = np.zeros((self.size_steps,)+self.dims)
        self.size_fitv = np.zeros((8,)+self.dims)
        self.size_est = np.zeros((3,)+self.dims)

        self.SFTC = np.zeros((self.sf_steps,)+self.dims)
        self.sf_fitted_curve = np.zeros((self.sf_steps,)+self.dims)
        self.sf_fitv = np.zeros((8,)+self.dims)


    def __call__(self,times=None):
        if topo.sim.time() != times[-1]:
            return None

        if self.sf_measure:
            self.frequencies = np.linspace(0.0,self.sf_max,self.sf_steps)
            self.measure_SFTC()

            try:
                self.fit_DoG()
            except:
                pass

            #self.plot_SFTC()

            if self.pickle:
                pkl_file = open(param.normalize_path('SFTC_%s.pkl' % self.sheet), 'wb')
                pickle.dump((self.SFTC,self.frequencies,self.contrasts,self.unit_list), pkl_file)
                pkl_file.close()

        if self.size_measure:
            self.sizes = np.linspace(0.0,self.size_max,self.size_steps)

            self.frequency = [0]*len(self.unit_list)
            for uidx,unit in enumerate(self.unit_list):
                if self.sf_measure:
                    self.frequency[uidx] = self.frequencies[self.SFTC[:,-1,uidx].argmax()]
                else:
                    self.frequency[uidx] = self.sf_fallback

            self.measure_ASTC()

            if self.pickle:
                pkl_file = open(param.normalize_path('ASTC_%s.pkl' % self.sheet), 'wb')
                pickle.dump((self.ASTC,(self.sizes,self.contrasts,self.unit_list)), pkl_file)
                pkl_file.close()

            try:
                self.fit_iDoG()
            except:
                pass

            #self.plot_ASTC()

    def fit_iDoG(self):
        for uidx,unit in enumerate(self.unit_list):
            for cidx,contrast in enumerate(self.contrasts):
                max_idx = np.argmax(self.ASTC[:,cidx,uidx])
                r_max = self.sizes[max_idx]
                min_idx = np.argmin(self.ASTC[max_idx:,cidx,uidx]) + max_idx
                r_min = self.sizes[min_idx]
                cs_idx = np.argmax(self.ASTC[min_idx:,cidx,uidx]) + min_idx
                r_cs = self.sizes[cs_idx]
                self.size_est[0,cidx,uidx,sidx] = r_max
                self.size_est[1,cidx,uidx,sidx] = r_min
                self.size_est[2,cidx,uidx,sidx] = r_cs
                if 'LGN' in self.sheet:
                    fit,pcov = curve_fit(lgn_idogmodel,self.sizes,self.ASTC[:,cidx,uidx],self.size_fit,maxfev=10000)
                elif 'V1' in self.sheet:
                    fit,pcov = curve_fit(lgn_idogmodel,self.sizes,self.ASTC[:,cidx,uidx],self.size_fit,maxfev=10000)
                for fidx,s in enumerate(self.sizes):
                    if 'LGN' in self.sheet:
                        self.size_fitted_curve[fidx,cidx,uidx] = lgn_idogmodel(s,*fit)
                    elif 'V1' in self.sheet:
                        self.size_fitted_curve[fidx,cidx,uidx] = v1_normmodel(s,*fit)
                    self.size_fitv[0,cidx,uidx] = math.sqrt(fit[3]*2)
                    self.size_fitv[1,cidx,uidx] = math.sqrt(fit[4]*2)
                    self.size_fitv[2,cidx,uidx] = (fit[2]*fit[4])/(fit[1]*fit[3])
                    for fidx in range(0,len(fit)):
                        self.size_fitv[fidx+3,cidx,uidx] = fit[fidx]


    def fit_DoG(self):
        for uidx,unit in enumerate(self.unit_list):
            for cidx,contrast in enumerate(self.contrasts):
                fit,pcov = curve_fit(lgn_dogmodel,self.frequencies,self.SFTC[:,cidx,uidx],self.sf_fit,maxfev=100000)
                for fidx,f in enumerate(self.frequencies):
                    self.sf_fitted_curve[fidx,cidx,uidx] = lgn_dogmodel(f,fit[0],fit[1],fit[2],fit[3],fit[4])
                self.sf_fitv[0,cidx,uidx] = fr2sp(fit[3])
                self.sf_fitv[1,cidx,uidx] = fr2sp(fit[4])
                self.sf_fitv[2,cidx,uidx] = (fit[2]*fit[4])/(fit[1]*fit[3])
                for fidx in range(0,len(fit)):
                    self.sf_fitv[fidx+3,cidx,uidx] = fit[fidx]


    def plot_SFTC(self):
        """
        Plot SFTC and Fitted DoG Curve
        """

        f = open(param.normalize_path('DoG_Fit_%s.txt' % self.sheet), 'w')
        for uidx,unit in enumerate(self.unit_list):
            for cidx,contrast in enumerate(self.contrasts):
                fig = plt.figure()
                plt.title('{sheet}{unit} Contrast:{contrast}% Time:{time} SF Tuning Curve'.format(sheet=self.sheet,unit=topo.sim[self.sheet].sheet2matrixidx(unit[0],unit[1]),contrast=contrast,time=topo.sim.time()))
                plt.plot(self.frequencies,self.SFTC[:,cidx,uidx],label='Measured Response')
                plt.plot(self.frequencies,self.sf_fitted_curve[:,cidx,uidx],label='Fitted Response - SI: {0:.3f} o_e: {1:.3f} vdeg, o_i: {2:.3f} vdeg'.format(self.sf_fitv[2,cidx,uidx],self.sf_fitv[0,cidx,uidx],self.sf_fitv[1,cidx,uidx]))
                plt.xlabel('Spatial Frequency (cyc/deg)')
                plt.ylabel('Activity')
                lgd = plt.legend(loc = 'upper center', bbox_to_anchor = (0.5, -0.1))
                path = param.normalize_path('{sheet}{unit}{contrast}__{time:09.2f}_SFTC.pdf'.format(sheet=self.sheet,unit=str(unit),time=float(topo.sim.time()),contrast=contrast))
                plt.savefig(path,bbox_extra_artists=[lgd],bbox_inches='tight',format='pdf')
                plt.close(fig)
                f.write('{sheet}{unit}{contrast}: {fit}\n'.format(sheet=self.sheet,unit=str(unit),contrast=contrast,fit=str(self.sf_fitv[:,cidx,uidx])))
        f.close()

    def plot_ASTC(self):
        f = open(param.normalize_path('iDoG_Fit_%s.txt' % self.sheet), 'w')
        for uidx,unit in enumerate(self.unit_list):
            for cidx,contrast in enumerate(self.contrasts):
                fig = plt.figure()
                plt.title('{sheet}{unit} Contrast:{contrast}% Time:{time} Area Summation Curve'.format(sheet=self.sheet,unit=topo.sim[self.sheet].sheet2matrixidx(unit[0],unit[1]),contrast=contrast,time=str(float(topo.sim.time()))))
                plt.plot(self.sizes,self.ASTC[:,cidx,uidx],label='Measured Response')
                if self.size_fitted_curve != None:
                    plt.plot(self.sizes,self.size_fitted_curve[:,cidx,uidx],label='Fitted Response - SI: {0:.3f} o_e: {1:.3f} vdeg, o_i: {2:.3f} vdeg'.format(self.size_fitv[2,cidx,uidx],self.size_fitv[0,cidx,uidx],self.size_fitv[1,cidx,uidx]))
                plt.xlabel('Stimulus Diameter (in deg of visual angle)')
                plt.ylabel('Activity')
                lgd = plt.legend(loc = 'upper center', bbox_to_anchor = (0.5, -0.1))
                path = param.normalize_path('{sheet}{unit}{contrast}__{time:09.2f}_Size_TC.pdf'.format(sheet=self.sheet,unit=str(unit),time=float(topo.sim.time()),contrast=contrast))
                plt.savefig(path,bbox_extra_artists=[lgd],bbox_inches='tight',format='pdf')
                plt.close(fig)
                f.write('{unit},{contrast}: {fit};{estim}\n'.format(sheet=self.sheet,unit=str(unit),contrast=contrast,fit=str(self.size_fitv[:,cidx,uidx]),estim=self.size_est[:,cidx,uidx]))
        f.close()


    def measure_ASTC(self):
        """
        Measure Size Tuning Response for different sheets and units,
        store and return results.
        """
        for uidx,coord in enumerate(self.unit_list):
            unit = topo.sim[self.sheet].sheet2matrixidx(coord[0],coord[1])
            if 'LGN' in self.sheet:
                topo.command.pylabplot.measure_size_response(preference_fn=DSF_MaxValue,curve_parameters=self.c_dict,num_sizes=self.size_steps-1,
                                                             max_size=self.size_max,num_phase=self.num_phase,sheet=self.sheet,coords=[coord],
                                                             pattern_presenter=CoordinatedPatternGenerator(pattern_generator=Disk(smoothing=0.0,contrast_parameter="weber_contrast")))
            else:
                topo.command.pylabplot.measure_size_response(preference_fn=DSF_MaxValue,curve_parameters=self.c_dict,num_sizes=self.size_steps-1,
                                                             max_size=self.size_max,num_phase=self.num_phase,sheet=self.sheet,
                                                             frequencies=[self.frequency[uidx]],coords=[coord])
            for cidx,contrast in enumerate(self.contrasts):
                diameters = sorted(topo.sim[self.sheet].curve_dict['size']['Contrast = {c}%'.format(c=int(contrast))].keys(),key=lambda x: float(x))
                for didx,diameter in enumerate(diameters):
                    activity = topo.sim[self.sheet].curve_dict['size']['Contrast = {c}%'.format(c=int(contrast))][diameter].view()[0]
                    self.ASTC[didx,cidx,uidx] = activity[unit[0]][unit[1]]


    def measure_SFTC(self):
        """
        Measure Spatial Frequency Tuning Response
        """
        for uidx,coord in enumerate(self.unit_list):
            unit = topo.sim[self.sheet].sheet2matrixidx(coord[0],coord[1])
            topo.command.pylabplot.measure_frequency_response(preference_fn=DSF_MaxValue,curve_parameters=self.c_dict,
                                                              max_freq=self.sf_max,num_freq=self.sf_steps-1,
                                                              num_phase=self.num_phase,sheet=self.sheet,
                                                              size=self.sf_size,coords=[coord])
            for cidx,contrast in enumerate(self.contrasts):
                fr_keys = sorted(topo.sim[self.sheet].curve_dict['frequency']['Contrast = {c}%'.format(c=int(contrast))].keys(),key=lambda x: float(x))
                for fidx,frequency in enumerate(fr_keys):
                    activity = topo.sim[self.sheet].curve_dict['frequency']['Contrast = {c}%'.format(c=int(contrast))][frequency].view()[0]
                    self.SFTC[fidx,cidx,uidx] = activity[unit[0]][unit[1]]

class measure_ormap(param.Parameterized):

    frequencies = param.List(default=[1.8])

    num_phase = param.Number(default=9)

    num_orientation = param.Number(default=9)

    pickle = param.Boolean(default=False)

    sheets = param.List(default=[])

    def __init__(self,**params):
        super(measure_ormap,self).__init__(**params)
        if len(self.sheets) == 0: print "Warning no measurement sheet specified"

        #self.__name__ = 'measure_ormap'

    def __call__(self,times=None,**kwargs):
        # if analysis_group('b'):
        #     return None

        if self.pickle:
            or_pkl = open(param.normalize_path('ormap_{time}.pkl'.format(time=topo.sim.time())),'wb')
            pkl_data = {}

        for f in self.frequencies:
            save_plotgroup("Orientation Preference",use_cached_results=False,saver_params={'filename_suffix':'_{0}'.format(f)},
                           pre_plot_hooks=[measure_sine_pref.instance(frequencies=[f],num_phase=self.num_phase,num_orientation=self.num_orientation,
                                                                      preference_fn=DSF_WeightedAverage(value_scale=(0., 1./math.pi)))])
            if self.pickle:
                for sheet in self.sheets:
                    im = topo.sim[sheet].sheet_views['OrientationPreference'].view()[0][0:-1,0:-1]
                    try:
                        polar_or = image.hue_to_polar(im)
                        pwdata = analysisJL.polarmap_contours(polar_or)
                        pws = analysisJL.identify_pinwheels(*pwdata)
                        pwbitsofinformation = type('pwdata', (), dict(zip(['recontours',  'imcontours', 'intersections', 'pinwheels'] , pwdata+(pws,)  )))
                        pw_results = analysisJL.pinwheel_analysis(im, len(pws), ignore_DC=False)
                        pkl_data['OR_Analysis_{freq}_{sheet}'.format(freq = f,sheet=sheet)] = pw_results
                    except:
                        print "OR Pinwheel Analysis failed"
                    or_sel = topo.sim[sheet].sheet_views['OrientationSelectivity'].view()[0]
                    pkl_data['OrientationPreference_{freq}_{sheet}'.format(freq = f,sheet=sheet)] = im
                    pkl_data['OrientationSelectivity_{freq}_{sheet}'.format(freq = f,sheet=sheet)] = or_sel
        if self.pickle:
            pickle.dump(pkl_data,or_pkl)


class measure_position(param.Parameterized):

    divisions = param.Number(default=20)

    x_range = param.NumericTuple((-0.5,0.5))

    y_range = param.NumericTuple((-0.5,0.5))

    size = param.Number(default=0.1)

    scale = param.Number(default=1.0)

    def __init__(self,**params):
        super(measure_position,self).__init__(**params)

    def __call__(self,times=None,**kwargs):
        if analysis_group('b'):
            return None

        save_plotgroup("Position Preference",use_cached_results=False,
                       pre_plot_hooks=[measure_position_pref.instance(divisions=self.divisions,
                                                                      x_range=self.x_range,
                                                                      y_range=self.y_range,
                                                                      size=self.size,scale=self.scale)])


class measure_rf(param.Parameterized):

    input_sheet = param.String(default="Retina")

    measurement_sheet = param.String(default=None)

    sampling_rate = param.Number(default=2.0)

    unit_slice = param.NumericTuple(default=(50,50))

    pickle = param.Boolean(default=True)


    def __init__(self,**params):
        super(measure_rf,self).__init__(**params)

    def __call__(self,times=None,**kwargs):
        if topo.sim.time() != times[-1]:
            return None

        measure_rfs(sampling_area=self.unit_slice,input_sheet=topo.sim[self.input_sheet],
                    sampling_interval=self.sampling_rate,pattern_presenter=CoordinatedPatternGenerator(RawRectangle()))

        # Get Retina Information
        retina = topo.sim[self.input_sheet]
        sheet = topo.sim[self.measurement_sheet]
        left, bottom, right, top = sheet.nominal_bounds.lbrt()
        sheet_density = float(sheet.nominal_density)
        x_units,y_units = sheet.shape
        unit_size = 1.0 / sheet_density
        half_unit_size = (unit_size / 2.0)   # saves repeated calculation.

        # Sample V1
        v1_units = 4

        sampling_range = (top - half_unit_size - (unit_size * np.floor((y_units-v1_units)/2)) , bottom + half_unit_size +  (unit_size * np.ceil(y_units-v1_units)/2))

        # Create Samplign Grid
        spacing = np.linspace(sampling_range[0],sampling_range[1], sheet.density)
        X, Y = np.meshgrid(spacing, spacing)
        sheet_coords = zip(X.flatten(),Y.flatten())
        coords = list(set([sheet.sheet2matrixidx(x,y) for (x,y) in sheet_coords]))

        # Filter and sort RFs
        keys = [key for key in retina.sheet_views.keys() if key[0] == 'RFs']
        filtered_keys = sorted(list(set([key for key in keys if sheet.sheet2matrixidx(key[2],key[3]) in coords])),key=operator.itemgetter(2,3))
        rfs = np.dstack([topo.sim[self.input_sheet].sheet_views[key].view()[0] for key in filtered_keys])
        coords = np.vstack([(key[2],key[3]) for key in filtered_keys])
        plt.imsave(param.normalize_path('central_rf.png'),rfs[:,:,int(len(rfs[0,0,:])/2)])

        # Pickle and save
        if self.pickle:
            pkl_file = open(param.normalize_path('RFs.pkl'), 'wb')
            pickle.dump((coords,rfs), pkl_file)
            pkl_file.close()

        try:
            img = rf_image(rfs,coords,norm='All')
            img.save(param.normalize_path('V1_RFs_{time}.png'.format(time=topo.sim.time())))
        except:
            pass

def save_snapshot(times=None, **kwargs):
    if topo.sim.time() != times[-1]:
        return None
    from topo.command import save_snapshot
    save_snapshot()


def activity_plotgroup(times=None, **kwargs):
    if analysis_group('a'):
        return None

    from topo.command.analysis import save_plotgroup
    save_plotgroup("Activity")

def activity_stats(times=None, **kwargs):
    if analysis_group('a'):
        return None

    sheets = ['LGNOff','LGNOn','V1Exc','V1Inh']

    f = open(param.normalize_path('mean_activity.txt'), 'a')
    for sheet in sheets:
        mean_activity = np.mean(topo.sim[sheet].activity)
        max_activity = np.max(topo.sim[sheet].activity)
        f.write('{sheet},{time}: Mean {act:.3f} Max {max_act:.3f}\n'.format(sheet=sheet,time=topo.sim.time(),act=mean_activity,max_act=max_activity))
    f.close()

def projection_plotgroup(times=None,**kwargs):
    if analysis_group('b'):
        return None

    from topo.command.analysis import save_plotgroup
    projections = [('V1PV','PVLGNOnAfferent'),('V1PV','PVLGNOffAfferent'),('V1Exc','LGNOnAfferent'),('V1Exc','LGNOffAfferent'),('V1Exc','LateralExcitatory'),('V1Exc','V1PV2V1Exc'),('V1SOM','V1Exc2V1SOM'),('V1Exc','V1PV2V1Exc'),('V1','LGNOnAfferent'),('V1','LGNOffAfferent'),('V1PV','V1Exc2V1PV'),('V1','LateralInhibitory')]
    
    for proj in projections:
        try:
            save_plotgroup("Projection",projection=topo.sim[proj[0]].projections(proj[1]))
        except:
            print "{0} does not exist".format(proj)
                

def connection_plotgroup(times=None,**kwargs):
    if analysis_group('b'):
        return None

    projections = [('V1PV','PVLGNOnAfferent'),('V1PV','PVLGNOffAfferent'),('V1Exc','LGNOnAfferent'),('V1Exc','LGNOffAfferent'),('V1Exc','LateralExcitatory'),('V1Exc','V1PV2V1Exc'),('V1SOM','V1Exc2V1SOM'),('V1Exc','V1PV2V1Exc'),('V1','LGNOnAfferent'),('V1','LGNOffAfferent'),('V1PV','V1Exc2V1PV'),('V1','LateralInhibitory')]
    units = [(0,0)]

    for sidx,projection in enumerate(projections):
        for uidx,unit in enumerate(units):
            try:
                coords = topo.sim[projection[0]].sheet2matrixidx(unit[0],unit[1])
                weights = topo.sim[projection[0]].projections()[projection[1]].cfs[coords[0]][coords[1]].weights
                max_w=np.max(weights)
                plt.figure()
                path = param.normalize_path('{sheet}{unit}__{time:09.2f}_{projection}_ConnectionField_{maxw:.3f}.png'.format(sheet=projection[0],projection=projection[1],unit=str(unit[0]),time=float(topo.sim.time()),maxw=float(max_w)))
                plt.imsave(path, weights)
            except:
                print "{0} does not exist".format(projection)

                
def unit_bitpattern(times=None,**kwargs):

    if analysis_group('b'):
        return None

    sheets = [('V1Exc','LGNOff','LGNOffAfferent'),('V1Exc','LGNOn','LGNOnAfferent')]
    units = [(0,0)]

    f = open(param.normalize_path('mean_bitpattern.txt'), 'a')
    for sidx, sheet in enumerate(sheets):
        for uidx,unit in enumerate(units):
            coords = topo.sim[sheet[0]].sheet2matrixidx(unit[0],unit[1])
            shape = topo.sim[sheet[0]].projections()[sheet[2]].cfs[coords[0]][coords[1]].weights.shape
            coords = topo.sim[sheet[1]].sheet2matrixidx(unit[0],unit[1])
            if shape[0] % 2:
                activity = topo.sim[sheet[1]].activity[coords[0]-np.ceil(shape[0]/2):coords[0]+np.floor(shape[0]/2),coords[1]-np.ceil(shape[1]/2):coords[1]+np.floor(shape[1]/2)]
            else:
                activity = topo.sim[sheet[1]].activity[coords[0]-np.floor(shape[0]/2):coords[0]+np.floor(shape[0]/2),coords[1]-np.floor(shape[1]/2):coords[1]+np.floor(shape[1]/2)]
            path = param.normalize_path('{unit}__{time:09.2f}_{sheet}BitPattern.png'.format(sheet=sheet[2],unit=str(unit),time=float(topo.sim.time())))
            plt.imsave(path, activity, cmap=plt.cm.gray)
            mean_activity = np.mean(activity)
            f.write('{sheet},{unit},{time}: {act:.3f}\n'.format(sheet=sheet[2],unit=unit,time=topo.sim.time(),act=mean_activity))
    f.close()

class measure_ortc(param.Parameterized):

    sheet = param.String(default=None)

    num_phases = param.Number(default=9)

    num_orientation = param.Number(default=12)

    contrasts = param.List(default=[10,100])

    unit_list = param.List(default=[(0,0)])

    pickle = param.Boolean(default=True)

    def __init__(self,**params):
        super(measure_ortc,self).__init__(**params)

        self.c_dict = []
        for contrast in self.contrasts:
            self.c_dict.append({"contrast":contrast})

        self.ORTC = np.zeros((self.num_orientation,len(self.contrasts),len(self.unit_list)))

    def __call__(self,times=None,**kwargs):
        if topo.sim.time() != times[-1]:
            return None

        topo.command.pylabplot.measure_or_tuning_fullfield(preference_fn=DSF_MaxValue,curve_parameters=self.c_dict,
                                                           num_phase=self.num_phases,sheet=self.sheet)
        for uidx,coord in enumerate(self.unit_list):
            unit = topo.sim[self.sheet].sheet2matrixidx(coord[0],coord[1])
            for cidx,contrast in enumerate(self.contrasts):
                or_keys = sorted(topo.sim[self.sheet].curve_dict['orientation']['Contrast = {c}%'.format(c=int(contrast))].keys(),key=lambda x: float(x))
                for oidx,orientation in enumerate(or_keys):
                    activity = topo.sim[self.sheet].curve_dict['orientation']['Contrast = {c}%'.format(c=int(contrast))][orientation].view()[0]
                    self.ORTC[oidx,cidx,uidx] = activity[unit[0]][unit[1]]

        if self.pickle:
            pkl_file = open(param.normalize_path('ORTC.pkl'), 'wb')
            pickle.dump((self.ORTC,(or_keys,self.contrasts,self.unit_list,self.sheet)), pkl_file)
            pkl_file.close()


class measure_cstc(param.Parameterized):

    sheet = param.String(default=None)

    num_phases = param.Number(default=9)

    num_orientation = param.Number(default=12)

    center_contrast = param.Number(default=50)

    sizecenter=param.Number(default=0.25,bounds=(0,None),doc="""
        The size of the central pattern to present.""")

    sizesurround=param.Number(default=1.0,bounds=(0,None),doc="""
        The size of the surround pattern to present.""")

    thickness=param.Number(default=0.5,bounds=(0,None),softbounds=(0,1.5),doc="""Ring thickness.""")

    surround_contrasts = param.List(default=[5,10,30,50,70])

    unit_list = param.List(default=[(0,0)])

    pickle = param.Boolean(default=True)

    def __init__(self,**params):
        super(measure_cstc,self).__init__(**params)

        self.c_dict = []
        for contrast in self.surround_contrasts:
            self.c_dict.append({"contrastsurround":contrast})

        self.CSTC = np.zeros((self.num_orientation,9,len(self.surround_contrasts),len(self.unit_list)))

    def __call__(self,times=None,**kwargs):
        if topo.sim.time() != times[-1]:
            return None

        for coidx,coord in enumerate(self.unit_list):
            prefix = str(coord) + "_"
            unit = topo.sim[self.sheet].sheet2matrixidx(coord[0],coord[1])
            topo.command.pylabplot.measure_orientation_contrast(preference_fn=DSF_MaxValue,curve_parameters=self.c_dict,
                                                                num_phase=self.num_phases,sheet=self.sheet,coords=[coord],
                                                                contrastcenter=self.center_contrast,num_orientation=self.num_orientation,
                                                                sizecenter=self.sizecenter,sizesurround=self.sizesurround,thickness=self.thickness,
                                                                measurement_prefix=prefix)
            for cidx,contrast in enumerate(self.surround_contrasts):
                or_keys = sorted(topo.sim[self.sheet].curve_dict[prefix+'orientationsurround']['Contrastsurround = {c}%'.format(c=int(contrast))].keys(),key=lambda x: float(x))
                for oidx,orientation in enumerate(or_keys):
                    unit = topo.sim[self.sheet].sheet2matrixidx(coord[0],coord[1])
                    units = [(unit[0]+x_offset,unit[1]+y_offset) for x_offset in xrange(-1,2) for y_offset in  xrange(-1,2)]
                    activity = topo.sim[self.sheet].curve_dict[prefix+'orientationsurround']['Contrastsurround = {c}%'.format(c=int(contrast))][orientation].view()[0]
                    for uidx,unit in enumerate(units):
                        self.CSTC[oidx,uidx,cidx,coidx] = activity[unit[0]][unit[1]]

        if self.pickle:
            pkl_file = open(param.normalize_path('CSTC.pkl'), 'wb')
            pickle.dump((self.CSTC,(or_keys,self.unit_list,self.surround_contrasts,self.sheet,self.unit_list)), pkl_file)
            pkl_file.close()


