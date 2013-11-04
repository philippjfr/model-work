"""
LESI DISPATCH
"""
CLUSTER = True

#Import Basics
import sys, os

sys.path.append(os.getcwd())

# Import Lancet
import logging
logging.basicConfig(level=logging.INFO)
from lancet import QLauncher, Launcher, Args, Range, List, review_and_launch
from lancet.topographica import Analysis, RunBatchCommand

time_array = [0,1000,5000,10000,20000]
#time_array = [250 * a for a in range(41)]

# Threads
threads = 4
Launcher = lancet.Launcher if CLUSTER else lancet.QLauncher
launcher_kwargs = dict(
    qsub_flag_options = dict(b='y',
                             pe=('memory-2G',str(threads)),
                             v='OMP_NUM_THREADS=%s' % str(threads),
                             P='inf_ndtc',
                             R='y',
                             m='a')
    ) if CLUSTER else dict(max_concurrency=1)

from analysis import save_snapshot, measure_ormap, connection_plotgroup, activity_plotgroup, projection_plotgroup,measure_spatial_curves, measure_position, measure_ortc, measure_cstc

def ormap(times=None):
    try:
        measure_ormap(frequencies=[1.6,2],sheets=['V1Exc','VSDLayer'],num_phase=18,pickle=True)(times,**kwargs)
    except:
        print "OR Map Measurement Failed"
    #measure_ormap(orientations=[2.4],sheets=['V1'],num_phase=18,pickle=True)(times,**kwargs)

def spatial_curves(times=None):
    #measure_spatial_curves(contrasts=[5,10,30,70],size_max=3.0,size_steps=16,sheet='V1',unit_list=[(0.0,0.0)])(times,**kwargs)
    try:
        measure_spatial_curves(contrasts=[5,10,30,50,70],size_max=3.0,size_steps=16,sheet='V1Exc',unit_list=[(-0.5,-0.5),(0.0,-0.5),(-0.5,0.0),(0,0),(0.0,0.5),(0.5,0.0),(0.5,0.5)])(times,**kwargs)
    except:
        print "Spatial Measurements Failed"

def position(times=None):
    try:
        measure_position(divisions=20)(times,**kwargs)
    except:
        print "Position Measurement Failed"

def ortc(times=None):
    try:
        measure_ortc(contrasts=[5,10,50,30,70],sheet='V1Exc',unit_list=[(-0.5,-0.5),(0.0,-0.5),(-0.5,0.0),(0,0),(0.0,0.5),(0.5,0.0),(0.5,0.5)])(times,**kwargs)
    except:
        print "ORTC Measurement Failed"
    #measure_ortc(contrasts=[5,10,30,70],sheet='V1',unit_list=[(0,0)])(times,**kwargs)

def cstc(times=None):
    try:
        measure_cstc(surround_contrasts=[10,30,50,70],sheet='V1Exc',unit_list=[(-0.5,-0.5),(0.0,-0.5),(-0.5,0.0),(0,0),(0.0,0.5),(0.5,0.0),(0.5,0.5)])(times,**kwargs)
    except:
        print "OSTC Measurement Failed"
    #measure_cstc(contrasts=[10,30,50,70],sheet='V1',unit_list=[(0,0)])(times,**kwargs)



models = List('model', ['LESPI'])
models = Args(model='LESPI')

model_files= {'LESPI':'./lespi.ty',
              'SCAL':'./scal.ty',
              'DPGCAL':'',
              'LESI':''}

@review_and_launch(output_directory=os.path.join(os.getcwd(), 'Output'),
                   launch_args=models, review=True)
def topo_analysis(model):

    if model == 'LESI-SOMPV':
        model_spec = Range('som_strength',-0.75,-1.25,3) *  Range('lat_som_strength',2.0,2.5,3) * Range('lat_som_loc_strength',0.75,1.0,2) * Range('som_pv_strength',-0.75,-1.25,3) * List('dataset',["'Nature'","'Gaussian'"])
    elif model == 'LESI':
        model_spec = Range('lat_exc_strength',0.0,1.0,5) * List('dataset',["'Nature'","'Gaussian'"])
    elif model == 'SCAL':
        model_spec = Range('exc_strength',1.0,1.0,1) * Range('inh_strength',2.0,2.0,1) * Range('area',3.0,3.0,1) * List('dataset',["'Nature'","'Gaussian'"])
    elif model == 'DPGCAL':
        model_spec = Range('lat_loc_strength',3.0,3.0,1) * Range('pv_strength',3.25,3.25,1) * Range('lat_pv_strength',3.25,3.25,1) * List('dataset',["'Nature'","'Gaussian'"])

    # Completed Specifier
    combined_spec = Args(times=time_array) * run_batch_spec * model_spec
    # Command
    run_batch_command = RunBatchCommand(model_files[model], snapshot=False)
    # Launcher
    tasklauncher = Launcher(model, combined_spec, run_batch_command, **launcher_kwargs)
    analysis = Analysis(paths=[os.getcwd()])

    # Analysis
    analysis.add_analysis_fn(activity_plotgroup)#, 'Measure and save activity plots')
    # #analysis.add_analysis_fn(activity_stats)#, 'Activity Statistics')
    # analysis.add_analysis_fn(projection_plotgroup)#, 'Store Projections')
    analysis.add_analysis_fn(connection_plotgroup)#, 'Save connection fields of specific units')
    # #analysis.add_analysis_fn(unit_bitpattern)# , 'Save incoming bit patterns for specific units.')
    #analysis.add_analysis_fn(position)# , 'Measure SFTC and ASC fitting DoG and iDoG')
    analysis.add_analysis_fn(ormap)#, 'Measure and save OR plot.')
    analysis.add_analysis_fn(save_snapshot)#, 'Save Snapshot')
    #analysis.add_analysis_fn(ortc)#, 'Measure OR Tuning curves')
    #analysis.add_analysis_fn(cstc)#, 'Measure Orientation Contrast Tuning curves')
    #analysis.add_analysis_fn(spatial_curves)#, 'Measure SFTC and ASC fitting DoG and iDoG')
    # #analysis.add_analysis_fn(rf_analysis)#, 'Save incoming bit patterns for specific units.')
    return tasklauncher

if __name__=='__main__': topo_analysis()
