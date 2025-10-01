"Original code by Anarya Ray, adapted from https://github.com/AnaryaRay1/bbh-subpopulations-scripts/blob/main/generate_simulations_example/run_pe.py"

import numpy as np

import bilby

from copy import deepcopy

from tqdm.auto import trange

import argparse
import os
import glob
import sys
from astropy.cosmology import z_at_value, Planck15
from astropy import units

def parse_cmd():
    parser=argparse.ArgumentParser()
    parser.add_argument("--luminosityDistance", help = "luminosity distance of injected source")
    parser.add_argument("--mass1", help = "mass of the primary component injected source")
    parser.add_argument("--mass2", help = "mass of the secondary component injected source")
    parser.add_argument("--mc-min", help = "chirp mass minimum",required=False,default =None)
    parser.add_argument("--waveform-approximant", help = "waveform approximant")
    parser.add_argument("--eos", help = "eos")
    parser.add_argument("--psd-dir", help = "directory containing power spectral densities of detectors")
    parser.add_argument("--ra",help= "injected right ascension")
    parser.add_argument("--dec",help= "injected declination")
    parser.add_argument("--inc",help= "injected inclination")
    parser.add_argument("--psi",help= "injected polarization")
    parser.add_argument("--observing-run", help='observing run')
    parser.add_argument("--chi-eff",help= "injected effective spin")
    parser.add_argument("--time",help= "event gpstime")
    parser.add_argument("--phase",help= "coalesence phase")
    parser.add_argument("--outdir", help='output directory', required=False, default=None)
    parser.add_argument("--index", help=' index of the simulated event')
    args=parser.parse_args()
    return args
args=parse_cmd()

#Parse command line


D=float(args.luminosityDistance)
m1=float(args.mass1)
m2=float(args.mass2)
ra=float(args.ra)
dec=float(args.dec)
inc=float(args.inc)
psi=float(args.psi)
chi_eff = float(args.chi_eff)
phase = float(args.phase)

chi1 = chi_eff
chi2 = chi_eff


z=z_at_value(Planck15.luminosity_distance,float(D)*units.Mpc).value
obr=args.observing_run
approximant=args.waveform_approximant
eos=args.eos
psd_files=glob.glob(args.psd_dir+'*.txt')
mc=(m1*m2)**(3./5.)/(m1+m2)**(1./5.)#>1.9 or (m1*m2)**(3./5.)/(m1+m2)**(1./5.)
mc*=(1.+z)
print(mc,z,m1,m2)

injection_parameters = dict(
    chirp_mass=mc, mass_ratio=m2/m1, chi_1=chi1, chi_2=chi2, ra=ra, dec=dec, luminosity_distance=float(D), theta_jn=inc, psi=psi, phase=phase, geocent_time=float(args.time))#,fiducial=1)
#sys.exit()

if args.outdir is None:
    outdir = '/home/anarya.ray/gppop-prod/MDC_from_scratch/outdir/'
else: 
    outdir = args.outdir

if outdir[-1]!='/':
    outdir+='/'
    

outdir+=f'S2402{args.index}/'
if(not os.path.exists(outdir)):
    os.makedirs(outdir)


#initialize signal parameters
label = 'bbh'
bilby.core.utils.setup_logger(outdir=outdir, label=label)

minimum_frequency = 20
reference_frequency = 50.


signal_duration =bilby.gw.utils.calculate_time_to_merger(
            frequency=minimum_frequency,
            mass_1=m1*(1+z),
            mass_2=m2*(1+z))

duration = max(4, 2**(np.ceil(np.log2(signal_duration))))
        

# set up priors
if args.mc_min is None:
    if mc<12:
        min_mc = mc-2
    elif mc<20:
        min_mc=mc-5
    else:
        min_mc=10
else:
    min_mc = float(args.mc_min)

if mc<12:
    max_mc=15
elif mc<20:
    max_mc = 30
elif mc<40:
    max_mc=60
elif mc<80:
    max_mc = 100
else:
    max_mc= 150

priors = bilby.gw.prior.BBHPriorDict(aligned_spin=True)

priors["luminosity_distance"] = bilby.core.prior.PowerLaw(alpha=2, name='luminosity_distance', minimum=100, maximum=10000, unit='Mpc', latex_label='$d_L$')

priors["chirp_mass"] = bilby.gw.prior.UniformInComponentsChirpMass(minimum=min_mc, maximum=max_mc, name='chirp_mass', latex_label='$\\mathcal{M}$', unit=None, boundary=None)

priors["mass_1"] = bilby.core.prior.base.Constraint(minimum=5, maximum=1000, name='mass_1', latex_label='$m_1$', unit=None)
priors["mass_2"] = bilby.core.prior.base.Constraint(minimum=5, maximum=1000, name='mass_2', latex_label='$m_2$', unit=None)
print(priors["chirp_mass"])

for key in [
    "psi",
    "ra",
    "dec",
    "geocent_time",
    "phase",
]:
    priors[key] = injection_parameters[key]



while True:
    try:
        priors.validate_prior(duration, minimum_frequency)
    except ValueError as e:
        print(e)
        print(f'trying with new duration={2**(np.ceil(np.log2(duration))+1)}')
        duration = 2**(np.ceil(np.log2(duration))+1)
        continue
    break

if duration <= 8:
    sampling_frequency = 2048
elif duration > 64:
    print('Duration of 128s! Too high, skipping. Check priors.')
    raise
    #continue
else:
    sampling_frequency = 4096

   
# set up simulated data
np.random.seed(88170235)
file_to_det={'H1':"aligo",'L1':'aligo','V1':'avirgo','K1':'kagra'}

waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    frequency_domain_source_model = bilby.gw.source.lal_binary_black_hole,
    parameter_conversion = bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,

    waveform_arguments=dict(
        waveform_approximant=approximant,
        reference_frequency=reference_frequency,
        minimum_frequency=minimum_frequency
    )
)

if obr is None:
    obr='O4'

if obr=='O4':
    interferometers =bilby.gw.detector.InterferometerList(['H1', 'L1', 'V1','K1'])

    for ifo in interferometers:
        for fn in psd_files:
            if file_to_det[ifo.name] in fn:
                print(ifo.name,fn)
                ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(asd_file=fn)
elif obr=='O3':
    interferometers =bilby.gw.detector.InterferometerList(['H1', 'L1', 'V1'])

    for ifo in interferometers:
        for fn in psd_files:
            if ifo.name in fn:
                print(ifo.name,fn)
                ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(asd_file=fn)
elif obr=='O2':
    interferometers =bilby.gw.detector.InterferometerList(['H1', 'L1', 'V1'])

    for ifo in interferometers:
        ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(asd_file=psd_files[0])
else:
    print(obr)
    raise

for interferometer in interferometers:
    interferometer.minimum_frequency = minimum_frequency
    interferometer.maximum_frequency =sampling_frequency/2

interferometers.set_strain_data_from_zero_noise(sampling_frequency, duration, start_time=injection_parameters['geocent_time'] - duration + 2.)
interferometers.inject_signal(
    parameters=injection_parameters,
    waveform_generator=waveform_generator
)


        
    

# Set up the fiducial parameters for the relative binning likelihood to be the
# injected parameters. Note that because we sample in chirp mass and mass ratio
# but injected with mass_1 and mass_2, we need to convert the mass parameters
#fiducial_parameters = injection_parameters.copy()

likelihood = bilby.gw.GravitationalWaveTransient(
    interferometers=interferometers,
    waveform_generator=waveform_generator,
    priors=priors,
    distance_marginalization=True,
    #fiducial_parameters=fiducial_parameters,
)



# sampling
#npool = 8
nact = 10
nlive = 1024
result = bilby.run_sampler(
    likelihood=likelihood, priors=priors, sampler='dynesty', use_ratio=True, naccept= 60, nlive= nlive,check_point_plot= True, check_point_delta_t= 1800, print_method= 'interval-60', sample= 'acceptance-walk', npool= 1, 
    injection_parameters=injection_parameters, outdir=outdir, label=label,
    conversion_function=bilby.gw.conversion.generate_all_bbh_parameters,
    result_class=bilby.gw.result.CBCResult,
)    


# del injection_parameters["fiducial"]

# #result.plot_corner(filename=outdir+label+'_corner.png',parameters = injection_parameters,save=True)
# #result = bilby.core.result.read_in_result(filename=outdir+'bbh_relative_binning_result.json',extension='json')
# alt_waveform_generator = bilby.gw.WaveformGenerator(
#     duration=duration,
#     sampling_frequency=sampling_frequency,
#     frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
#     # frequency_domain_source_model=lal_binary_black_hole_relative_binning,
#     parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
#     waveform_arguments=dict(
#         waveform_approximant=approximant,
#         reference_frequency=reference_frequency,
#         minimum_frequency=minimum_frequency
#     ),
# )
# alt_likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
#     interferometers=interferometers,
#     waveform_generator=alt_waveform_generator,
# )
# likelihood.distance_marginalization = False
# weights = list()
# for ii in trange(len(result.posterior)):
#     parameters = dict(result.posterior.iloc[ii])
#     likelihood.parameters.update(parameters)
#     alt_likelihood.parameters.update(parameters)
#     weights.append(
#         alt_likelihood.log_likelihood_ratio() - likelihood.log_likelihood_ratio()
#     )
# weights = np.exp(weights)
# print(
#     f"Reweighting efficiency is {np.mean(weights)**2 / np.mean(weights**2) * 100:.2f}%"
# )
# print(f"Binned vs unbinned log Bayes factor {np.log(np.mean(weights)):.2f}")

# # Generate result object with the posterior for the regular likelihood using
# # rejection sampling
# alt_result = deepcopy(result)
# keep = weights > np.random.uniform(0, max(weights), len(weights))
# alt_result.posterior = result.posterior.iloc[keep]

# alt_result.save_to_file(filename = outdir+'bbh_regular_result.json')
# # Make a comparison corner plot.
# bilby.core.result.plot_multiple(
#     [result, alt_result],
#     labels=["Binned", "Reweighted"],
#     filename=outdir+f'bbh_corner.png',parameters=injection_parameters,
# )

