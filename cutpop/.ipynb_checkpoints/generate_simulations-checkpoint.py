"Original code by Anarya Ray, adapted from https://github.com/AnaryaRay1/bbh-subpopulations-scripts/blob/main/generate_simulations_example/generate_simulations.py"

import numpy as np
import bilby
from astropy.cosmology import Planck15
from astropy import units as u

from gwpopulation.models import mass

from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz
import scipy.stats as ss
import glob
import json

import matplotlib.pyplot as plt
import os

np.random.seed(88126939)
pop_file = 'pop4b.json'



with open(pop_file,'r') as jf:
    underlying_pop = json.load(jf)

#pop_file=pop_file[:pop_file.index('.json')]+'_rel2.json'

plotdir = f"plots/{pop_file[:pop_file.index('.json')]}/"
if(not os.path.exists(plotdir)):
    os.makedirs(plotdir)

gps_start = 1238112018
gps_end = 1269363618



def generate_population(Nsamples,underlying_pop):
    
    ms = np.exp(np.linspace(np.log(1),np.log(100),1000))
    m1_pdf_CE = mass.SinglePeakSmoothedMassDistribution().p_m1( {"mass_1":ms},
    alpha=underlying_pop["alpha1"],
    mmin=underlying_pop["mmin"],
    delta_m=underlying_pop["delta_m"],
    mpp=underlying_pop["mpp"],                                             sigpp=underlying_pop["sigpp"],
    mmax=underlying_pop["mmax"],
    lam=underlying_pop["lam1"])
    
    m1_pdf_CHE = mass.SinglePeakSmoothedMassDistribution().p_m1( {"mass_1":ms},
    alpha=underlying_pop["alpha2"],
    mmin=underlying_pop["mmin"],
    delta_m=underlying_pop["delta_m"],
    mpp=underlying_pop["mpp"],                                             sigpp=underlying_pop["sigpp"],
    mmax=underlying_pop["mmax"],
    lam=underlying_pop["lam2"])
    
    mix_frac = underlying_pop['f']
    
    
    
    m1_cdf_CE = cumtrapz(m1_pdf_CE, ms, initial=0)
    m1_cdf_CE/=m1_cdf_CE[-1]
    
    m1_cdf_CHE = cumtrapz(m1_pdf_CHE, ms, initial=0)
    m1_cdf_CHE/=m1_cdf_CHE[-1]
    
    if_f = np.random.uniform(0,1,size=Nsamples)
    
    m1_samples = (if_f<=mix_frac)*interp1d(m1_cdf_CE,ms)(np.random.rand(Nsamples))+(if_f>mix_frac)*interp1d(m1_cdf_CHE,ms)(np.random.rand(Nsamples))
    
    m1_samples = m1_samples[np.where(m1_samples>underlying_pop["mmin"])[0]]
    Nsamples = len(m1_samples)
                            
    
    q_samples = np.array([bilby.core.prior.analytical.PowerLaw(alpha=underlying_pop['beta_q'], minimum=underlying_pop["mmin"]/m1, maximum=1).sample(size=1)[0] for m1 in m1_samples])
    
    mu_chi_q = underlying_pop["mu_chi1"]#+underlying_pop["delta_mu_chi"]*(q_samples>0.9).astype(float)
    sigma_chi = 10.0**(underlying_pop["log10_sigma_chi1"])#+underlying_pop["delta_sigma_chi"]*(q_samples-1.0))
    
    chi_min_scaled = (-1-mu_chi_q)/sigma_chi
    chi_max_scaled = (1-mu_chi_q)/sigma_chi
    
    chieff_samples_CE = ss.truncnorm.rvs(a=chi_min_scaled,b=chi_max_scaled,loc=mu_chi_q,scale=sigma_chi,size=Nsamples)
    
    mu_chi_q = underlying_pop["mu_chi2"]#+underlying_pop["delta_mu_chi"]*(q_samples>0.9).astype(float)
    sigma_chi = 10.0**(underlying_pop["log10_sigma_chi2"])#+underlying_pop["delta_sigma_chi"]*(q_samples-1.0))
    
    chi_min_scaled = (-1-mu_chi_q)/sigma_chi
    chi_max_scaled = (1-mu_chi_q)/sigma_chi
    
    
    chieff_samples_CHE = ss.truncnorm.rvs(a=chi_min_scaled,b=chi_max_scaled,loc=mu_chi_q,scale=sigma_chi,size=Nsamples)
    
    chieff_samples = (if_f<=mix_frac)*chieff_samples_CE+(if_f>mix_frac)*chieff_samples_CHE
    
    try:
        zs = np.linspace(0.0001,underlying_pop['z_max'],1000)
    except KeyError:
        zs = np.linspace(0.0001,1,1000)
    z_pdf = 4.0*np.pi*(1+zs)**(underlying_pop["lamb"]-1)*Planck15.differential_comoving_volume(zs).to(u.Gpc**3/u.sr).value
    z_cdf = cumtrapz(z_pdf, zs, initial=0)
    z_cdf/=z_cdf[-1]
    z_samples = interp1d(z_cdf,zs)(np.random.rand(Nsamples))
    
    
    
    ra_samples = bilby.core.prior.analytical.Uniform(minimum=0, maximum=6.283185307179586, name='ra', latex_label='$\\mathrm{RA}$', unit=None, boundary='periodic').sample(size=Nsamples)
    
    dec_samples = bilby.core.prior.analytical.Cosine(minimum=-1.5707963267948966, maximum=1.5707963267948966, name='dec', latex_label='$\\mathrm{DEC}$', unit=None, boundary=None).sample(size=Nsamples)
    
    theta_jn_samples = bilby.core.prior.analytical.Sine(minimum=0, maximum=3.141592653589793, name='theta_jn', latex_label='$\\theta_{JN}$', unit=None, boundary=None).sample(size=Nsamples)
    
    psi_samples = bilby.core.prior.analytical.Uniform(minimum=0, maximum=3.141592653589793, name='psi', latex_label='$\\psi$', unit=None, boundary='periodic').sample(size=Nsamples)
    
    time_samples = np.random.uniform(gps_start,gps_end,size=Nsamples)
    
    phase_samples = bilby.core.prior.analytical.Uniform(minimum=0, maximum=6.283185307179586, name='phase', latex_label='$\\phi$', unit=None, boundary='periodic').sample(size=Nsamples)
    
    return m1_samples,q_samples,z_samples,chieff_samples, ra_samples, dec_samples, theta_jn_samples, psi_samples, time_samples, phase_samples

m1s,qs,zs,chis, ras, decs, theta_jns, psis, times, phases = generate_population(1000000,underlying_pop)

dLs = Planck15.luminosity_distance(zs).value

_=plt.hist(m1s,density=True,histtype='step',bins=20)
plt.yscale('log')
plt.xscale('log')
plt.savefig(plotdir+'m1_underlying.png')
plt.close()

_=plt.hist(qs,density=True,histtype='step',bins=20)
plt.yscale('log')
plt.savefig(plotdir+'q_underlying.png')
plt.close()

_=plt.hist(chis,density=True,histtype='step',bins=20)
plt.yscale('log')
plt.savefig(plotdir+'chi_underlying.png')
plt.close()

plt.plot(qs,chis,'.',markersize=1,alpha = 0.5)
plt.axhline(0)
plt.savefig(plotdir+'chi_q_underlying.png')
plt.close()

plt.plot(m1s,chis,'.',markersize=1,alpha = 0.5)
plt.xscale('log')
plt.savefig(plotdir+'chi_m1_underlying.png')
plt.close()

plt.plot(m1s,qs,'.',markersize=1,alpha = 0.5)
plt.xscale('log')
plt.savefig(plotdir+'q_m1_underlying.png')
plt.close()

plt.plot(m1s,qs*m1s,'.',markersize=1,alpha = 0.5)
plt.xscale('log')
plt.yscale('log')
plt.savefig(plotdir+'m2_m1_underlying.png')
plt.close()

_=plt.hist(zs,density=True,histtype='step',bins=20)
plt.yscale('log')
plt.savefig(plotdir+'z_underlying.png')
plt.close()

Nevents = 274

psd_files=glob.glob('/home/anarya.ray/gppop-prod/MDC_from_scratch/o3_noise_curves/*.txt')

minimum_frequency = 20
reference_frequency = 50.
#sampling_frequency = 2048.
#duration = 4.0
approximant = 'IMRPhenomD'

rho_threshold=9.0

interferometers =bilby.gw.detector.InterferometerList(['H1', 'L1', 'V1'])

for ifo in interferometers:
    for fn in psd_files:
        if ifo.name in fn:
            print(ifo.name,fn)
            ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(asd_file=fn)



m1_det,m2_det,dL_det,chi_eff_det,rho_opt = [ ], [ ], [ ], [ ], [ ]
n=0
with open(f"bbh_{pop_file[:pop_file.index('.json')]}_O3_zeronoise_{Nevents}_chi1chi2right.txt",'w') as f:
    
    for i,(m1,q,z,chi_eff, ra, dec, theta_jn, psi, time, phase) in enumerate(zip(m1s,qs,zs,chis, ras, decs, theta_jns, psis, times, phases)):

        dL = Planck15.luminosity_distance(z).value

        m1z = m1*(1+z)
        m2z = m1z*q
        mcz = (m1z*m2z)**(3/5)/(m1z+m2z)**(1/5)

        chi1 = chi_eff
        chi2 = chi_eff

        injection_parameters = dict(
        chirp_mass=mcz, mass_ratio=q, chi_1=chi1, chi_2=chi2, ra=ra, dec=dec, luminosity_distance=dL, theta_jn=theta_jn, psi=psi, phase=phase, geocent_time=time)
        
       
        duration = np.round(bilby.gw.utils.calculate_time_to_merger(
        frequency=minimum_frequency,
        mass_1=m1z,
        mass_2=m2z),1)
        
        duration = max(4, 2**(np.ceil(np.log2(duration))))
        
        if duration <= 8:
            sampling_frequency = 2048
        elif duration > 64:
            print('Duration of 128s! Too high, skipping. Check priors.')
            raise
            #continue
        else:
            sampling_frequency = 4096
            
        for ifo in interferometers:
            ifo.maximum_frequency =sampling_frequency/2
            ifo.minimum_frequency = minimum_frequency
            
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
        interferometers.set_strain_data_from_zero_noise(sampling_frequency, duration, start_time=injection_parameters['geocent_time'] - duration + 2.)
        #waveform_polarizations = waveform_generator.frequency_domain_strain(parameters=injection_parameters)
        
        interferometers.inject_signal(parameters=injection_parameters, waveform_generator=waveform_generator, raise_error=False)
        

        
        r_opt, r_mf = {},{}
        this_ropt = 0
        for ifo in interferometers:
            this_ropt+=ifo.meta_data['optimal_SNR']**2
            #r_opt[ifo.name]=np.absolute(ifo.optimal_snr_squared(signal))**0.5

        rho_opt_net=this_ropt**0.5
        
        if(rho_opt_net>=rho_threshold):
            if n%50==0:
                print(i,n,rho_opt_net)
            f.write(f"--psd-dir=/home/anarya.ray/gppop-prod/MDC_from_scratch/o3_noise_curves/   --mass1={m1}   --mass2={m1*q}   --luminosityDistance={dL}   --waveform-approximant=IMRPhenomPv2 --ra={ra} --dec={dec} --inc={theta_jn} --psi={psi} --chi-eff={chi_eff} --time={time} --observing-run=O3 --phase={phase} --index={int(n+10)} --outdir=/home/anarya.ray/gppop-prod/MDC_from_scratch/outdir_O3/{pop_file[:pop_file.index('.json')]}_chi1chi2right")
            f.write("\n")
            m1_det.append(m1)
            m2_det.append(m1*q)
            rho_opt.append(rho_opt_net)
            chi_eff_det.append(chi_eff)
            dL_det.append(dL)
            n+=1
        total_generated = i
        if n>=Nevents:
            break
        
print('total_generated: ',total_generated)
np.savetxt(f"observed_{pop_file[:pop_file.index('.json')]}.txt",np.array([m1_det,m2_det,dL_det,chi_eff_det,rho_opt,[total_generated for k in range(Nevents)]]).T, header='m1_source      m2_source       dL          chi_eff       rho_opt       Ndraw')

m1_det,m2_det,dL_det,chi_eff_det,rho_opt,_ = np.loadtxt(f"observed_{pop_file[:pop_file.index('.json')]}.txt",skiprows=1,unpack=True)
_=plt.hist(m1_det,density=True,histtype='step',bins=5)
plt.yscale('log')
plt.xscale('log')
plt.savefig(plotdir+'m1_observed.png')
plt.close()

_=plt.hist(m2_det/m1_det,density=True,histtype='step',bins=5)
plt.yscale('log')
plt.savefig(plotdir+'q_observed.png')
plt.close()

_=plt.hist(chi_eff_det,density=True,histtype='step',bins=5)
plt.yscale('log')
plt.savefig(plotdir+'chi_observed.png')
plt.close()

plt.plot(m2_det/m1_det,chi_eff_det,'o',markersize=4,alpha = 0.5)
plt.savefig(plotdir+'chi_q_observed.png')
plt.close()

plt.plot(m1_det,chi_eff_det,'o',markersize=4,alpha = 0.5)
plt.xscale('log')
plt.savefig(plotdir+'chi_m1_observed.png')
plt.close()

plt.plot(m1_det,m2_det/m1_det,'o',markersize=4,alpha = 0.5)
plt.xscale('log')
plt.savefig(plotdir+'q_m1_observed.png')
plt.close()

plt.plot(m1_det,m2_det,'o',markersize=4,alpha = 0.5)
plt.xscale('log')
plt.yscale('log')
plt.savefig(plotdir+'m2_m1_observed.png')
plt.close()

_=plt.hist(dL_det,density=True,histtype='step',bins=5)
plt.yscale('log')
plt.savefig(plotdir+'dL_observed.png')
plt.close()