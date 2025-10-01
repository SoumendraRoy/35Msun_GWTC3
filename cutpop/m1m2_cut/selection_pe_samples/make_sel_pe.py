import h5py
import numpy as np
import matplotlib.pyplot as plt
import astropy.cosmology as cosmo
from astropy.cosmology import Planck15 as lvk_cosmology
import astropy.units as u
import copy
from scipy.stats import truncnorm
import scipy.stats as ss
import pandas as pd
from scipy.interpolate import RectBivariateSpline, interp1d
from tqdm import tqdm

seed = 1023895

################################################################################################################
# # Load GWTC3 Selection File
def chi_eff_marginal(chi_eff, q):
    # Placeholder: replace with your actual marginalization function
    return np.ones_like(chi_eff)

def load_selection(file):
    with h5py.File(file, 'r') as f:
        # Load dataset into DataFrame
        injections = f['injections']
        df = pd.DataFrame({key: injections[key][()] for key in injections.keys()})
        
        # Get metadata
        T_yr = f.attrs['analysis_time_s'] / (3600 * 24 * 365.25)
        N = f.attrs['total_generated']

    # Compute new columns
    df['mass_1'] = df['mass1_source'] * (1 + df['redshift'])
    df['q'] = df['mass2_source'] / df['mass1_source']
    df['chi_eff'] = (df['spin1z'] + df['q'] * df['spin2z']) / (1 + df['q'])

    # Spin magnitudes
    a1 = np.sqrt(df['spin1x']**2 + df['spin1y']**2 + df['spin1z']**2)
    a2 = np.sqrt(df['spin2x']**2 + df['spin2y']**2 + df['spin2z']**2)

    # Spin sampling PDF
    spin_sampling_pdf = 1 / (16 * np.pi**2 * a1**2 * a2**2)

    # Sampling PDFs
    df['sampling_pdf_qchieff'] = (
        df['sampling_pdf'] / spin_sampling_pdf * df['mass1_source'] *
        chi_eff_marginal(df['chi_eff'], df['q'])
    )

    df['sampling_pdf_q'] = df['sampling_pdf'] * df['mass1_source']

    # Luminosity distance in Gpc (vectorized)
    df['luminosity_distance'] = lvk_cosmology.luminosity_distance(df['redshift'].values).to(u.Gpc).value
    
    # Comoving distance in Gpc
    dc = lvk_cosmology.comoving_transverse_distance(df['redshift'].values).to(u.Gpc).value
    
    # Hubble distance (c / H(z)) in Gpc
    dh_z = (2.99792e8 * u.m / u.s / lvk_cosmology.H(df['redshift'].values)).to(u.Gpc).value
    
    # Final PDF
    df['sampling_pdf_m1dqdlchieff'] = (
        df['sampling_pdf_qchieff'] / (1 + df['redshift']) / (dc + (1 + df['redshift']) * dh_z)
    )

    return df, T_yr, N

################################################################################################################

injection_file = "/mnt/home/ccalvk/ceph/sensitivity-estimates/o1+o2+o3_bbhpop_real+semianalytic-LIGO-T2100377-v2.hdf5"
df, T_yr, N_selection = load_selection(injection_file)
default_far_thresh = 1
default_snr_thresh = 10
df = df[
    ((df['name'] == b'o3') & ((df['far_cwb'] < default_far_thresh) | (df['far_gstlal'] < default_far_thresh) | (df['far_mbta'] < default_far_thresh) | (df['far_pycbc_bbh'] < default_far_thresh) | (df['far_pycbc_hyperbank'] < default_far_thresh)))|
    ((df['name'].isin([b'o1', b'o2'])) & (df['optimal_snr_net'] > default_snr_thresh))]
print('Number of Selection Samples=', len(df), 'Dataframe Keys:', df.keys())

################################################################################################################

# Construction of Likelihood
with h5py.File('optimal_snr_aligo_O3actual_L1_100gridpoints.h5', 'r') as inp:
    ms = np.array(inp['ms'])
    osnrs = np.array(inp['SNR'])

osnr_interp = RectBivariateSpline(ms, ms, osnrs)

def snr_unit_dl_unit_theta(m1z, m2z):
    return osnr_interp.ev(m1z, m2z)

m1z = np.array(df['mass_1'])
m2z = np.array(df['mass_1'])*np.array(df['q'])
dl = np.array(df['luminosity_distance'])
rho = np.array(df['optimal_snr_net'])
Theta = (1./3.**0.5)*dl*rho/snr_unit_dl_unit_theta(m1z, m2z)

print("Deleting the Theta>1 Samples:", len(np.where(Theta>1)[0])) # this can potentially give bias if lots of samples rejected.
m1z, m2z, dl, rho, Theta, df = m1z[Theta<1.], m2z[Theta<1.], dl[Theta<1.], rho[Theta<1.], Theta[Theta<1.], df[Theta<1.]
rho_obs = Theta*snr_unit_dl_unit_theta(m1z, m2z)/dl + np.random.randn(len(m1z))
print(np.where(rho_obs<0))

################################################################################################################

uncert = {
    'threshold_snr': 8,
    'Theta': 0.05,
    'mcz': 0.03,
    'logq': 0.15 # change it to 0.3 for cut in m1 to match the catalog errorbar of m1.
}

def mcz_add_err(Mcz, rho_obs, uncert, Nsamp):
    Nobs = Mcz.shape[0]
    sigma_Mcz = uncert['threshold_snr']/rho_obs*uncert['mcz']
    logMczo = np.log(Mcz) + sigma_Mcz*np.random.randn(Nobs)
    logMczs = logMczo[:,None] + sigma_Mcz[:,None]*(np.random.randn(Nobs*Nsamp).reshape(Nobs,Nsamp))
    return np.exp(logMczs)

'''def mcz_add_err(Mcz, rho_obs, uncert, Nsamp):
    Nobs = Mcz.shape[0]
    sigma_Mcz = uncert['threshold_snr']/rho_obs*uncert['mcz']
    Mczo = Mcz+ sigma_Mcz*np.random.randn(Nobs)
    Mczs = np.random.normal(Mczo[:,None], sigma_Mcz[:,None], size=(Nobs,Nsamp))
    return Mczs'''

def logq_add_err(logq, rho_obs, uncert, Nsamp):
    Nobs = logq.shape[0]
    sigma_logq = uncert['threshold_snr']/rho_obs*uncert['logq']
    bo = -logq/sigma_logq
    logqo = truncnorm.rvs(a=-np.inf*np.ones(Nobs), b=bo, loc=logq, scale=sigma_logq, size=Nobs)

    bs = []
    for i in range(Nobs):
        bs.append(np.repeat(-logqo[i]/sigma_logq[i], Nsamp))
    bs = np.array(bs)
    logqs = truncnorm.rvs(a=-np.inf*np.ones((Nobs,Nsamp)), b=bs, loc=logqo[:,None],
                          scale=sigma_logq[:,None], size=(Nobs,Nsamp))
    
    logqs_reweighted = []
    for i in range(Nobs):
        w = ss.norm.cdf(-logqo[i]/sigma_logq[i])/ss.norm.cdf(-logqs[i]/sigma_logq[i])
        logqs_reweighted.append(np.random.choice(logqs[i], size=Nsamp, p=w/np.sum(w), replace=True))
    logqs_reweighted = np.array(logqs_reweighted)

    return logqs_reweighted

def Theta_add_err(Theta, rho_obs, uncert, Nsamp):
    Nobs = Theta.shape[0]
    sigma_Theta = uncert['threshold_snr']/rho_obs*uncert['Theta']
    ao_T = -Theta/sigma_Theta
    bo_T = (1-Theta)/sigma_Theta
    Thetao = truncnorm.rvs(a=ao_T, b=bo_T, loc=Theta, scale=sigma_Theta, size=Nobs)

    as_T = []
    for i in range(Nobs):
        as_T.append(np.repeat(-Thetao[i]/sigma_Theta[i], Nsamp))
    as_T = np.array(as_T)
    
    bs_T = []
    for i in range(Nobs):
        bs_T.append(np.repeat((1-Thetao[i])/sigma_Theta[i], Nsamp))
    bs_T = np.array(bs_T)
    
    Thetas = truncnorm.rvs(a=as_T, b=bs_T, loc=Thetao[:,None],
                          scale=sigma_Theta[:,None], size=(Nobs,Nsamp))
    
    Thetas_reweighted = []
    for i in range(Nobs):
        w = (ss.norm.cdf((1-Thetao[i])/sigma_Theta[i]) - ss.norm.cdf(-Thetao[i]/sigma_Theta[i]))/(ss.norm.cdf((1-Thetas[i])/sigma_Theta[i]) - ss.norm.cdf(-Thetas[i]/sigma_Theta[i]))
        Thetas_reweighted.append(np.random.choice(Thetas[i], size=Nsamp, p=w/np.sum(w), replace=True))
    Thetas_reweighted = np.array(Thetas_reweighted)
    
    return Thetas_reweighted

def rhos_samples(rho_obs, Nsamp):
    Nobs = rho_obs.shape[0]
    rhos = np.random.normal(rho_obs[:,None], np.ones(Nobs)[:,None], size=(Nobs,Nsamp))
    return rhos

def dl_add_err(dl, Mczs, logqs, Thetas, rhos, uncert, Nsamp):
    Nobs = dl.shape[0]
    dfid = 1.
    m1zs = Mczs*(np.exp(logqs)**(-3./5.))*(1.+np.exp(logqs))**(1./5.)
    m2zs = Mczs*(np.exp(logqs)**(2./5.))*(1.+np.exp(logqs))**(1./5.)
    ds = snr_unit_dl_unit_theta(m1zs, m2zs)*Thetas/rhos
    
    return ds

def reweighted_samples(Mczs, logqs, Thetas, ds):
    Nobs = Mczs.shape[0]
    Nsamp = Mczs.shape[1]
    
    m1zs_reweighted = []
    m2zs_reweighted = []
    Thetas_reweighted = []
    ds_reweighted = []

    dfid = 1.
    m1zs = Mczs*(np.exp(logqs)**(-3./5.))*(1.+np.exp(logqs))**(1./5.)
    m2zs = Mczs*(np.exp(logqs)**(2./5.))*(1.+np.exp(logqs))**(1./5.)
    qs = m2zs/m1zs
    etas = (m1zs*m2zs)/(m1zs+m2zs)**2.
    
    for i in tqdm(range(Nobs)):
        wfishbach = (m1zs[i]-m2zs[i])*etas[i]**0.6/(m1zs[i]+m2zs[i])**2.
        wme = 1./((qs[i]-1./qs[i])*etas[i]**2.)
        w = wfishbach*wme*(Thetas[i]*snr_unit_dl_unit_theta(m1zs[i], m2zs[i]))/ds[i]**2
        m1zs_reweighted.append(np.random.choice(m1zs[i], size=Nsamp, p=w/np.sum(w), replace=True))
        m2zs_reweighted.append(np.random.choice(m2zs[i], size=Nsamp, p=w/np.sum(w), replace=True))
        Thetas_reweighted.append(np.random.choice(Thetas[i], size=Nsamp, p=w/np.sum(w), replace=True))
        ds_reweighted.append(np.random.choice(ds[i], size=Nsamp, p=w/np.sum(w), replace=True))
    m1zs_reweighted = np.array(m1zs_reweighted)
    m2zs_reweighted = np.array(m2zs_reweighted)
    Thetas_reweighted = np.array(Thetas_reweighted)
    ds_reweighted = np.array(ds_reweighted)
    
    return m1zs_reweighted, m2zs_reweighted, Thetas_reweighted, ds_reweighted

Mcz = (m1z*m2z)**0.6/(m1z+m2z)**0.2
q = m2z/m1z
logq = np.log(q)
Nsamp = 10000
Mczs = mcz_add_err(Mcz, rho_obs, uncert, Nsamp)
logqs = logq_add_err(logq, rho_obs, uncert, Nsamp)
Thetas = Theta_add_err(Theta, rho_obs, uncert, Nsamp)

rhos = rhos_samples(rho_obs, Nsamp)
ds = dl_add_err(dl, Mczs, logqs, Thetas, rhos, uncert, Nsamp)

m1zs_reweighted, m2zs_reweighted, Thetas_reweighted, ds_reweighted = reweighted_samples(Mczs, logqs, Thetas, ds)
zinterp = np.linspace(0, 12, 100000)
dlinterp = lvk_cosmology.luminosity_distance(zinterp).to(u.Gpc).value
zs_reweighted = np.interp(ds_reweighted, dlinterp, zinterp)

m1s_reweighted = m1zs_reweighted/(1.+zs_reweighted)
m2s_reweighted = m2zs_reweighted/(1.+zs_reweighted)

filename = "/mnt/home/sroy1/ceph/bumpcosmologygwtc3_Mc/data/Selection_Samples_With_Mock_PE.h5"

with h5py.File(filename, "w") as h5f:
    df_info = h5f.create_group("info")
    df_info.create_dataset("analysis_time_yr", data=T_yr)
    df_info.create_dataset("total_injections", data=N_selection)
    
    # Save the original DataFrame as a group under "injections"
    df_group = h5f.create_group("injections")
    for col in df.columns:
        # Save each column in the "injections" group
        df_group.create_dataset(col, data=df[col].values)

    # Save the two-dimensional arrays under the "injections-pe" key
    pe_group = h5f.create_group("injections-pe")
    pe_group.create_dataset("Source_Frame_m1", data=m1s_reweighted)
    pe_group.create_dataset("Source_Frame_m2", data=m2s_reweighted)
    pe_group.create_dataset("Redshift", data=zs_reweighted)

print("Data saved to h5py file.")