using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using ArgParse
using Distributed

s = ArgParseSettings()
@add_arg_table s begin
    "--Nchain"
        help="Number of chains to use"
        arg_type=Int
        default=4
    "--Nmcmc"
        help="Number of MCMC steps to take"
        arg_type=Int
        default=1000
    "--Nselection"
        help="Number of selection samples to use"
        arg_type=Int
        default=8192
    "--Nkde"
        help="Number of KDE samples to use"
        arg_type=Int
        default=128
    "--gwtc-2-dir"
        help="Path to GWTC-2.1 directory"
        default="/mnt/home/ccalvk/ceph/GWTC-2.1"
    "--gwtc-3-dir"
        help="Path to GWTC-3 directory"
        default="/mnt/home/ccalvk/ceph/GWTC-3"
    "--evt-table-file"
        help="Path to GWTC event table file"
        default="/mnt/home/ccalvk/ceph/gwosc-snapshots/snapshot-2023-11-04/GWTC/GWTC.json"
    "--o1o2o3-sensitivity-file"
        help="Path to O1+O2+O3 sensitivity file"
        default="/mnt/home/ccalvk/ceph/sensitivity-estimates/o1+o2+o3_bbhpop_real+semianalytic-LIGO-T2100377-v2.hdf5"
end

parsed_args = parse_args(s)

Nchain = parsed_args["Nchain"]
Nmcmc = parsed_args["Nmcmc"]
Nselection = parsed_args["Nselection"]
Nkde = parsed_args["Nkde"]

addprocs(Nchain)

@everywhere begin 
    using AdvancedHMC
    using ArviZ
    using BumpCosmologyGWTC3
    using CairoMakie
    using Colors
    using Cosmology
    using DataFrames
    using Distributions
    using GaussianKDEs
    using HDF5
    using JSON
    using LaTeXStrings
    using MCMCChainsStorage
    using NCDatasets
    using Printf
    using ProgressLogging
    using Random
    using ReverseDiff
    using StatsBase
    using SpecialFunctions
    using StatsFuns
    using Tapir
    using Trapz
    using Turing
    using Unitful
    using UnitfulAstro
    using UnitfulChainRules
    using Zygote
end

gwtc_2_dir = parsed_args["gwtc-2-dir"]
gwtc_3_dir = parsed_args["gwtc-3-dir"]
evt_table_file = parsed_args["evt-table-file"]
o1o2o3_sensitivity_file = parsed_args["o1o2o3-sensitivity-file"]

all_pe = load_pe( ; gwtc_2_dir=gwtc_2_dir, gwtc_3_dir=gwtc_3_dir)
evt_table = load_event_table(evt_table_file)
all_pe = join_pe_evt_tables(all_pe, evt_table)
pe_table = far_cut(chirp_mass_cut(all_pe))

evt_names = [String(evt[1,:commonName]) for evt in groupby(pe_table, :commonName, sort=true)]

Nevt = length(groupby(pe_table, :commonName, sort=true))
@info "Analyzing $(Nevt) events"

sel_table, T, Nsel = load_selection(o1o2o3_sensitivity_file)
det_table = cut_selection_table(sel_table)

# Fixup Nselection if it's too big:
if Nselection > size(det_table, 1)
    @warn "Reducing Nselection from $Nselection to $(size(det_table, 1))"
    Nselection = size(det_table, 1)
end

zmax = 1 + max(maximum(det_table.redshift), maximum(pe_table.redshift))
zs_interp = expm1.(range(log(1), log(1+zmax), length=1024))
@info "zmax = $(round(zmax, digits=2))"

m_grid = collect(10.0:0.25:100.0)

sel_samples, log_sel_pdraw = sel_dataframe_to_cosmo_samples_array(det_table, Nselection; rng=Random.Xoshiro(168733815688665017))
log_sel_pdraw = @. log_sel_pdraw - log(T) # per year
chie_bw_sel = chie_bandwidth(sel_samples)

evt_kdes = pe_dataframe_to_cosmo_evt_kdes(pe_table, Nkde; rng=Random.Xoshiro(4063861647701281830))

# Using the KDE model; here the number of parameters is large (in the 100s), so
# we have to use a backward autodiff module, and the best I've found so far is
# Tapir.jl (seems really cool).  Because of the large number of parameters it's
# not feasible to use a dense metric, either, so we use the default diagonal
# metric.
model = pop_model_cosmo_kde(evt_kdes, sel_samples, log_sel_pdraw, Nsel, chie_bw_sel, m_grid, zs_interp)
trace = sample(model, Turing.NUTS(Nmcmc, 0.65; adtype=AutoTapir(safe_mode=false)), MCMCDistributed(), Nmcmc, Nchain)
genq = generated_quantities(model, trace)
trace = append_generated_quantities(trace, genq)
trace = from_mcmcchains(trace; coords=Dict(:event=>evt_names), dims=Dict(:mcs => [:event],
                                                                         :qs => [:event],
                                                                         :chieff_units => [:event],
                                                                         :zs => [:event],
                                                                         :chi_effs => [:event],
                                                                         :m1s => [:event]))

chainfilename = "chains_cosmo_kde.nc"
to_netcdf(trace, joinpath(@__DIR__, "..", "data", chainfilename))

@info "4*N = $(4*Nevt)"
@info "Minimum Neff_sel = $(round(minimum(trace.posterior.Neff_sel), digits=2))"