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
        default=32768
    "--p-cut"
        help="minimal credible-level of posterior to include in catalog"
        arg_type=Float64
        default=0.5
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
        default="/mnt/home/sroy1/ceph/bumpcosmologygwtc3_Mc/data/Selection_Samples_With_Mock_PE.h5"
        # default="/mnt/home/ccalvk/ceph/sensitivity-estimates/o1+o2+o3_bbhpop_real+semianalytic-LIGO-T2100377-v2.hdf5"
end

parsed_args = parse_args(s)

p_cut = parsed_args["p-cut"]
@assert 0 <= p_cut <= 1

Nchain = parsed_args["Nchain"]
Nmcmc = parsed_args["Nmcmc"]
Nselection = parsed_args["Nselection"]

addprocs(Nchain)

@everywhere begin 
    using AdvancedHMC
    using ArviZ
    using BumpCosmologyGWTC3
    using CairoMakie
    using Colors
    using Cosmology
    using DataFrames
    using DelimitedFiles
    using Distributions
    using GaussianKDEs
    using HDF5
    using InferenceObjects
    using JSON
    using LaTeXStrings
    using MCMCChainsStorage
    using Mooncake
    using NCDatasets
    using Printf
    using ProgressLogging
    using Random
    using StatsBase
    using SpecialFunctions
    using StatsFuns
    using Trapz
    using Turing
    using Unitful
    using UnitfulAstro
    using UnitfulChainRules
end

gwtc_2_dir = parsed_args["gwtc-2-dir"]
gwtc_3_dir = parsed_args["gwtc-3-dir"]
evt_table_file = parsed_args["evt-table-file"]
o1o2o3_sensitivity_file = parsed_args["o1o2o3-sensitivity-file"]

all_pe = load_pe( ; gwtc_2_dir=gwtc_2_dir, gwtc_3_dir=gwtc_3_dir)
evt_table = load_event_table(evt_table_file)
all_pe = join_pe_evt_tables(all_pe, evt_table)
all_pe[:, :prior_logwt_m1qzchie] = li_nocosmo_prior_logwt_m1qzchie(all_pe)
pe_table = far_cut(chirp_mass_cut(all_pe; thresh=p_cut))

evt_names = [String(evt[1,:commonName]) for evt in groupby(pe_table, :commonName, sort=true)]

if parsed_args["p-cut"] == 0.9
    npost_file = joinpath(@__DIR__, "..", "data", "nposts.csv")
else
    npost_file = joinpath(@__DIR__, "..", "data", "nposts_$(round(parsed_args["p-cut"], digits=2)).csv")
end

# Loop until we satisfy Nposts or cannot increase any more.
while true
    reduced_nposts = Set()
    try
        global Nposts = readdlm(npost_file, ',', Int)
    catch e
        @warn "Could not read Nposts from $(npost_file): $e"
        @warn "Using default of 16 samples per event"
        global Nposts = 16*ones(Int, length(evt_names))
    end
    Nposts_old = Nposts
    global Nposts = [min(np, size(evt,1)) for (np, evt) in zip(Nposts, groupby(pe_table, :commonName, sort=true))]
    for (npo, np, en) in zip(Nposts_old, Nposts, evt_names)
        if npo != np
            @info "Reducing Nposts for event $en from $npo to $np"
            push!(reduced_nposts, en)
        end
    end

    Nevt = length(groupby(pe_table, :commonName, sort=true))
    @info "Analyzing $(Nevt) events"

    sel_table, sel_table_pe, T, Nsel = load_selection(o1o2o3_sensitivity_file)
    det_table = cut_selection_table(sel_table, sel_table_pe; thresh=p_cut)

    """
    sel_table, T, Nsel = load_selection(o1o2o3_sensitivity_file)
    det_table = cut_selection_table(sel_table)
    """

    # Fixup Nselection if it's too big:
    if Nselection > size(det_table, 1)
        @warn "Reducing Nselection from $Nselection to $(size(det_table, 1))"
        global Nselection = size(det_table, 1)
    end

    zs_interp = expm1.(range(log(1), log(1+10), length=1024))
    m_grid = collect(10.0:0.25:100.0)

    sel_samples, log_sel_pdraw = sel_dataframe_to_samples_array(det_table, Nselection; rng=Random.Xoshiro(168733815688665017))
    log_sel_pdraw = @. log_sel_pdraw - log(T) # per year
    chie_bw_sel = chie_bandwidth(sel_samples)

    evt_samples, log_prior_wts = pe_dataframe_to_samples_array(pe_table, Nposts; rng=Random.Xoshiro(4063861647701281830))
    chi_eff_bws = [chie_bandwidth(s) for s in evt_samples]

    # Using the samples model; here the number of parameters are small, so we
    # get better performance by using a dense metric for the HMC sampler
    model = pop_model_samples(evt_samples, log_prior_wts, chi_eff_bws, sel_samples, log_sel_pdraw, Nsel, chie_bw_sel, m_grid, zs_interp)
    trace = sample(model, Turing.NUTS(Nmcmc, 0.65; metricT=AdvancedHMC.DenseEuclideanMetric, adtype=AutoMooncake(config=nothing)), MCMCDistributed(), Nmcmc, Nchain)
    genq = generated_quantities(model, trace)
    trace = append_generated_quantities(trace, genq)
    trace = from_mcmcchains(
        trace,
        coords=Dict(:event => evt_names),
        dims=Dict(
            :Neff_samps => (:event,),
            :m1s => (:event,),
            :m2s => (:event,),
            :qs => (:event,),
            :chi_effs => (:event,),
            :zs => (:event,)
        )
    )

    if p_cut == 0.9
        p_cut_str = ""
    else
        p_cut_str = "_$(round(p_cut, digits=2))"
    end

    chainfilename = "chains" * p_cut_str * ".nc"

    to_netcdf(trace, joinpath(@__DIR__, "..", "data", chainfilename))

    @info "4*N = $(4*Nevt)"
    @info "Minimum Neff_sel = $(round(minimum(trace.posterior.Neff_sel), digits=2))"

    # Using the samples model: we want to report the minimum Neff_samps for each
    # event, and adjust Npost to achieve a reasonable number (>> 1 is fine).
    min_Neff_samps = dropdims(minimum(trace.posterior.Neff_samps, dims=(:chain, :draw)); dims=(:chain, :draw))
    @info "Minimum Neff_samps = $(round(minimum(min_Neff_samps), digits=2)) for event $(groupby(pe_table, :commonName, sort=true)[argmin(min_Neff_samps)][1,:commonName])"

    function up_np(np, neff, n_desired=8)
        if neff < n_desired/2
            return round(Int, 2*np)
        elseif neff > n_desired*2
            return round(Int, np/2)
        else
            return np
        end
    end
    new_Nposts = [up_np(ne, new_ne) for (ne, new_ne) in zip(Nposts, min_Neff_samps)]

    all_updates_in_reduced_npost_set = true
    if any(new_Nposts .!= Nposts)
        for (np_new, np, name) in zip(new_Nposts, Nposts, evt_names)
            if np_new != np
                @info "Updating Nposts for event $name from $np to $np_new"
                if !(name in reduced_nposts)
                    all_updates_in_reduced_npost_set = false
                end
            end
        end

        @info "Updated Nposts.  Writing to $(npost_file) = $(new_Nposts)"
        writedlm(npost_file, new_Nposts, ',')

        if all_updates_in_reduced_npost_set
            @info "All Npost updates truncated by pe sample size; stopping, but may not be converged; check neff_samps!"
            break
        end
    else
        @info "No updates to Nposts"
        break # End the while loop
    end
    @info "Looping again with updated Nposts!"
end
