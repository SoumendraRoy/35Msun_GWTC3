## Preliminaries

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using AdvancedHMC
using ArviZ
using BumpCosmologyGWTC3
using CairoMakie
using Colors
using Cosmology
using DataFrames
using DimensionalData
using Distributions
using GaussianKDEs
using HDF5
using InferenceObjects
using JSON
using LaTeXStrings
using MCMCChainsStorage
using NCDatasets
using PairPlots
using PolyLog
using PopModels
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

include("plots.jl")

## Samples and Selection

## Set up paths
struct Paths
    gwtc_2_dir::String
    gwtc_3_dir::String
    evt_table_file::String
    o1o2o3_sensitivity_file::String
end

system = :rusty
if system == :rusty
    paths = Paths(
        "/mnt/home/ccalvk/ceph/GWTC-2.1", 
        "/mnt/home/ccalvk/ceph/GWTC-3", 
        "/mnt/home/ccalvk/ceph/gwosc-snapshots/snapshot-2023-11-04/GWTC/GWTC.json",
        "/mnt/home/ccalvk/ceph/sensitivity-estimates/o1+o2+o3_bbhpop_real+semianalytic-LIGO-T2100377-v2.hdf5"
    )
elseif system == :wmflaptop
    paths = Paths(
        "/Users/wfarr/Research/gwtc-2.1",
        "/Users/wfarr/Research/o3b_data/PE",
        "/Users/wfarr/Research/gwosc-snapshots/snapshot-2023-11-04/GWTC/GWTC.json",
        "/Users/wfarr/Research/o3b_data/O1O2O3-Sensitivity/o1+o2+o3_bbhpop_real+semianalytic-LIGO-T2100377-v2.hdf5"
    )
end

## Load PE
all_pe = load_pe(; gwtc_2_dir=paths.gwtc_2_dir, gwtc_3_dir=paths.gwtc_3_dir)
evt_table = load_event_table(paths.evt_table_file)
all_pe = join_pe_evt_tables(all_pe, evt_table)
all_pe[:, :prior_logwt_m1qzchie] = li_nocosmo_prior_logwt_m1qzchie(all_pe)
pe_table = far_cut(chirp_mass_cut(all_pe))

## Check number of events
n0p1, n0p5, n0p9 = length(groupby(far_cut(chirp_mass_cut(all_pe, thresh=0.1)), :gwname)), length(groupby(far_cut(chirp_mass_cut(all_pe, thresh=0.5)), :gwname)), length(groupby(far_cut(chirp_mass_cut(all_pe, thresh=0.9)), :gwname))
@info "N(p > 0.1) = $(n0p1); N(p > 0.5) = $(n0p5); N(p > 0.9) = $(n0p9)"

## m1-m2 KDE plot
f = m1m2_kde_plot(all_pe; p_cut=0.5, p_cut_eventlist=0.0, draw_cut_lines=true)
save(joinpath(@__DIR__, "..", "figures", "m1m2_kde1.pdf"), f)
f