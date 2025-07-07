const default_far_thresh = 1

function chirp_mass_cut(df; mc_min=default_mc_min, mc_max=default_mc_max, thresh=0.9)
    evt_df = groupby(df, :commonName)
    df = subset(evt_df, :chirp_mass_source => mcs -> sum((mcs .>= mc_min) .&& (mcs .< mc_max)) / length(mcs) > thresh, ungroup=true)
    df = df[df.chirp_mass_source .>= mc_min .&& df.chirp_mass_source .< mc_max, :]
end

function far_cut(df; far_thresh=default_far_thresh)
    evt_df = groupby(df, :commonName)
    df = subset(evt_df, :far => fars -> all(fars .< far_thresh), ungroup=true)
end

function cut_selection_table(df, df_pe; mc_min=default_mc_min, mc_max=default_mc_max, far_thresh=default_far_thresh, snr_thresh=10, thresh=0.9)
    # Get chirp mass samples from df_pe
    m1s = df_pe["Source_Frame_m1"]
    m2s = df_pe["Source_Frame_m2"]
    mc_samples = chirp_mass.(m1s, m2s)

    # Compute fraction in [mc_min, mc_max) for each event (column)
    keep = [sum((mc_samples[:, i] .>= mc_min) .& (mc_samples[:, i] .< mc_max)) / size(mc_samples, 1) > thresh
            for i in 1:size(mc_samples, 2)]

    # Filter rows of df accordingly
    filtered_df = DataFrame([row for (i, row) in enumerate(eachrow(df)) if keep[i]])

    return filtered_df
end

"""
function cut_selection_table(df; mc_min=default_mc_min, mc_max=default_mc_max, far_thresh=default_far_thresh, snr_thresh=10)
    subset(df, 
        [:mass1_source, :mass2_source] => ((m1, m2) -> @. chirp_mass(m1, m2) > mc_min && chirp_mass(m1, m2) < mc_max),
        [:name, :optimal_snr_net, :far_cwb, :far_gstlal, :far_mbta, :far_pycbc_bbh, :far_pycbc_hyperbank] => (ns, os, fars...) -> [(n == "o3" && any(fs .< far_thresh)) || ((n=="o1" || n=="o2") && (s > snr_thresh)) for (n, s, fs...) in zip(ns, os, fars...)]
    )
end
"""

function resample_selection(log_wt_fn, samples, log_pdraw, Ndraw, Nsamp)
    log_pdraw_new_unnorm = [log_wt_fn(s...) for s in samples]
    log_norm = logsumexp(log_pdraw_new_unnorm .- log_pdraw) - log(Ndraw)
    log_pdraw_new = log_pdraw_new_unnorm .- log_norm

    log_wts = log_pdraw_new .- log_pdraw
    wts = exp.(log_wts .- logsumexp(log_wts))
    inds = sample(1:length(samples), Weights(wts), Nsamp, replace=false)

    (samples[inds], log_pdraw_new[inds], Nsamp)
end