square(x) = x*x

raw"""
    log_dNdm(mbh, alpha, mtr, mbhmax, sigma)

Returns the log of the black hole mass function at `mbh` given parameters
describing the initial-final mass relation.

The black hole mass function here is derived by assuming a power law mass
function for the initial mass:

``\frac{\mathrm{d}N}{\mathrm{d}m} = \left( \frac{m}{m_\mathrm{max}}
\right)^{-\alpha}``

and that the average black hole mass is a piecewise function of the initial
mass, given by 

``m_\mathrm{BH}\left( m \right) = \begin{cases} m & m < m_\mathrm{tr} \\
    m_\mathrm{BH,max} - c \left( m - m_\mathrm{max} \right)^2 & m \geq m_\mathrm{tr}
\end{cases}``

This is a linear relationship for ``m < m_\mathrm{tr}``, and then smoothly
transitions to a quadratic with a maximum black hole mass of
``m_\mathrm{BH,max}`` achieved at ``m = m_\mathrm{max} = 2 m_\mathrm{BH,max} -
m_\mathrm{tr}`` (the smoothness condition sets ``c = 1/\left( 4 \left(
m_\mathrm{BH,max} - m_\mathhrm{tr}\right)\right)``).

Note that the function is normalized so that the power-law initial mass function
is unity at the initial mass that corresponds to the maximum expected black hole
mass.
"""
function log_dNdm(mbh, alpha, mtr, mbhmax, sigma)
    # mbh = mbhmax - c * (m - m_max)^2
    c = 1/(4*(mbhmax-mtr))
    m_max = 2*mbhmax - mtr

    if mbh < mbhmax
        a = mbhmax - mbh
        b = c
        x = a*a/(4*sigma*sigma)
        log_wt = log(sqrt(a*pi/(2*b))/(4*sigma)) + log(besselix(-0.25, x) + besselix(0.25, x))

        if mbh < mtr
            mlow = mbh
            mhigh = m_max + sqrt((mbhmax - mbh)/c)
            log_wt_low = 0.0
            log_wt_high = log_wt
        else
            d = sqrt((mbhmax - mbh)/c)
            mlow = m_max - d
            mhigh = m_max + d
            log_wt_low = log_wt
            log_wt_high = log_wt
        end
    else
        a = mbh-mbhmax
        b = c
        x = a*a/(4*sigma*sigma)

        log_wt = log(sqrt(a/(b*pi))/(4*sigma)) - 2*x + log(besselkx(0.25, x))

        mlow = m_max
        mhigh = m_max
        log_wt_low = log_wt
        log_wt_high = log_wt
    end

    logplow = -alpha*log(mlow / m_max)
    logphigh = -alpha*log(mhigh / m_max)

    logaddexp(log_wt_low + logplow, log_wt_high + logphigh)
end

function _c(mtr, mbhmax)
    1/(4*(mbhmax-mtr))
end

function _mini_max(mtr, mbhmax)
    2*mbhmax - mtr
end

function mrem_of_mini(mini, mtr, mbhmax)
    if mini < mtr
        return mini
    else
        c = _c(mtr, mbhmax)
        m_max = _mini_max(mtr, mbhmax)
    
        return mbhmax - c*(mini - m_max)^2
    end
end

function mini_left_of_mrem(mrem, mtr, mbhmax)
    if mrem > mbhmax
        return _mini_max(mtr, mbhmax)
    elseif mrem < mtr
        return mrem
    else
        c = _c(mtr, mbhmax)
        m_max = _mini_max(mtr, mbhmax)
        d = sqrt((mbhmax - mrem)/c)
    
        return m_max - d
    end
end

function mini_right_of_mrem(mrem, mtr, mbhmax)
    if mrem > mbhmax
        return _mini_max(mtr, mbhmax)
    else
        c = _c(mtr, mbhmax)
        m_max = _mini_max(mtr, mbhmax)
        d = sqrt((mbhmax - mrem)/c)
    
        return m_max + d
    end
end

function log_trapz(xs, log_ys)
    log_dx = log.(xs[2:end] .- xs[1:end-1])

    log_wts = log(0.5) .+ log_dx .+ logaddexp.(log_ys[1:end-1], log_ys[2:end])
    logsumexp(log_wts)
end

function mini_integral_log(mrem, alpha, mtr, mbhmax, sigma)
    m_max = _mini_max(mtr, mbhmax)

    mrem_low = max(0.01, mrem - 5*sigma) # Ensure positive
    mrem_high = mrem + 5*sigma

    if mrem_high < mbhmax
        # Then two distinct regions.

        mill = mini_left_of_mrem(mrem_low, mtr, mbhmax)
        milh = mini_left_of_mrem(mrem_high, mtr, mbhmax)
        mi = range(mill, stop=milh, length=128)
        log_ys = [-alpha*log(m / m_max) + logpdf(Normal(mrem_of_mini(m, mtr, mbhmax), sigma), mrem) for m in mi]
        log_i1 = log_trapz(mi, log_ys)

        mirl = mini_right_of_mrem(mrem_high, mtr, mbhmax)
        mirh = mini_right_of_mrem(mrem_low, mtr, mbhmax)
        mi = range(mirl, stop=mirh, length=128)
        log_ys = [-alpha*log(m / m_max) + logpdf(Normal(mrem_of_mini(m, mtr, mbhmax), sigma), mrem) for m in mi]
        log_i2 = log_trapz(mi, log_ys)

        logaddexp(log_i1, log_i2)
    elseif mrem_low < mbhmax
        # One regions
        ml = mini_left_of_mrem(mrem_low, mtr, mbhmax)
        mr = mini_right_of_mrem(mrem_low, mtr, mbhmax)
        mi = range(ml, stop=mr, length=128)
        log_ys = [-alpha*log(m / m_max) + logpdf(Normal(mrem_of_mini(m, mtr, mbhmax), sigma), mrem) for m in mi]
        log_trapz(mi, log_ys)
    else
        # Both are above.
        mr = max(0.01, mbhmax - 5*sigma)
        ml = mini_left_of_mrem(mr, mtr, mbhmax)
        mr = mini_right_of_mrem(mr, mtr, mbhmax)
        mi = range(ml, stop=mr, length=128)
        log_ys = [-alpha*log(m / m_max) + logpdf(Normal(mrem_of_mini(m, mtr, mbhmax), sigma), mrem) for m in mi]
        log_trapz(mi, log_ys)
    end
end

function make_log_dNdm_gridded_efficient(alpha, mtr, mbhmax, sigma; mmin=10.0, mmax=75.0)
    dm = 10*sigma / 128
    ms = collect(mmin:dm:mmax)

    log_dNdms = [mini_integral_log(m, alpha, mtr, mbhmax, sigma) for m in ms]

    function log_dNdm_gridded(m)
        interp1d(m, ms, log_dNdms)
    end
    log_dNdm_gridded
end

function make_log_dNdm_gridded(mgrid, alpha, mtr, mbhmax, sigma; mmin=10.0, mmax=100.0)
    ms = mgrid

    m_max = _mini_max(mtr, mbhmax)

    log_dNdms = [log_trapz(ms, [-alpha*log(mi / m_max) + logpdf(Normal(mrem_of_mini(mi, mtr, mbhmax), sigma), mr) for mi in ms]) for mr in ms]

    function log_dNdm_gridded(m)
        interp1d(m, ms, log_dNdms)
    end
    log_dNdm_gridded
end

raw"""
    chieff_interp(q, l, h)

Returns the mean or s.d. of the effective spin distribution at a given mass
ratio.

`l` is the mean or s.d. at ``q = 0`` and `h` is the mean or s.d. at ``q = 1``.
`q` is the mass ratio in question.  (This is just simple linear interpolation,
but written in a way that ensures that `l` and `h` are the extremal values of
the interpolation.)
"""
function chieff_interp(q, l, h)
    q*h + (1-q)*l
end

raw"""
    log_dNdq(q, beta)

The mass ratio distribution, given by 
``\frac{\mathrm{d}N}{\mathrm{d}q} = \left( \frac{1 + q}{2} \right)^\beta``

Note the normalization, where the value is `1` at `q = 1`.
"""
function log_dNdq(q, beta)
    beta*(log1p(q) - log(2))
end

raw"""
    log_mdsfr(z, lambda, zp, kappa)

Returns the log of the (un-normalized) Madau-Dickinson SFR:

``\frac{\left( 1 + z \right)^\lambda}{1 + \left(\frac{1 + z}{1 + z_p} \right)^\kappa}``
"""
function log_mdsfr(z, lambda, zp, kappa)
    lambda * log1p(z) - log1p(((1+z)/(1+zp))^kappa)
end

function make_log_dNdm_hm(log_r, m0, alpha)
    function log_dN(m)
        log_r - alpha*log(m/m0) - log1p(exp(-(m-m0))) + log(2)
    end

    log_dN
end

function make_combined_log_dNdm(log_rhm, alphahm, alpha, mtr, mbhmax, sigma; mgrid=collect(10:0.25:100), mnorm=35.0)
    log_dNdm = make_log_dNdm_gridded(mgrid, alpha, mtr, mbhmax, sigma)
    log_dNdm_norm = log(mnorm) + log_dNdm(mnorm)

    log_r = log_rhm + log_dNdm(mbhmax) - log_dNdm_norm
    log_dNdm_hm = make_log_dNdm_hm(log_r, mbhmax, alphahm)

    function log_dN(m)
        logaddexp(log_dNdm(m) - log_dNdm_norm, log_dNdm_hm(m))
    end

    log_dN
end

function make_log_dNdm1dqdchiedVdt(log_rhm, alphahm, alpha, mtr, mbhmax, sigma, beta, mul, muh, sigmal, sigmah, lambda, zp, kappa; mgrid=collect(10:0.25:100), mnorm=35.0, qnorm=1.0, znorm=0.0)
    log_dNdm_total = make_combined_log_dNdm(log_rhm, alphahm, alpha, mtr, mbhmax, sigma; mgrid=mgrid, mnorm=mnorm)
    
    # These should be the values of the corresponding terms in the model at (mnorm, qnorm, chi_eff_norm, znorm)
    log_znorm = log_mdsfr(znorm, lambda, zp, kappa)
    log_qnorm = log_dNdq(qnorm, beta)

    function log_dNdm1dqdchiedVdt(m, q, chi_eff, z, chie_bw; ignore_chi_eff=false)
        m2 = q*m
        mu_chi = chieff_interp(q, mul, muh)
        sigma_chi = chieff_interp(q, sigmal, sigmah)

        sigma_chi_marg = sqrt(sigma_chi^2 + chie_bw^2)

        log_m1_pop = log_dNdm_total(m)
        log_q_pop = log_dNdm_total(m2) + log(m)
        log_pair_pop = log_dNdq(q, beta) - log_qnorm
        if ignore_chi_eff
            log_chieff_pop = 0.0
        else
            log_chieff_pop = logpdf(Normal(mu_chi, sigma_chi_marg), chi_eff)
        end
        log_VT_pop = log_mdsfr(z, lambda, zp, kappa) - log_znorm
                
        log_m1_pop + log_q_pop + log_pair_pop + log_chieff_pop + log_VT_pop
    end

    return log_dNdm1dqdchiedVdt
end

function chie_bandwidth(samples)
    pts = hcat(samples...)'
    C = cov(pts)
    B = C / length(samples)^(1/4)

    F_chie_chie = (B \ [0.0, 0.0, 1.0, 0.0])[3]
    1/sqrt(F_chie_chie)
end

function pe_dataframe_to_samples_array(df, Nposts; rng=Random.default_rng())
    evts = groupby(df, :commonName, sort=true)
    shuffled_evts = [shuffle(rng, evt) for evt in evts]
    pe_samples = [[[evt[i, :mass_1_source], evt[i, :mass_ratio], evt[i, :chi_eff], evt[i, :redshift]] for i in 1:np] for (np, evt) in zip(Nposts, shuffled_evts)]
    log_pe_wts = [vec(evt[1:np, :prior_logwt_m1qzchie]) for (np, evt) in zip(Nposts, shuffled_evts)]
    (pe_samples, log_pe_wts)
end

function evt_dataframe_to_kde(df, Nkde; rng=Random.default_rng())
    df = shuffle(rng, df)
    log_wts = .- li_nocosmo_prior_logwt_m1qzchie(df)
    wts = exp.(log_wts .- logsumexp(log_wts))
    inds = sample(1:size(df, 1), Weights(wts), 2*Nkde)
    df_sel = df[inds, :]
    pts = Array(df_sel[1:2*Nkde, [:mass_1_source, :mass_ratio, :chi_eff, :redshift]])'
    bw_opt_kde(pts[:, 1:Nkde], pts[:, Nkde+1:end])
end

function pe_dataframe_to_evt_kdes(df, Nkde; rng=Random.default_rng())
    evts = groupby(df, :commonName, sort=true)
    [evt_dataframe_to_kde(evt, Nkde; rng=rng) for evt in evts]
end

function sel_dataframe_to_samples_array(df, Nsamp=1024; rng=Random.default_rng())
    shuffled_df = shuffle(rng, df)
    sel_samples = [[shuffled_df[i, :mass1_source], shuffled_df[i, :q], shuffled_df[i, :chi_eff], shuffled_df[i, :redshift]] for i in 1:Nsamp]
    log_sel_pdraw = log.(shuffled_df[1:Nsamp, :sampling_pdf_qchieff])
    (sel_samples, log_sel_pdraw)
end

@model function pop_model_samples(evt_samples, log_prior_wts, evt_chie_bws, sel_samples, log_sel_pdraw, Ndraw, chie_bw_sel, m_grid, zs_interp)
    nevt = length(evt_samples)
    dh = dH(h_lvk)
    dcs_interp = dc_over_dh(zs_interp, Ω_M_lvk)
    dvdz_interp = dvdz_over_vh(zs_interp, Ω_M_lvk, dcs_interp)

    log_dV_interp = 3*log(dh) .+ log.(dvdz_interp) .- log1p.(zs_interp)

    # Priors
    log_rhm ~ Uniform(log(0.01), log(0.5))
    alphahm ~ Uniform(0, 6)

    alpha ~ Uniform(-4, 4)
    mtr ~ Uniform(25, 45)
    mbhmax ~ Uniform(mtr + 1, mtr + 10)
    sigma ~ Uniform(1, 10)

    beta ~ Uniform(-2.5, 10)

    mul ~ Uniform(-1, 1)
    muh ~ Uniform(-1, 1)

    sigmal ~ Uniform(0.0, 2)
    sigmah ~ Uniform(0.0, 2)

    lambda ~ Uniform(2, 10)
    zp ~ Uniform(1, 4)
    kappa ~ Uniform(3, 8)

    # Pop density
    log_dNdm1dqdchiedVdt = make_log_dNdm1dqdchiedVdt(log_rhm, alphahm, alpha, mtr, mbhmax, sigma, beta, mul, muh, sigmal, sigmah, lambda, zp, kappa; mgrid=m_grid)

    function log_pop_density(theta)
        m1, q, chi_eff, z, chie_bw = theta
        log_dNdm1dqdchiedVdt(m1, q, chi_eff, z, chie_bw) + interp1d(z, zs_interp, log_dV_interp)
    end

    thetas = map(evt_samples, evt_chie_bws) do samples, chie_bw
        map(samples) do s
            m1, q, chi_eff, z = s
            [m1, q, chi_eff, z, chie_bw]
        end
    end

    thetas_sel = map(sel_samples) do s
        m1, q, chi_eff, z = s
        [m1, q, chi_eff, z, chie_bw_sel]
    end

    log_likelihood_sum, log_normalization_sum, model_genq = pop_model_body(log_pop_density, thetas, log_prior_wts, thetas_sel, log_sel_pdraw, Ndraw)
    Turing.@addlogprob! log_likelihood_sum
    Turing.@addlogprob! log_normalization_sum

    alpha_chi_eff = muh - mul # Derivative of mean chi_eff wrt q
    beta_chi_eff = sigmah - sigmal # Derivative of s.d. chi_eff wrt q

    # Now we draw samples
    m1s = map(model_genq.thetas_popwt) do tp
        tp[1]
    end
    qs = map(model_genq.thetas_popwt) do tp
        tp[2]
    end
    chi_effs = map(model_genq.thetas_popwt, evt_chie_bws) do tp, chie_bw
        q = tp[2]
        chi_e_draw = tp[3]

        mu_pop = chieff_interp(q, mul, muh)
        sigma_pop = chieff_interp(q, sigmal, sigmah)

        mu = (mu_pop*chie_bw^2 + chi_e_draw*sigma_pop^2) / (chie_bw^2 + sigma_pop^2)
        sigma = 1/sqrt(1/chie_bw^2 + 1/sigma_pop^2)
        rand(Normal(mu, sigma))
    end
    zs = map(model_genq.thetas_popwt) do tp
        tp[4]
    end
    m2s = m1s .* qs

    m1_draw = model_genq.theta_draw[1]
    q_draw = model_genq.theta_draw[2]
    z_draw = model_genq.theta_draw[4]

    chie_draw = model_genq.theta_draw[3]
    mu_pop = chieff_interp(q_draw, mul, muh)
    sigma_pop = chieff_interp(q_draw, sigmal, sigmah)

    mu = (chie_draw*sigma_pop^2 + mu_pop*chie_bw_sel^2) / (sigma_pop^2 + chie_bw_sel^2)
    sigma = 1/sqrt(1/sigma_pop^2 + 1/chie_bw_sel^2)
    chi_eff_draw = rand(Normal(mu, sigma))

    (Neff_sel = model_genq.Neff_sel, R = model_genq.R, Neff_samps = model_genq.Neff_samps, rhm = exp(log_rhm), alpha_chi_eff=alpha_chi_eff, beta_chi_eff=beta_chi_eff, m1s = m1s, m2s = m2s, qs = qs, chi_effs = chi_effs, zs = zs, m1_draw = m1_draw, m2_draw = m1_draw * q_draw, q_draw = q_draw, chi_eff_draw = chi_eff_draw, z_draw = z_draw)
end

## Now for the cosmologically-varying version.
## Here the canonical variables are going to be: 
## Detector-frame m1, q, chi_eff, and luminosity-distance.
## KDEs and draw probability will be in this space.
function pe_dataframe_to_cosmo_samples_array(df, Nposts; rng=Random.default_rng())
    evts = groupby(df, :commonName, sort=true)
    shuffled_evts = [shuffle(rng, evt) for evt in evts]
    pe_samples = [[[evt[i, :mass_1], evt[i, :mass_ratio], evt[i, :chi_eff], evt[i, :luminosity_distance]/1000] for i in 1:np] for (np, evt) in zip(Nposts, shuffled_evts)]
    log_pe_wts = [vec(evt[1:np, :prior_logwt_m1dqdlchie]) for (np, evt) in zip(Nposts, shuffled_evts)]
    (pe_samples, log_pe_wts)
end

function evt_dataframe_to_cosmo_kde(df, Nkde; rng=Random.default_rng())
    df = shuffle(rng, df)
    log_wts = .- li_nocosmo_prior_logwt_m1dqdlchie(df)
    wts = exp.(log_wts .- logsumexp(log_wts))
    inds = sample(1:size(df, 1), Weights(wts), 2*Nkde)
    df_sel = df[inds, :]
    pts = Array(df_sel[1:2*Nkde, [:mass_1, :mass_ratio, :chi_eff, :luminosity_distance]])'
    pts[4, :] ./= 1000 # Our distances are in Gpc
    bw_opt_kde(pts[:, 1:Nkde], pts[:, Nkde+1:end])
end

function pe_dataframe_to_cosmo_evt_kdes(df, Nkde; rng=Random.default_rng())
    evts = groupby(df, :commonName, sort=true)
    [evt_dataframe_to_cosmo_kde(evt, Nkde; rng=rng) for evt in evts]
end

function sel_dataframe_to_cosmo_samples_array(df, Nsamp=1024; rng=Random.default_rng())
    shuffled_df = shuffle(rng, df)
    sel_samples = [[shuffled_df[i, :mass_1], shuffled_df[i, :q], shuffled_df[i, :chi_eff], shuffled_df[i, :luminosity_distance]] for i in 1:Nsamp]
    log_sel_pdraw = log.(shuffled_df[1:Nsamp, :sampling_pdf_m1dqdlchieff])
    (sel_samples, log_sel_pdraw)
end

@model function pop_model_cosmo_kde(evt_kdes, sel_samples, log_sel_pdraw, Ndraw, chie_bw_sel, m_grid, zs_interp)
    nevt = length(evt_kdes)

    # Priors
    h ~ Uniform(0.5, 0.9)
    Ω_M ~ Uniform(0.1, 0.5)
    Ω_Λ = 1 - Ω_M

    alpha ~ Uniform(-4, 4)
    mtr ~ Uniform(25, 45)
    mbhmax ~ Uniform(mtr + 1, mtr + 10)
    sigma ~ Uniform(1, 10)

    beta ~ Uniform(-2.5, 10)

    mul ~ Uniform(-1, 1)
    muh ~ Uniform(-1, 1)

    sigmal ~ Uniform(0.0, 2)
    sigmah ~ Uniform(0.0, 2)

    lambda ~ Uniform(2, 10)
    zp ~ Uniform(1, 4)
    kappa ~ Uniform(3, 8)

    R_scaled ~ Normal(0,1)

    # Per-system parameters.  We sample in chirp mass and q because these are
    # less correlated than m1 and m2 or m1 and q.  The sampler uses a
    # unit-Normal prior on the `chieff_units` variable, which is subsequently
    # transformed chi_eff = mu(q) + chi_eff_units * sigma(q), ensuring (1) the
    # sampler sees a constant scale in chieff_unit no matter if sigma(q) -> 0
    # and (2) the *induced* prior distribution on chi_eff is N(mu(q), sigma(q)).
    #
    # Variables here are given otherwise flat distributions (i.e. constant
    # density) because we will accumulate the population density later.  There
    # is an additional Jacobian factor for each system to transform from mc to
    # m1 (in which the population density is expressed).
    #
    # We now avoid constraining the population in the source-frame chirp mass,
    # since this will introduce an artificial edge that will provide false
    # cosmological constraints.
    #
    # The fact that the population was selected based on the source-frame chirp
    # mass *assuming a LVK standard cosmology* will already be reflected in the
    # selection function samples.
    mcs ~ filldist(Uniform(default_mc_min/2, 2*default_mc_max), nevt)
    qs ~ filldist(Uniform(0, 1), nevt)
    chieff_units ~ filldist(Normal(0,1), nevt)
    zs ~ filldist(Uniform(0, zs_interp[end]), nevt)

    # Handle chi_eff separately because we want to ensure that the sampler
    # always sees N(0,1); this transformation takes the sample variable N(0,1)
    # to N(mu(q), sigma(q)), so we use `ignore_chi_eff=true` in the population
    # model.
    chieffs = chieff_interp.(qs, mul, muh) .+ chieff_units .* chieff_interp.(qs, sigmal, sigmah)

    # Sampling in mc, but have likelihoods and population in terms of m1, so need d(m1) / d(mc) Jacobian.
    m1s = mcs .* (1 .+ qs).^(1/5) ./ qs.^(3/5)
    Turing.@addlogprob! sum((1/5) .* log1p.(qs) .- (3/5) .* log.(qs))

    dh = dH(h)
    dcs_interp = dc_over_dh(zs_interp, Ω_M)
    dls_interp = dl_over_dh(zs_interp, dcs_interp)
    dvdz_interp = dvdz_over_vh(zs_interp, Ω_M, dcs_interp)

    # Pop density
    log_dNdm1dqdchiedVdt = make_log_dNdm1dqdchiedVdt(alpha, mtr, mbhmax, sigma, beta, mul, muh, sigmal, sigmah, lambda, zp, kappa; mgrid=m_grid)
    log_likes = map(evt_kdes, m1s, qs, chieffs, zs) do kde, m, q, chi_eff, z
        md = m * (1+z)
        dl = dh * interp1d(z, zs_interp, dls_interp)
        logpdf(kde, [md, q, chi_eff, dl]) + log_dNdm1dqdchiedVdt(m, q, chi_eff, z, 0.0; ignore_chi_eff=true) + 3*log(dh) + log(interp1d(z, zs_interp, dvdz_interp)) - log1p(z)
    end
    Turing.@addlogprob! sum(log_likes)

    log_sel_wts = map(sel_samples, log_sel_pdraw) do samples, log_pdraw
        m1d, q, chi_eff, dl = samples
        z = interp1d(dl/dh, dls_interp, zs_interp)
        m1 = m1d / (1+z)
        dc = dl / (1+z)

        ddl_dz = dc + (1+z)*dh/ez(z, Ω_M)

        log_dN_m1z = log_dNdm1dqdchiedVdt(m1, q, chi_eff, z, chie_bw_sel) + 3*log(dh) + log(interp1d(z, zs_interp, dvdz_interp)) - log1p(z)
        log_dN_m1ddl = log_dN_m1z - log1p(z) - log(ddl_dz)
        log_dN_m1ddl - log_pdraw
    end

    log_mu = logsumexp(log_sel_wts) - log(Ndraw)
    log_s2 = logsubexp(logsumexp(2 .* log_sel_wts) - 2.0*log(Ndraw), 2*log_mu - log(Ndraw))
    Neff_sel_est = exp(2*log_mu - log_s2)
    Neff_sel = 1/(1/Neff_sel_est + 1/Ndraw)

    Turing.@addlogprob! -nevt*log_mu
    
    mu = exp(log_mu)
    R = nevt / mu + R_scaled * sqrt(nevt)/mu

    alpha_chi_eff = muh - mul # Derivative of mean chi_eff wrt q
    beta_chi_eff = sigmah - sigmal # Derivative of s.d. chi_eff wrt q

    (Neff_sel = Neff_sel, R = R, m1s = m1s, chi_effs = chieffs, alpha_chi_eff=alpha_chi_eff, beta_chi_eff=beta_chi_eff, Ω_Λ = Ω_Λ)
end