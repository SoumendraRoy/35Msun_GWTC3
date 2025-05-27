function dNdm_trapz(m, alpha, mtr, mbh_max, sigma)
    c = 1/(4*(mbh_max-mtr))
    m_max = 2*mbh_max - mtr

    function mbh_of_m(m)
        if m < mtr
            return m
        else
            return mbh_max - c*(m - m_max)^2
        end
    end

    ms = collect(0:0.01:200)[2:end]
    pms = (ms ./ m_max).^(-alpha) .* pdf.(Normal.(mbh_of_m.(ms), sigma), m)

    trapz(ms, pms)
end

@testset "model.jl tests" begin
    @testset "log_dNdm tests" begin
        @test isapprox(log_dNdm(10.0, 2.35, 20.0, 50.0, 1.0), log(dNdm_trapz(10.0, 2.35, 20.0, 50.0, 1.0)), atol=0.05, rtol=0)
        @test isapprox(log_dNdm(21.0, 2.35, 20.0, 50.0, 1.0), log(dNdm_trapz(21.0, 2.35, 20.0, 50.0, 1.0)), atol=0.05, rtol=0)
        @test isapprox(log_dNdm(30.0, 2.35, 20.0, 50.0, 1.0), log(dNdm_trapz(30.0, 2.35, 20.0, 50.0, 1.0)), atol=0.05, rtol=0)
        @test isapprox(log_dNdm(40.0, 2.35, 20.0, 50.0, 1.0), log(dNdm_trapz(40.0, 2.35, 20.0, 50.0, 1.0)), atol=0.05, rtol=0)
        @test isapprox(log_dNdm(51.0, 2.35, 20.0, 50.0, 1.0), log(dNdm_trapz(51.0, 2.35, 20.0, 50.0, 1.0)), atol=0.05, rtol=0)
    end
end