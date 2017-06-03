using Krylov
using ForwardDiff
using LinearOperators

function reg_conf(nlp; η = 0.25 ,η1 = 0.25, η2 = 0.75, σ1 = 0.5, σ2 = 2.0, atol = 1e-6, rtol = 0.0)
    k_max = 10000
    tempo_max = 30
    saida = 0
    tempo = 0.0
    tempo_inicial = time()
    f(x) = obj(nlp, x)
    g(x) = grad(nlp, x)
    H(x) = hess_op(nlp, x)
    x = nlp.meta.x0
    fx = f(x)
    gx = g(x)
    Hx = H(x)
    n = nlp.meta.nvar
    Δ = min(max(0.1*norm(gx), 1), 100)
    tol = atol + rtol*norm(gx)
    k = 0
    while norm(gx) > tol
        d, stats = cg(Hx,-gx, radius=Δ)
        ared = fx - f(x + d)
        pred = -dot(d,gx) - 0.5*dot(Hx*d,d)
        ρ = ared/pred
        if ρ > η
            x = x + d
            gx = g(x)
            fx = f(x)
            Hx = H(x)
        end
        if ρ < η1
            Δ = σ1*Δ
        elseif ρ > η2
            Δ = σ2*Δ
        end
        if k >= k_max
            saida = 1
            break
        end
        tempo = time() - tempo_inicial
        if tempo >= tempo_max
            saida = 2
            break
        end
        k = k + 1
    end
    return x, fx, norm(gx)
end
