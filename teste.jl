using Base.Test
using NLPModels

include("solve_unc.jl")

# Testar min f(x)
nlp = ADNLPModel(x->(x[1] - 1)^2 + 100*(x[2]-x[1]^2)^2, [-1.2; 1.0])
x, fx, ngx = solve_unc(nlp) #solve_unc é sua implementação
@test norm(x - ones(2)) < 1e-6
@test abs(fx) < 1e-6
println(x)
