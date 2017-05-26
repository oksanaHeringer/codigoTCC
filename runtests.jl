include("PNLPenalidade.jl")

using NLPModels


function runtest()
  #@green("Rodando hs26")
  nlp = ADNLPModel(x->(x[1] - x[2])^2 + (x[2] - x[3])^4, [-2.6; 2.0; 2.0], c = (1 + x[2]^2) * x[1] + x[3]^4 - 3)
  x, fx, nL, ncx, iter, t, s = penalidade_quadratica(nlp) #solve_unc é sua implementação
  #@test norm(x - [0.192249,0.192249] ) < 1e-6
  @test norm(fx) < 1e-6
  println(x)
end
