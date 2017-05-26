using Base.Test
using NLPModels

include("solve_unc.jl")

macro green(msg)
  quote
    println("\033[0;32m$($(msg))\033[m")
  end
end

# Testar min f(x)
begin
  @green("Rodando Rosenbrock")
  nlp = ADNLPModel(x->(x[1] - 1)^2 + 100*(x[2]-x[1]^2)^2, [-1.2; 1.0])
  x, fx, ngx = solve_unc(nlp) #solve_unc é sua implementação
  @test norm(x - ones(2)) < 1e-6
  @test abs(fx) < 1e-6 #o que significa???
  println(x)
end

begin
  @green("Rodando hs11")
  nlp = ADNLPModel(x->(x[1] - 5)^2 + x[2]^2 -25, [4.9; 0.1]) # hs11
  x, fx, ngx = solve_unc(nlp) #solve_unc é sua implementação
  @test norm(x - [5;0] ) < 1e-6
  @test abs(fx - (-25)) < 1e-6
  println(x)
end

begin # Problema SINEVAL do CUTEst
  @green("Rodando SINEVAL")
  nlp = ADNLPModel(x->1e4*(x[2]-sin(x[1]))^2 + x[1]^2/4, [1.0;4.712389]) # Problema SINEVAL do CUTEst
  x, fx, ngx = solve_unc(nlp) #solve_unc é sua implementação
  @test norm(x - [1.39371e-8,1.39373e-8] ) < 1e-6
  @test norm(fx) < 1e-6
  println(x)
end

begin # Problema DENSCHNC do CUTEst
  @green("Rodando DENSCHNC")
  nlp = ADNLPModel(x->(-2+x[1]^2+x[2]^2)^2 + (-2+exp(x[1]-1)+x[2]^3)^2, [1.0;2.0])
  x, fx, ngx = solve_unc(nlp) #solve_unc é sua implementação
  @test norm(fx) < 1e-6
  println(x)
end

begin
  @green("Rodando hs26")
  nlp = ADNLPModel(x->(x[1] - x[2])^2 + (x[2] - x[3])^4, [-2.6, 2.0, 2.0] )
  #c = (1 + x[2]^2) * x[1] + x[3]^4 - 3 == 0
  x, fx, ngx = solve_unc(nlp) #solve_unc é sua implementação
  #@test norm(x - [0.192249,0.192249] ) < 1e-6
  @test norm(fx) < 1e-6
  println(x)
end
