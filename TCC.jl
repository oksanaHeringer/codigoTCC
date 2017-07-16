using CUTEst, BenchmarkProfiles, Plots
include("PNLPenalidade.jl")

#problems = CUTEst.select(min_con=1, only_free_var=true, only_equ_con=true)
problems = CUTEst.select(max_var=100, max_con=100, min_con=1, only_free_var=true, only_equ_con=true)
#problems = filter(x->contains(x, "HS") && length(x) <= 5, CUTEst.select(only_free_var=true, only_equ_con=true))
#problems = ["BT1", "HS26", "HS27", "HS28", "HS39"]
metodos = [lagrangiano_exato, lagrangiano_aumentado, penalidade_quadratica]
sort!(problems)
np = length(problems)
nmet = length(metodos)
T = -ones(np, nmet)
Av = -ones(np, nmet)
for (j,metodo) in enumerate(metodos)
  open("$metodo.txt", "w") do file
    str = @sprintf("%8s  %4s  %4s  %10s  %10s  %7s %7s %7s  %7s  %7s  %7s  %10s\n",
      "Problema", "nvar", "ncon", "f(x)", "‖∇ℓ(x,λ)‖", "‖c(x)‖","iter","saida","nf","nc","soma","tempo")
    print(str)
    print(file,str)
    for (i,p) in enumerate(problems)
      nlp = CUTEstModel(p)
      c = nlp.counters
      try
       x, fx, nlx, ncx, max_iter, max_time, s = metodo(nlp)# Seu método
       str = @sprintf("%8s  %4d  %4d  %10.4e  %10.4e  %10.4e %7d   %7d  %7d  %7d  %7d  %10.8f\n",
          p, nlp.meta.nvar, nlp.meta.ncon, fx, nlx, ncx, max_iter, s, c.neval_obj, c.neval_cons,
          sum_counters(nlp), max_time)
       print(str)
       print(file, str)
       if s == 0
         T[i,j] = max_time
         Av[i,j] = sum_counters(nlp)
       end
       reset!(nlp)
       catch
       #str = @printf("%-7s  %s\n", p, "failure")
       str = @sprintf("%8s  %4d  %4d  %10.4e  %10.4e  %10.4e %7d   %7d  %7d  %7d  %7d  %10.8f\n",
          p, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3, 0.0, 0.0,0, 0.0)
       print(file, str)
       reset!(nlp)
      finally
        finalize(nlp)
      end
    end
  end
end

performance_profile(T, ["Lagrange λ quad min", "Lagrange λ iterativo", "Penalidade"], legend=:bottomright)
xlabel!("Tempo (escala logarítmica)")
ylabel!("Proporção de problemas")
png("perf-tempo")

performance_profile(Av, ["Lagrange λ quad min", "Lagrange λ iterativo", "Penalidade"], legend=:bottomright)
xlabel!("Número de Avaliações (escala logarítmica)")
ylabel!("Proporção de problemas")
png("avaliacoes")
