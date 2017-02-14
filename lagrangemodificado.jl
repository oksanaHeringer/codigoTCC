#Oksana Heringer da Silva - GRR20110467 - Trabalho MiniTCC - Data:02/12/2015
#Método de Lagrange

using NLPModels
using Krylov

function lagrangeano_aumentado(nlp,W;u=1.0, ϵ=1e-6, max_time=60, max_iter=1000)
    exit_flag = 0
    iter = 1
    x = nlp.meta.x0
    f(x) = obj(nlp, x)
    g(x) = grad(nlp, x)
    J(x) = (hess_op(nlp, x))
    c(x) = cons(nlp, x)#restrição
    fx = f(x)
    gx = g(x)
    Jx = J(x)
    cx = c(x)
    λ = (Jx*Jx')\(Jx*gx)
    Wx = W(x,λ)
    #L=fx-dot(cx,λ)
    gL = gx-Jx'*λ
    start_time = time()
    elapsed_time = 0.0

  while norm(gL) > ϵ || norm(cx) > ϵ
    #t = 1.0
    d, stats =  cg(Hx,-∇fx)
    x = x + d
    #println("x = $(round(x,3))")
    fx = f(x)
    gx = grad(nlp, x)
    Jx = (hess_op(nlp, x))
    Wx = W(x,λ)
    cx = c(x)
    #Lnew = fx-dot(cx,λ)
    #L=Lnew
    λ = (Jx*Jx')\(Jx*gx)
    u = u*1.1
    #τ = τ*0.9
    gL = gx-Jx'*λ
    iter = iter + 1
    if iter >= max_iter
        exit_flag = 1
        break
    end
    elapsed_time = time() - start_time
    if elapsed_time >= max_time
        exit_flag = 2
        break
    end
  end
  return x, fx, gx, exit_flag, iter, elapsed_time
end


#=function newton(Wx,gx,Jx,λ,u,cx)
  d = -Wx\(gx - Jx'*(λ-u*cx))
  return d
end=#
