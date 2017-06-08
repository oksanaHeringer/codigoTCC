using Base.Test, NLPModels, Krylov, ForwardDiff
include("reg.jl")

function lagrangiano_aumentado(nlp;μ=10, ϵ=1e-8, λ_min=0, max_time=30, max_iter=1000)
  exit_flag = 0
  iter = 1
  x = nlp.meta.x0
  f(x) = obj(nlp, x)
  c(x) = cons(nlp, x)
  gx = grad(nlp, x)
  Jx = jac_op(nlp, x)
  λ = zeros(nlp.meta.ncon)
  fx = f(x)
  cx = c(x)
  ϵsub = 1/μ
  ∇LA = gx + Jx'*(λ + μ*cx)
  η = 1/μ^(0.1)

  start_time = time()
  elapsed_time = 0.0

  while norm(gx + Jx'*λ) > ϵ || norm(cx) > ϵ && (iter < max_iter)
    subnlp = create_sub_problem(nlp, x, μ, λ)
    x, fx, ng = reg_conf(subnlp, atol=ϵsub)
    fx = f(x)
    cx = c(x)
    gx = grad(nlp, x)
    Jx = jac_op(nlp, x)

    if norm(cx) < η
      λ = λ + μ*cx
      η = η/μ^(0.9)
      ϵsub = ϵsub/μ
    else
      μ = 10*μ
      η = 1/μ^(0.1)
      ϵsub = 1/μ
    end
    ∇LA = gx + Jx'*(λ + μ*cx)
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
  return x, fx, norm(∇LA), norm(cx), iter, elapsed_time, exit_flag
end

function create_sub_problem(nlp, x, μ, λ)
  c(x) = cons(nlp, x)
  LA(x) = obj(nlp,x) + dot(λ,cons(nlp,x)) + μ/2*norm(cons(nlp, x))^2
  ∇LA(x) = grad(nlp,x) + jtprod(nlp, x, λ + μ*c(x))
  HLAv(x, v; obj_weight=1.0, y=zeros) = hprod(nlp, x, v, y=μ*c(x)+λ) + μ*jtprod(nlp, x, jprod(nlp, x, v))
  subnlp = SimpleNLPModel(LA, x, g=∇LA, Hp=HLAv)
end
