using Base.Test, NLPModels, ForwardDiff
include("reg.jl")

function penalidade_quadratica(nlp;μ=10, ϵ=1e-8, max_iter=1000, max_time=30)
  exit_flag = 0
  iter = 1
  x = nlp.meta.x0
  f(x) = obj(nlp, x)
  c(x) = cons(nlp, x)
  gx = grad(nlp, x)
  Jx = jac_op(nlp, x)
  λ = cgls(Jx', -gx)[1]
  fx = f(x)
  cx = c(x)
  ϵsub = 1/μ
  ∇L = gx + Jx'*λ
  η = 1/μ^(0.1)

  start_time = time()
  elapsed_time = 0.0

  while norm(∇L) > ϵ || norm(cx) > ϵ && (iter < max_iter)
    subnlp = create_sub_problem(nlp, x, μ)
    x, fx, ng = reg_conf(subnlp, atol=ϵsub)
    fx = f(x)
    cx = c(x)
    gx = grad(nlp, x)
    Jx = jac_op(nlp, x)

    if norm(cx) < η
      λ = cgls(Jx', -gx)[1]
      η = η/μ^(0.9)
      ϵsub = ϵsub/μ
    else
      μ = 10*μ
      η = 1/μ^(0.1)
      ϵsub = 1/μ
    end

    ∇L = gx + Jx'*λ
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
  return x, fx, norm(∇L), norm(cx), iter, elapsed_time, exit_flag
end

function create_sub_problem(nlp, x, μ)
  c(x) = cons(nlp, x)
  Q(x) = obj(nlp, x) + μ/2*norm(c(x))^2
  ∇Q(x) = grad(nlp, x) + μ*jtprod(nlp, x, c(x))
  HQv(x, v; obj_weight=1.0, y=zeros) = hprod(nlp, x, v, y=μ*c(x)) + μ*jtprod(nlp, x, jprod(nlp, x, v))
  subnlp = SimpleNLPModel(Q, x, g=∇Q, Hp=HQv)
end
