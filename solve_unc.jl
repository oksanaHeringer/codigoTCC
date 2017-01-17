function solve_unc(nlp; tol=1e-6, maxiter = 1000)
    x = nlp.meta.x0
    f(x) = obj(nlp, x)
    ∇f(x) = grad(nlp, x)
    H(x) = full(hess_op(nlp, x))
    iter = 0
    ef = 0
    fx = f(x)
    ∇fx = ∇f(x)
    Hx = H(x)
    while norm(∇fx) > tol
        t = 1.0
        d = -Hx\∇fx
        while f(x + t*d) > fx + 0.01*t*dot(∇fx,d)
            t = t*0.9
        end
        x = x + t*d
        fx = f(x)
        ∇fx = ∇f(x)
        Hx = H(x)
        iter += 1
        if iter >= maxiter
            ef = 1
            break
        end

    end
    return x, fx, norm(∇fx)
end
