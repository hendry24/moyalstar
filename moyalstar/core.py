import sympy as sm

from .utils import get_symbols, _get_symbols_0, make_prime, collect_by_diff

__all__ = ["bopp",
           "star"]


def bopp(q, left=False):
    """
    Bopp shift the input quantity for the calculation of the Moyal star-product. 

    `A(x,p)★B(x,p) = A(x + (i/2)*dp, p - (i/2)*dx) * B(xx, pp)`

    `A(x,p)★B(x,p) = B(x - (i/2)*ddp, p + (i/2)*ddx) * A(xx, pp)`
            
    Parameters
    ----------

    q : sympy object
        Quantity to be Bopp-shifted.

    left : bool, default: False
        Whether the star-product operator is to the left of `q`. 

    Returns
    -------

    out : sympy object
        Bopp-shifted sympy object. 

    References
    ----------
    
        T. Curtright, D. Fairlie, and C. Zachos, A Concise Treatise On Quantum Mechanics In Phase Space (World Scientific Publishing Company, 2013)    

        https://physics.stackexchange.com/questions/578522/why-does-the-star-product-satisfy-the-bopp-shift-relations-fx-p-star-gx-p

    """

    x0, p0 = _get_symbols_0()
    I, W, x, p, xx, pp, ddx, ddp = get_symbols()

    if not(q.is_commutative):
        msg = "Cannot Bopp shift with primed variables present."
        raise ValueError(msg)

    q = q.subs({x:x0, p:p0})
    sgn = 1
    if left:
        sgn = -1
    out = q.subs({x0 : x + sgn * I/2 * ddp,
                   p0 : p - sgn * I/2 * ddx})
    return sm.expand(out)

def _first_index_and_diff_order(q):
    """
    Here q is one term in the expanded sum containing no sum at all.
    q.args thus give its factors.
    """

    I, W, x, p, xx, pp, ddx, ddp = get_symbols()

    """
    Everything to the right of the first "derivative operator" symbol
    must be ordered in .args since we have specified the noncommutativity
    of the primed symbols. It does not matter if the unprimed symbols get
    stuck in the middle since the operator does not work on them. What is 
    important is that x' and p' are correctly placed with respect to the
    derivative operators.
    """
    for idx, qq in enumerate(q.args): 
        if qq == ddx:
            return idx, xx, 1
        if ddx in qq.args:
            return idx, xx, qq.args[1]
        
        if qq == ddp:
            return idx, pp, 1
        if ddp in qq.args:
            return idx, pp, qq.args[1]
        
    return None

def _replace_diff(q):
    if isinstance(q, sm.Add):
        msg = "_replace_diff error. Input q must not be a sympy.Add object"
        msg += "since q.args is expected to give its factors."
        raise ValueError(msg)
    
    fido = _first_index_and_diff_order(q)

    if fido:
        cut_idx, diff_var, diff_order = fido
        prefactor = q.args[:cut_idx]
        q_leftover = sm.Mul(q.args[cut_idx+1:])
        return sm.Mul(*prefactor,
                        sm.Derivative(_replace_diff(q_leftover),
                                      *[diff_var for _ in range(diff_order)]))
                                    # dxdp is treated separately. 
    else:
        return q

def _eval_star(q, do = True):
    I, W, x, p, xx, pp, ddx, ddp = get_symbols()
    
    q = sm.expand(q)

    out = 0
    for qq in q.args: # each term in the sum, order does not matter. 
        out += _replace_diff(qq)

    if do:
        out = out.doit().subs({xx:x, pp:p})
        
    return out

def star(A, B, do = True, collect = True):
    """
    The Moyal star-product A(x,p) ★ B(x,p), calculated using the Bopp shift.
    See `bopp`.

    Parameters
    ----------

    A : sympy object
        Left-hand-side operand. Only one of this and `B` may contain a `sympy.Function`; else an exception
        is raised. Must be unprimed.

    B : sympy object
        Right-hand-side operand, similar to `A`.

    do : bool, default: True
        Whether to evaluate the derivatives and replace the primed variables `xx`,`pp` with the default
        variables `x`,`p`.

    collect : bool, default: True
        Collect the output by the derivative order. This calls `utils.collect_by_diff` with the module's
        `W` as the function.

    Returns
    -------

    out : sympy object
        The Moyal star-product between `A` and `B`.

    References
    ----------
    
        T. Curtright, D. Fairlie, and C. Zachos, A Concise Treatise On Quantum Mechanics In Phase Space (World Scientific Publishing Company, 2013)    

        https://physics.stackexchange.com/questions/578522/why-does-the-star-product-satisfy-the-bopp-shift-relations-fx-p-star-gx-p
    """

    if not(A.is_commutative) or not(B.is_commutative):
        msg = "A and B must be unprimed. Consider substituting xx and pp by x and p, respectively."
        raise ValueError(msg)

    is_function_inside_A = bool(A.atoms(sm.Function))
    is_function_inside_B = bool(B.atoms(sm.Function))

    if is_function_inside_A and is_function_inside_B:
        msg = "One of A and B must be sympy.Function-free to be correctly Bopp shifted."
        raise ValueError(msg)
    
    if is_function_inside_A:
        A = make_prime(A)
        B = bopp(B, left=True)
        q = sm.expand(B * A)    # Expand gives a sum of terms. Redundancy in _eval.
    else:
        A = bopp(A, left=False)
        B = make_prime(B)
        q = sm.expand(A * B)

    out = _eval_star(q, do=do)

    if collect:
        out = collect_by_diff(out)

    return out
