import sympy as sm

__all__ = ["get_symbols",
           "make_prime",
           "bopp",
           "star"]

def get_symbols():
    """
    Get all the sympy symbols used in this module. In practice, only `I`, `x` and 
    `p` are needed for ease of use.
        - `I`: `simpy.I`.
        - `x`, `p`: position and momentum.
        - `xx`, ``pp``: primed position and momentum. These are deliberately made
           noncommutative to signify that the ``derivative operator'' symbols must 
           operate on them. 
        - ``ddx``, ``ddp``: primed ``derivative operator`` symbols. Needed for the 
           algebraic manipulation prior to differentiation. 

    Returns
    -------

    I : simpy.I
        The imaginary number compatible with sympy.

    W : simpy.Function
        The Wigner function variable, `simpy.Function("W")'.

    x : simpy.Symbol
        Position variable. This is the Wigner transform of the position operator.

    p : simpy.Symbol
        Momentum variable. This is the Wigner transform of the momentum operator.

    xx : simpy.Symbol
        Primed position `x'`, deliberately made noncommutative to signify that the 
        ``derivative operator'' symbols must operate on them.

    pp : simpy.Symbol
        Primed momentum `p'`, similar to `xx`.

    ddx : simpy.Symbol
        Primed "x-partial-derivative operator" `D_(x')'' used in the algebraic manipulation prior
        to differentiation in this module. Operates only on `xx`. In a Moyal
        star-product, Bopp-shifting one of the operands results in derivative
        operators working on the other operands only. 

    ddp : simpy.Symbol
        Primed "p-partial-derivative operator" `D_(p')`, similar to `ddx`.

    """
    I = sm.I
    W = sm.Function("W")
    x, p = sm.symbols(r"x p", real=True)
    xx, pp = sm.symbols(r"x' p'", commutative = False)
    ddx, ddp = sm.symbols(r"\partial_{x'} \partial_{p'}", commutative=False)
    return I, W, x, p, xx, pp, ddx, ddp

def _get_symbols_0():
    x0, p0 = sm.symbols("x_0 p_0", real=True)
    return x0, p0

def make_prime(q):
    """
    Turn the position and momentum variable into their noncommutative primed version,
    i.e. `x` --> `xx` and `p` --> `pp`.

    Parameters
    ----------

    q : sympy object
        A sympy object, e.g. `sympy.Symbol' or `sympy.Add', allowing its `x` and `p` 
        to be replaced by `xx` and `pp`. Refer to `get_symbols' for these variables.


    Returns
    -------

    qq : sympy object
        Primed `q`, is noncommutative. 


    """
    x0, p0 = _get_symbols_0()
    I, x, p, xx, pp, ddx, ddp = get_symbols()
    q = q.subs({x:x0, p:p0}) 

    qq = q.subs({x0 : xx, p0 : pp})

    return qq

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
    I, x, p, xx, pp, ddx, ddp = get_symbols()

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

    I, x, p, xx, pp, ddx, ddp = get_symbols()

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
        q_leftover = q.args[cut_idx+1:]
        return sm.Mul(*prefactor,
                        sm.Derivative(_replace_diff(sm.Mul(*q_leftover)),
                                      *[diff_var for _ in range(diff_order)]))
    else:
        return q

def _eval_star(q, do = True):
    I, x, p, xx, pp, ddx, ddp = get_symbols()
    
    q = sm.expand(q)

    out = 0
    for qq in q.args: # each term in the sum.
        out += _replace_diff(qq)

    if do:
        out = out.doit().subs({xx:x, pp:p})
        
    return out

def star(A, B, do = True):
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

    return _eval_star(q, do=do)