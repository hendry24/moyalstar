import sympy as sm

__all__ = ["get_symbols",
           "make_prime",
           "collect_by_diff"]

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
    I, W, x, p, xx, pp, ddx, ddp = get_symbols()
    q = q.subs({x:x0, p:p0}) 

    qq = q.subs({x0 : xx, p0 : pp})

    return qq

def collect_by_diff(q, W = None):
    """
    Collect terms by the derivatives of the input function, by default those of `W`.

    Parameters
    ----------

    q : sympy object
        Quantity whose terms is to be collected. If `q` contains no
        function, then it is returned as is. 

    W : sympy.Function, default: `W`
        Function whose derivatives are considered.

    Returns
    -------

    out : sympy object
        The same quantity with its terms collected. 
    """

    if not(q.atoms(sm.Function)):
        return q

    if W is None:
        I, W, x, p, xx, pp, ddx, ddp = get_symbols()

    max_order = max([qq.derivative_count 
                     for qq in list(q.atoms(sm.Derivative))])

    def dx_m_dp_n(m, n):
        if m==0 and n==0:
            return W
        return sm.Derivative(W, *[x]*m, *[p]*n)
    
    return sm.collect(q, [dx_m_dp_n(m, n) for m in range(max_order) 
                                            for n in range(max_order - m)])
