import sympy as sm
from sympy.core.function import UndefinedFunction
from . import objects

__all__ = ["get_objects",
           "collect_by_derivative"]

def get_objects() \
    -> tuple[sm.Expr, objects.x, objects.p, UndefinedFunction]:
    """
    Get the basic sympy-compliant symbolic objects.
    
    Returns
    -------

    I : sympy.Expr
        The imaginary number compatible with sympy.

    x : sympy.Expr
        Position variable, equivalent to the Wigner transform of the position operator.

    p : sympy.Expr
        Momentum variable, equivalent to the Wigner transform of the momentum operator.
        
    W : sympy.Expr
        The Wigner function of x and p, `simpy.Function("W")(x,p)'.

    """
    I = sm.I
    x = objects.x()
    p = objects.p()
    W = sm.Function("W")(x,p)
    return I, x, p, W

def _get_primed_objects() \
    -> tuple[objects.xx, objects.pp, objects.dxx, objects.dpp]:
    """
    Get the primed non-commutative symbols.
    
    Returns
    -------

    xx : sympy.Expr
        Primed position `x'`, deliberately made noncommutative to signify that the 
        ``derivative operator'' symbols must operate on them.

    pp : sympy.Expr
        Primed momentum `p'`, similar to `xx`.

    dxx : sympy.Expr
        Primed "x-partial-derivative operator" `D_(x')'' used in the algebraic manipulation prior
        to differentiation in this module. Operates only on `xx`. In a Moyal star-product, 
        Bopp-shifting one of the operands results in derivative operators working on the other 
        operands only. `sympy.Derivative` objects need to have its operands specified right away, 
        while we want to move it around as an operator during the Bopp shift; this is a workaround.

    dpp : sympy.Expr
        Primed "p-partial-derivative operator" `D_(p')`, similar to `ddx`.

    """
    
    return objects.xx(), objects.pp(), objects.dxx(), objects.dpp()

def _make_prime(q : sm.Expr) \
    -> sm.Expr:
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
    
    I, x, p, W = get_objects()
    xx, pp, ddx, ddp = _get_primed_objects()
    
    return q.subs({x : xx,
                   p : pp})
    
def _remove_prime(q : sm.Expr) \
    -> sm.Expr:
    
    I, x, p, W = get_objects()
    xx, pp, ddx, ddp = _get_primed_objects()
    
    return q.subs({xx : x,
                   pp : p})

def collect_by_derivative(q : sm.Expr, 
                          f : None | UndefinedFunction = None) \
    -> sm.Expr:
    """
    Collect terms by the derivatives of the input function, by default those of the Wigner function `W`.

    Parameters
    ----------

    q : sympy object
        Quantity whose terms is to be collected. If `q` contains no
        function, then it is returned as is. 

    f : sympy.Function, default: `W`
        Function whose derivatives are considered.

    Returns
    -------

    out : sympy object
        The same quantity with its terms collected. 
    """

    q = q.expand()

    if not(q.atoms(sm.Function)):
        return q

    I, x, p, W = get_objects()
    if f is None:
        f = W

    max_order = max([qq.derivative_count 
                     for qq in list(q.atoms(sm.Derivative))])

    def dx_m_dp_n(m, n):
        if m==0 and n==0:
            return f
        return sm.Derivative(f, 
                             *[x for _ in range(m)], 
                             *[p for _ in range(n)])
    
    return sm.collect(q, [dx_m_dp_n(m, n) for m in range(max_order) 
                                            for n in range(max_order - m)])
