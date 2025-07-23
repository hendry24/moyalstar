import sympy as sm
from sympy.core.function import UndefinedFunction
from . import objects

__all__ = ["get_objects",
           "collect_by_derivative"]

def get_objects() \
    -> tuple[sm.Expr, objects.q, objects.p, UndefinedFunction]:
    """
    Get the basic sympy-compliant symbolic objects.
    
    Returns
    -------

    I : sympy.Expr
        The imaginary number compatible with sympy.

    q : sympy.Expr
        Canonical position variable, equivalent to the Wigner transform of the canonical position operator.

    p : sympy.Expr
        Canonical momentum variable, equivalent to the Wigner transform of the canonical momentum operator.
        
    W : sympy.Expr
        The Wigner function of q and p, `simpy.Function("W")(q,p)'.

    """
    I = sm.I
    q = objects.q()
    p = objects.p()
    W = sm.Function("W")(q,p)
    return I, q, p, W

def _get_primed_objects() \
    -> tuple[objects.qq, objects.pp, objects.dqq, objects.dpp]:
    """
    Get the primed non-commutative symbols.
    
    Returns
    -------

    qq : sympy.Expr
        Primed position `x'`, deliberately made noncommutative to signify that the 
        ``derivative operator'' symbols must operate on them.

    pp : sympy.Expr
        Primed momentum `p'`, similar to `qq`.

    dqq : sympy.Expr
        Primed "x-partial-derivative operator" `D_(x')'' used in the algebraic manipulation prior
        to differentiation in this module. Operates only on `qq`. In a Moyal star-product, 
        Bopp-shifting one of the operands results in derivative operators working on the other 
        operands only. `sympy.Derivative` objects need to have its operands specified right away, 
        while we want to move it around as an operator during the Bopp shift; this is a workaround.

    dpp : sympy.Expr
        Primed "p-partial-derivative operator" `D_(p')`, similar to `ddx`.

    """
    
    return objects.qq(), objects.pp(), objects.dqq(), objects.dpp()

def _make_prime(A : sm.Expr) \
    -> sm.Expr:
    """
    Turn the position and momentum variable into their noncommutative primed version,
    i.e. `q` --> `qq` and `p` --> `pp`.

    Parameters
    ----------

    A : sympy object
        A sympy object, e.g. `sympy.Symbol' or `sympy.Add', allowing its `q` and `p` 
        to be replaced by `qq` and `pp`. Refer to `get_symbols' for these variables.

    Returns
    -------

    out : sympy object
        Primed `A`, is noncommutative. 

    """
    
    I, q, p, W = get_objects()
    qq, pp, ddx, ddp = _get_primed_objects()
    
    return A.subs({q : qq,
                   p : pp})
    
def _remove_prime(A : sm.Expr) \
    -> sm.Expr:
    
    I, q, p, W = get_objects()
    qq, pp, ddx, ddp = _get_primed_objects()
    
    return A.subs({qq : q,
                   pp : p})

def collect_by_derivative(A : sm.Expr, 
                          f : None | UndefinedFunction = None) \
    -> sm.Expr:
    """
    Collect terms by the derivatives of the input function, by default those of the Wigner function `W`.

    Parameters
    ----------

    A : sympy object
        Quantity whose terms is to be collected. If `A` contains no
        function, then it is returned as is. 

    f : sympy.Function, default: `W`
        Function whose derivatives are considered.

    Returns
    -------

    out : sympy object
        The same quantity with its terms collected. 
    """

    A = A.expand()

    if not(A.atoms(sm.Function)):
        return A

    I, q, p, W = get_objects()
    if f is None:
        f = W

    max_order = max([A_.derivative_count 
                     for A_ in list(A.atoms(sm.Derivative))])

    def dx_m_dp_n(m, n):
        if m==0 and n==0:
            return f
        return sm.Derivative(f, 
                             *[q for _ in range(m)], 
                             *[p for _ in range(n)])
    
    return sm.collect(A, [dx_m_dp_n(m, n) 
                          for m in range(max_order) 
                          for n in range(max_order - m)])
