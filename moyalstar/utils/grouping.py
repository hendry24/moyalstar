import sympy as sm
from sympy.core.function import UndefinedFunction
from ..core import scalars

__all__ = ["collect_by_derivative"]
    
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

    q = scalars.q()
    p = scalars.p()
    if f is None:
        f = scalars.W()

    max_order = max([A_.derivative_count 
                     for A_ in list(A.atoms(sm.Derivative))]+[0])

    def dx_m_dp_n(m, n):
        if m==0 and n==0:
            return f
        return sm.Derivative(f, 
                             *[q for _ in range(m)], 
                             *[p for _ in range(n)])
    
    return sm.collect(A, [dx_m_dp_n(m, n) 
                          for m in range(max_order) 
                          for n in range(max_order - m)])
