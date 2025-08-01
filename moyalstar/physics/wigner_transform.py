import sympy as sm

from ..core.hilbert_ops import moyalstarOp
from ..core.star_product import Star
from ..utils.multiprocessing import _mp_helper

class WignerTransform():
    """
    The Wigner transform.
    
    Parameters
    ----------
    
    A : sm.Expr
    
    """

    def __new__(cls, A : sm.Expr):

        A = sm.sympify(A)
        
        if not(A.has(moyalstarOp)):
            return A

        if isinstance(A, moyalstarOp):
            return A.wigner_transform()
                
        ###
        
        A = A.expand() # generally a sum
        
        if isinstance(A, (sm.Add, sm.Mul)):
            res = _mp_helper(A.args, WignerTransform)
            if isinstance(A, sm.Add):
                return sm.Add(*res)
            return Star(*res).expand()
        
        if isinstance(A, sm.Pow):
            base : moyalstarOp = A.args[0]
            exponent = A.args[1]
            return (base.wigner_transform() ** exponent).expand()
        
        raise ValueError(r"Invalid input in WignerTransform: {%s}" %
                         (sm.latex(A)))