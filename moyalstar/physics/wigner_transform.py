import sympy as sm

from .hilbert_ops import moyalstarOp
from ..core import Star
from ..utils.multiprocessing import _mp_helper
            
class WignerTransform():

    def __new__(cls, A : sm.Expr | moyalstarOp):

        A = sm.sympify(A)
        
        if not(bool(A.atoms(moyalstarOp))):
            return A

        if isinstance(A, moyalstarOp):
            return A.wigner_transform
                
        ###
        
        A = A.expand() # generally a sum
        
        if isinstance(A, (sm.Add, sm.Mul)):
            res = _mp_helper(A.args, WignerTransform)
            if isinstance(A, sm.Add):
                return sm.Add(*res)
            return Star(*res, do=True).expand()
        
        if isinstance(A, sm.Pow):
            base : moyalstarOp = A.args[0]
            exponent = A.args[1]
            return (base.wigner_transform ** exponent).expand()
        
        raise ValueError(r"Invalid input in WignerTransform: {%s}" %
                         (sm.latex(A)))