import sympy as sm

from .hilbert_ops import moyalstarOp, positionOp, momentumOp, annihilateOp, createOp
from ..core import star


def wigner_transform(A : sm.Expr | moyalstarOp):
    
    q_op = positionOp()
    p_op = momentumOp()
    a_op = annihilateOp()
    ad_op = createOp()
    
    A_atoms = A.atoms()
    
    if not((q_op in A_atoms) or
           (p_op in A_atoms) or
           (a_op in A_atoms) or
           (ad_op in A_atoms)):
        return A
    
    A = A.expand() # generally a sum
    
    if isinstance(A, sm.Add):
        return sm.Add(*[wigner_transform(A_) for A_ in A.args])

    if isinstance(A, sm.Mul):
        return star(*[wigner_transform(A_) for A_ in A.args])
    
    if isinstance(A, sm.Pow):
        base : moyalstarOp = A.args[0]
        exponent = A.args[1]
        return base.wigner_transform ** exponent
    
    if isinstance(A, moyalstarOp):
        return A.wigner_transform
    
    raise ValueError("Invalid input.")