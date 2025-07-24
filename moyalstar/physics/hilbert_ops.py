import sympy as sm
from ..utils import objects

class moyalstarOp(sm.Expr):
    
    symb = NotImplemented
    wigner_transform = NotImplemented
    
    is_commutative = False
    
    def __new__(cls):
        return super().__new__(cls)
    
    def __str__(self):
        return r"{%s}" % (self.symb)
    
    def __repr__(self):
        return str(self)
    
    def _latex(self, printer):
        return str(self)
    
    def conj(self):
        raise NotImplementedError
    
class positionOp(moyalstarOp):
    symb = r"\hat{q}"
    wigner_transform = objects.q()
    
    def conj(self):
        return self
    
class momentumOp(moyalstarOp):
    symb = r"\hat{p}"
    wigner_transform = objects.p()
    
    def conj(self):
        return self
    
class annihilateOp(moyalstarOp):
    symb = r"\hat{a}"
    with sm.evaluate(False):
        wigner_transform = (objects.q()+sm.I * objects.p())/sm.sqrt(2)
        
    def conj(self):
        return createOp()
    
class createOp(moyalstarOp):
    symb = r"\hat{a}^{\dagger}"
    with sm.evaluate(False):
        wigner_transform = (objects.q()-sm.I * objects.p())/sm.sqrt(2)
        
    def conj(self):
        return annihilateOp()
        
class densityOp(moyalstarOp):
    symb = r"\rho"
    wigner_transform = objects.W()
    
    def conj(self):
        return self
    
class Dagger():
    """
    Hermitian conjugate of `A`.
    """
    def __new__(cls, A : sm.Expr | moyalstarOp):
        A = sm.sympify(A)
        
        if not(bool(A.atoms(moyalstarOp))):
            return A
        
        A.expand()
        
        if isinstance(A, sm.Add):
            return sm.Add(*[Dagger(A_) for A_ in A.args])
        
        if isinstance(A, sm.Mul):
            return sm.Mul(*list(reversed([Dagger(A_) for A_ in A.args])))
        
        if isinstance(A, sm.Pow):
            base : moyalstarOp = A.args[0]
            exponent  = A.args[1]
            return base.conj() ** exponent
        
        return A.conj()