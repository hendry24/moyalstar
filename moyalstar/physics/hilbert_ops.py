import sympy as sm
from ..utils import objects

class moyalstarOp(sm.Expr):
    
    symb = NotImplemented
    wigner_transform = NotImplemented
    
    is_commutative = False
    
    def __new__(cls):
        return sm.Basic.__new__(cls)
    
    def __str__(self):
        return r"{%s}" % (self.symb)
    
    def __repr__(self):
        return str(self)
    
    def _latex(self, printer):
        return str(self)
    
class positionOp(moyalstarOp):
    symb = r"\hat{q}"
    wigner_transform = objects.q()
    
class momentumOp(moyalstarOp):
    symb = r"\hat{p}"
    wigner_transform = objects.p()
    
class annihilateOp(moyalstarOp):
    symb = r"\hat{a}"
    with sm.evaluate(False):
        wigner_transform = (objects.q()+sm.I * objects.p())/sm.sqrt(2)
    
class createOp(moyalstarOp):
    symb = r"\hat{a}^{\dagger}"
    with sm.evaluate(False):
        wigner_transform = (objects.q()-sm.I * objects.p())/sm.sqrt(2)
        
class densityOp(moyalstarOp):
    symb = r"\rho"
    wigner_transform = objects.W()