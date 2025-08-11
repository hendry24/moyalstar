import sympy as sp
import typing

from . import scalars
from .base import moyalstarBase
from ..utils.multiprocessing import _mp_helper

class moyalstarOp(moyalstarBase):
    
    base = NotImplemented
    has_sub = True
    
    def _get_symbol_name_and_assumptions(cls, sub):
        return r"%s_{%s}" % (cls.base, sub), {"commutative":False}
    
    def __new__(cls, sub = None):
        return super().__new__(cls, scalars._treat_sub(sub, cls.has_sub))
    
    @property
    def sub(self):
        return self._custom_args[0]
    
    def dagger(self):
        raise NotImplementedError()
    
    def wigner_transform(self):
        raise NotImplementedError()
        
class Dagger():
    """
    Hermitian conjugate of `A`.
    """
    def __new__(cls, A : sp.Expr | moyalstarOp):
        A = sp.sympify(A)
        
        if not(bool(A.atoms(moyalstarOp))):
            return A.conjugate()
        
        A.expand()
        
        if isinstance(A, sp.Add):
            res = _mp_helper(A.args, Dagger)
            return sp.Add(*res)
        
        if isinstance(A, sp.Mul):
            res = _mp_helper(A.args, Dagger)
            return sp.Mul(*list(reversed(res)))
        
        if isinstance(A, sp.Pow):
            base : moyalstarOp = A.args[0]
            exponent  = A.args[1]
            return base.dagger() ** exponent
        
        return A.dagger()
    
class HermitianOp(moyalstarOp):
    
    @typing.final
    def dagger(self):
        return self
    
class qOp(HermitianOp):
    base = r"\hat{q}"

    def wigner_transform(self):
        return scalars.q(sub = self.sub)
    
class pOp(HermitianOp):
    base = r"\hat{p}"
        
    def wigner_transform(self):
        return scalars.p(sub = self.sub)
    
class annihilateOp(moyalstarOp):
    base = r"\hat{a}"
        
    def define(self):
        with sp.evaluate(False):
            return (qOp(sub=self.sub) + sp.I * pOp(sub=self.sub)) / sp.sqrt(2*scalars.hbar)
    
    def dagger(self):
        return createOp(sub = self.sub)
    
    def wigner_transform(self):
        with sp.evaluate(False):
            return scalars.alpha(sub = self.sub)
    
class createOp(moyalstarOp):
    base = r"\hat{a}^{\dagger}"
    
    def define(self):
        return self.dagger().define()
        
    def dagger(self):
        return annihilateOp(sub = self.sub)
    
    def wigner_transform(self):
        with sp.evaluate(False):
            return scalars.alphaD(sub = self.sub)
        
class densityOp(HermitianOp):
    base = r"\rho"
    has_sub = False
    
    def __new__(cls, sub=None):
        return super().__new__(cls, sub)
    
    def wigner_transform(self):
        return scalars.W()
    
class rho():
    def __new__(cls):
        return densityOp()