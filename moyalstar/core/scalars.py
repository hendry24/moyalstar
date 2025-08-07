import sympy as sm

from .base import moyalstarBase
from ..utils.cache import _qp_cache

__all__ = ["q", "p", "alpha", "alphaD", "W"]

def _treat_sub(sub, has_sub):
    if ((sub is None) or not(has_sub)):
        return sm.Symbol(r"")
    if isinstance(sub, sm.Symbol):
        return sub
    return sm.Symbol(sm.latex(sub))

class moyalstarScalar(moyalstarBase):
    base = NotImplemented
    has_sub = True
    
    def _get_symbol_name_and_assumptions(cls, sub):
        name = r"%s_{%s}" % (cls.base, sub)
        return name, {"real" : True}
        
    def __new__(cls, sub = None):
        obj = super().__new__(cls, _treat_sub(sub, cls.has_sub)) 
        
        global _qp_cache
        _qp_cache.update([obj])
        return obj

    @property
    def sub(self):
        return self._custom_args[0]
    
class t(moyalstarScalar):
    base = r"t"
    has_sub = False
    
class q(moyalstarScalar):
    """
    The canonical position operator or first phase-space quadrature.
    
    Parameters
    ----------
    
    sub : objects castable to sympy.Symbol
        Subscript signifying subsystem.
    """
    base = r"q"
    
class p(moyalstarScalar):
    """
    The canonical position operator or first phase-space quadrature.
    
    Parameters
    ----------
    
    sub : objects castable to sympy.Symbol
        Subscript signifying subsystem.
    """
    base = r"p"
    
class alpha():
    def __new__(cls, sub = None):
        with sm.evaluate(False):
            return (1 / sm.sqrt(2)) * (q(sub) + sm.I * p(sub))
        
class alphaD():
    def __new__(cls, sub = None):
        with sm.evaluate(False):
            return (1 / sm.sqrt(2)) * (q(sub) - sm.I * p(sub))

###

class _Primed(moyalstarBase):
    def _get_symbol_name_and_assumptions(cls, A):
        return r"{%s}'" % (sm.latex(A)), {"commutative" : False}
    
    def __new__(cls, A : sm.Expr):
        
        A = sm.sympify(A)
        
        if isinstance(A, (q,p)):
            return super().__new__(cls, A)
        
        return A.subs({X:_Primed(X) for X in A.atoms(q,p)})
    
    @property
    def base(self):
        return self._custom_args[0]
    
class _DePrimed():
    def __new__(cls, A : sm.Expr):
        subs_dict = {X : X.base for X in A.atoms(_Primed)}
        return A.subs(subs_dict)

###

class _DerivativeSymbol(moyalstarBase):
    
    def _get_symbol_name_and_assumptions(cls, primed_phase_space_coordinate):
        return r"\partial_{%s}" % (sm.latex(primed_phase_space_coordinate)), {"commutative":False}
    
    def __new__(cls, primed_phase_space_coordinate):
        if not(isinstance(primed_phase_space_coordinate, _Primed)):
            raise ValueError(r"'_DifferentialSymbol' expects '_Primed', but got '%s' instead" % \
                type(primed_phase_space_coordinate))
            
        return super().__new__(cls, primed_phase_space_coordinate)
    
    @property
    def diff_var(self):
        return self._custom_args[0]

####

class WignerFunction(sm.Function):
    """
    The Wigner function object.
    
    Parameters
    ----------
    
    *vars
        Variables of the Wigner function. 
    
    """
    def __str__(self):
        return r"W"
    
    def __repr__(self):
        return str(self)
    
    def _latex(self, printer):
        return str(self)
    
class W():
    """
    The 'WignerFunction' constructor. Constructs 'WignerFunction' using cached 'q' and 'p' as 
    variables. This is the recommended way to create the object since a user might miss 
    some variables with manual construction, leading to incorrect evaluations.
    """
    def __new__(cls):
        global _qp_cache
        return WignerFunction(*list(_qp_cache))