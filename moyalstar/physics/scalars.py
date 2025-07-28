import sympy as sm

__all__ = ["t", "q", "p", "W"]

def _treat_sub(sub):
    if sub is None:
        return r""
    if not(isinstance(sub, sm.Basic)):
        return sm.Symbol(str(sub))
        
class moyalstarScalar(sm.Symbol):
    base = NotImplemented
    is_commutative = NotImplemented
    is_real = NotImplemented
    
    def __new__(cls, sub = None, **assumptions):
            
        name = r"{%s}_{%s}" % (cls.base, _treat_sub(sub))
        
        obj = super().__new__(cls, 
                              name = name, 
                              commutative = cls.is_commutative,
                              real = cls.is_real)
        
        return obj
    
class t(moyalstarScalar):
    base = r"t"
    is_commutative = True
    is_real = True

class q(moyalstarScalar):
    base = r"q"
    is_commutative = True
    is_real = True
    
class p(moyalstarScalar):
    base = r"p"
    is_commutative = True
    is_real = True

class _qq(moyalstarScalar):
    base = r"q'"
    is_commutative = False
    is_real = False
    
class _pp(moyalstarScalar):
    base = r"p'"
    is_commutative = False
    is_real = False
    
class _dqq(moyalstarScalar):
    base = r"\partial_{q'}"
    is_commutative = False
    is_real = False
    
class _dpp(moyalstarScalar):
    base = r"\partial_{p'}"
    is_commutative = False
    is_real = False

class W(sm.Function):
    def __new__(cls, subs : list = []):
        if not(subs):
            vars = [q(), p()]
        else:
            vars = [q(sub=sub) for sub in subs] + [p(sub=sub) for sub in subs]
        return sm.Function(r"W")(*vars)