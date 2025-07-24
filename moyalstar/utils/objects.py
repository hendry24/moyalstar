import sympy as sm

__all__ = ["t", "q", "p", "W"]

class moyalstarObj(sm.Symbol):
    name = NotImplemented
    is_commutative = NotImplemented
    is_real = NotImplemented
    
    def __new__(cls, name = "", **assumptions):
        return super().__new__(cls, 
                               cls.name, 
                               commutative = cls.is_commutative,
                               real = cls.is_real)

class t(moyalstarObj):
    name = r"t"
    is_commutative = True
    is_real = True

class q(moyalstarObj):
    name = r"q"
    is_commutative = True
    is_real = True
    
class p(moyalstarObj):
    name = r"p"
    is_commutative = True
    is_real = True

class qq(moyalstarObj):
    name = r"q'"
    is_commutative = False
    is_real = False
    
class pp(moyalstarObj):
    name = r"p'"
    is_commutative = False
    is_real = False
    
class dqq(moyalstarObj):
    name = r"\partial_{q'}"
    is_commutative = False
    is_real = False
    
class dpp(moyalstarObj):
    name = r"\partial_{p'}"
    is_commutative = False
    is_real = False

class W(sm.Function):
    def __new__(cls):
        return sm.Function(r"W")(q(), p())