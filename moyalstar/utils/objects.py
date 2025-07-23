from sympy import Symbol

__all__ = ["x", "p"]

class moyalstarObj(Symbol):
    name = NotImplemented
    is_commutative = NotImplemented
    is_real = NotImplemented
    
    def __new__(cls):
        return super().__new__(cls, 
                               cls.name, 
                               commutative = cls.is_commutative,
                               real = cls.is_real)
    
class x(moyalstarObj):
    name = r"x"
    is_commutative = True
    is_real = True
    
class p(moyalstarObj):
    name = r"p"
    is_commutative = True
    is_real = True

class xx(moyalstarObj):
    name = r"x'"
    is_commutative = False
    is_real = False
    
class pp(moyalstarObj):
    name = r"p'"
    is_commutative = False
    is_real = False
    
class dxx(moyalstarObj):
    name = r"\partial_{x'}"
    is_commutative = False
    is_real = False
    
class dpp(moyalstarObj):
    name = r"\partial_{p'}"
    is_commutative = False
    is_real = False
