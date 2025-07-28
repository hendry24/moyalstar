import sympy as sm
from . import scalars

class moyalstarOp(sm.Expr):
    
    symb = NotImplemented
    wigner_transform = NotImplemented
    
    is_commutative = False
    
    def __new__(cls, sub = None):
        return super().__new__(cls, scalars._treat_sub(sub))
    
    @property
    def sub(self):
        return self.args[0]
    
    def __str__(self):
        return r"{%s}_{%s}" % (self.symb, self.sub)
    
    def __repr__(self):
        return str(self)
    
    def _latex(self, printer):
        return str(self)
    
    def conj(self):
        raise NotImplementedError
    
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
    
class positionOp(moyalstarOp):
    symb = r"\hat{q}"
    wigner_transform = scalars.q()
    
    def conj(self):
        return self
    
class momentumOp(moyalstarOp):
    symb = r"\hat{p}"
    wigner_transform = scalars.p()
    
    def conj(self):
        return self
    
class annihilateOp(moyalstarOp):
    symb = r"\hat{a}"
    with sm.evaluate(False):
        wigner_transform = (scalars.q()+sm.I * scalars.p())/sm.sqrt(2)
        
    def conj(self):
        return createOp()
    
class createOp(moyalstarOp):
    symb = r"\hat{a}^{\dagger}"
    with sm.evaluate(False):
        wigner_transform = (scalars.q()-sm.I * scalars.p())/sm.sqrt(2)
        
    def conj(self):
        return annihilateOp()
        
class densityOp(moyalstarOp):
    symb = r"\rho"
    wigner_transform = scalars.W()
    
    def conj(self):
        return self

def _get_ket_bra_string(a, b=None, rho_sub = None):
    if b is None:
        b=a
    
    rho_str = r"\rho"
    if rho_sub:
        rho_str += r"_\mathrm{%s}"%(rho_sub)
    
    return r"{%s} = \left| {%s} \right\rangle \left\langle {%s} \right|" % (rho_str, a, b)

class Fock(moyalstarOp):
    def __new__(cls, n):
        obj = super().__new__(cls)
        obj.symb = _get_ket_bra_string(a=n, rho_sub="Fock")
        
        hh = (scalars.q()**2 + scalars.p()**2)/2
        exponent = (-2*hh).expand()
        obj.wigner_transform = (-1)**n/sm.pi * sm.exp(exponent) * sm.laguerre(n, 4*hh)
        return obj
    
class Coherent(moyalstarOp):
    def __new__(cls, alpha):
        obj = super().__new__(cls)
        obj.symb = _get_ket_bra_string(a=r"\alpha", rho_sub="coh") + r",\quad \alpha={%s}"%(alpha)
        
        q0 = sm.re(alpha)
        p0 = sm.im(alpha)
        q = scalars.q()
        p = scalars.p() 
        with sm.evaluate(False):
            obj.wigner_transform = 1/sm.pi * sm.exp(-(q-q0)**2 - (p-p0)**2)
        return obj

class Thermal(moyalstarOp):
    def __new__(cls, n_avg):
        obj = super().__new__(cls)
        obj.symb = r"\rho_\mathrm{thermal}, \quad \tilde{n} = {%s}"%(n_avg)
        
        q = scalars.q()
        p = scalars.p()
        obj.wigner_transform = 1/(sm.pi * (n_avg + sm.Rational(1,2)))
        obj.wigner_transform *= sm.exp(-((q**2+p**2))/(n_avg + sm.Rational(1,2)))
        return obj