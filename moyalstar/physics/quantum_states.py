import sympy as sm
from ..core import scalars
from ..core.hilbert_ops import moyalstarOp

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
        obj.base = _get_ket_bra_string(a=n, rho_sub="Fock")
        obj._n = n
        return obj
    
    @property
    def n(self):
        return self._n
    
    def wigner_transform(self):
        hh = (scalars.q()**2 + scalars.p()**2)/2
        exponent = (-2*hh).expand()
        return (-1)**self.n/sm.pi * sm.exp(exponent) * sm.laguerre(self.n, 4*hh)
    
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