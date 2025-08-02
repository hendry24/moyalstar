import sympy as sm
from ..core import scalars
from ..core.hilbert_ops import densityOp

def _get_ket_bra_string(a, b=None, rho_sub = None):
    if b is None:
        b=a
    
    rho_str = r"\rho"
    if rho_sub:
        rho_str += r"_\mathrm{%s}"%(rho_sub)
    
    return r"{%s} = \left| {%s} \right\rangle \left\langle {%s} \right|" % (rho_str, a, b)

class Fock(densityOp):
    
    def __new__(cls, n):
        n = sm.sympify(n)
        
        base = _get_ket_bra_string(a=sm.latex(n), rho_sub="Fock")
        obj = super().__new__(cls)
        obj.base = base
        obj._n = n
        return obj
    
    @property
    def n(self):
        return self._n
        
    def wigner_transform(self):
        hh = (scalars.q()**2 + scalars.p()**2)/2
        exponent = (-2*hh).expand()
        return (-1)**self.n/sm.pi * sm.exp(exponent) * sm.laguerre(self.n, 4*hh)
    
class Coherent(densityOp):
    def __new__(cls, alpha):
        alpha = sm.sympify(alpha)
        
        obj = super().__new__(cls)
        obj.base = _get_ket_bra_string(a=r"\alpha", rho_sub="coh") + r",\quad \alpha={%s}"%(sm.latex(alpha))
        obj._alpha = alpha
        return obj
    
    @property
    def alpha(self):
        return self._alpha
    
    def wigner_transform(self):
        q0 = sm.re(self.alpha)
        p0 = sm.im(self.alpha)
        q = scalars.q()
        p = scalars.p() 
        with sm.evaluate(False):
            return 1/sm.pi * sm.exp(-(q-q0)**2 - (p-p0)**2)

class Thermal(densityOp):
    def __new__(cls, n_avg):
        n_avg = sm.sympify(n_avg)
        
        obj = super().__new__(cls)
        obj.symb = r"\rho_\mathrm{thermal}, \quad \tilde{n} = {%s}" % (sm.latex(n_avg))
        obj._n_avg = n_avg
        return obj
    
    @property
    def n_avg(self):
        return self._n_avg
    
    def wigner_transform(self):
        q = scalars.q()
        p = scalars.p()
        out = 1/(sm.pi * (self.n_avg + sm.Rational(1,2)))
        out *= sm.exp(-((q**2+p**2))/(self.n_avg + sm.Rational(1,2)))
        return out