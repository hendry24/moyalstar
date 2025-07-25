import sympy as sm
import sympy.physics.quantum as smq
from functools import cached_property

from .wigner_transform import WignerTransform
from . import scalars
from ..utils.functions import collect_by_derivative
from .hilbert_ops import densityOp, Dagger

__all__ = ["LindbladMasterEquation"]

class _AddOnlyExpr(sm.Expr):
    def __pow__(self, other):
        raise NotImplementedError()
    __rpow__ = __pow__
    __mul__ = __pow__
    __rmul__ = __pow__
    __sub__ = __pow__
    __rsub__ = __pow__
    __truediv__ = __pow__
    __rtruediv__ = __pow__
    
class _TimeDerivativeOfDensityOp(sm.Basic):
    wigner_transform = sm.Derivative(scalars.W(), scalars.t())
    
    def __new__(cls):
        return super().__new__(cls)
    
    def __str__(self):
        return sm.latex(sm.Derivative(densityOp(), scalars.t()))
    
    def __repr__(self):
        return str(self)
    
    def _latex(self, printer):
        return str(self)
    
class _LindbladDissipator(_AddOnlyExpr):
    def __new__(cls, rate = 1, operator_1 = 1, operator_2 = None):
        rate = sm.sympify(rate)
        
        operator_1 = sm.sympify(operator_1)
        
        operator_2 = operator_2 if (operator_2 is not None) else operator_1
        operator_2 = sm.sympify(operator_2)
        
        return super().__new__(cls, rate, operator_1, operator_2)
    
    @property
    def rate(self):
        return self.args[0]
    
    @property
    def operator_1(self):
        return self.args[1]
    
    @property
    def operator_2(self):
        return self.args[2]
    
    def __str__(self):
        if self.operator_1 == self.operator_2:
            op_str = sm.latex(self.operator_1)
        else:
            op_str = r"{%s},{%s}" % (sm.latex(self.operator_1), sm.latex(self.operator_2))

        return r"{{%s}\mathcal{D}\left({%s}\right)\left[\rho\right]}" \
                % (sm.latex(self.rate), op_str)
    
    def __repr__(self):
        return str(self)
    
    def _latex(self, printer):
        return str(self)
    
    def expand(self):
        rho = densityOp()
        P = self.operator_1
        
        Q = self.operator_2
        Qd = Dagger(Q)
        
        out = (2*P*rho*Qd - rho*Qd*P - Qd*P*rho)
        rate_mul = self.rate / 2
        with sm.evaluate(False): # force pretty printing
            out =  rate_mul * out
        return out
    
class LindbladMasterEquation(sm.Basic):
    collect_by_derivative = True
    
    def __new__(cls, 
                H : sm.Expr,
                dissipators : list[list[sm.Expr, sm.Expr]] = []):
        H = sm.sympify(H)
        dissipators = sm.sympify(dissipators)
        return super().__new__(cls, H, dissipators)
    
    @property
    def H(self):
        return self.args[0]
    
    @cached_property
    def dissipators(self):
        out = []
        for inpt in self.args[1]:
            if len(inpt) == 2:
                inpt.append(None) # or inpt[1], same outcome
            elif len(inpt) == 3:
                pass
            else:
                raise ValueError(r"Invalid dissipator specifier : {%s}"
                                 % (inpt))
            rate, operator_1, operator_2 = inpt
            out.append(_LindbladDissipator(rate=rate, 
                                          operator_1=operator_1, 
                                          operator_2=operator_2))
        return out
    
    @property
    def lhs(self):
        return _TimeDerivativeOfDensityOp()

    @property
    def rhs(self):
        out = -sm.I * smq.Commutator(self.H, densityOp())
        for dissip in self.dissipators:
            out += dissip
        return out
    
    @cached_property
    def wigner_transform(self):
        lhs = self.lhs.wigner_transform
        rhs = WignerTransform(self.rhs.doit().expand())
        if self.collect_by_derivative:
            rhs = collect_by_derivative(rhs, lhs.args[0])
        return sm.Equality(lhs, rhs)
    
    def __str__(self):
        return sm.latex(sm.Equality(self.lhs, self.rhs))
    
    def __repr__(self):
        return str(self)
    
    def _latex(self, printer):
        return str(self)