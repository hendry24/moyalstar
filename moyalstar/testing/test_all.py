import pytest
import dill
import random
import sympy as sm

from moyalstar.core.scalars import (moyalstarScalar, q, p,t, W, alpha, alphaD,
                                    _Primed, _DePrimed, _DerivativeSymbol, WignerFunction)
from moyalstar.utils.cache import _qp_cache

def get_random_poly(objects, coeffs=[1], max_pow=3, dice_throw=10):
    """
    Make a random polynomial in 'objects'.
    """
    return sm.Add(*[sm.Mul(*[random.choice(coeffs)*random.choice(objects)**random.randint(0, max_pow)
                             for _ in range(dice_throw)])
                    for _ in range(dice_throw)])

@pytest.mark.order(0)
class TestScalars():
    global _qp_cache
    
    def test_scalar_construction(self):
        for sub in [None, "1", 1, sm.Number(1), sm.Symbol("1")]:
            obj = moyalstarScalar(sub)
            assert isinstance(obj.sub, sm.Symbol)
            assert obj in _qp_cache
            assert dill.loads(dill.dumps(obj)) == obj
        
        for base, obj in zip(["t", "q", "p"], [t(), q(), p()]):
            assert isinstance(obj, moyalstarScalar)
            assert base in sm.latex(obj)
    
    def test_alpha(self):
        a_sc = alpha()
        a_sc_expanded = (q() + sm.I*p()) / sm.sqrt(2)
        assert (a_sc - a_sc_expanded).expand() == 0

        ad_sc = alphaD()
        ad_sc_expanded = (q() - sm.I*p()) / sm.sqrt(2)
        assert (ad_sc - ad_sc_expanded).expand() == 0
    
    def test_primed(self):
        rand_poly = get_random_poly(objects=[q(), p(), alpha(), alphaD(), sm.Symbol("x")],
                                    coeffs=[1, sm.Symbol(r"\kappa"), sm.exp(-sm.I/2*sm.Symbol(r"\Gamma"))])
        assert _Primed(rand_poly).atoms(_Primed)
        assert not(_Primed(sm.I*2*sm.Symbol("x")).atoms(_Primed))
        assert (_DePrimed(_Primed(rand_poly)) - rand_poly).expand() == 0
        assert not(_Primed(rand_poly).is_commutative)

    def test_derivative_symbol(self):
        try:
            _DerivativeSymbol(q())
            raise TypeError("Input must be _Primed.")
        except:
            pass
        der = _DerivativeSymbol(_Primed(q()))
        assert isinstance(der.diff_var, _Primed)
        assert not(der.is_commutative)
        
    def test_W(self):
        assert isinstance(W(), WignerFunction)
        assert isinstance(W(), sm.Function)
        assert W().free_symbols == _qp_cache
        W_str = sm.latex(W)
        assert ("W" in W_str and
                "q" not in W_str and
                "p" not in W_str)
        
@pytest.mark.order(1)
class TestHilbertOps():
    def test_operator_construction(self):
        pass
    
    def test_dagger(self):
        rand_poly = get_random_poly(objects = (),
                                    coeffs = list(range(10)) + sm.symbols([]))
    
    def test_wigner_transform(self):
        pass