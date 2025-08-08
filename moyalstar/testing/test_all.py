import pytest
import dill
import random
import sympy as sm

from moyalstar.core.scalars import (hbar, moyalstarScalar, q, p, t, W, alpha, alphaD,
                                    _Primed, _DePrimed, _DerivativeSymbol, WignerFunction)
from moyalstar.core.hilbert_ops import (moyalstarOp, positionOp, momentumOp, createOp, annihilateOp,
                                        densityOp, rho, Dagger)
from moyalstar.core.star_product import (Bopp, Star, _star_base,
                                         _first_index_and_diff_order, _replace_diff)

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
        a_sc_expanded = (q() + sm.I*p()) / sm.sqrt(2*hbar)
        assert (a_sc - a_sc_expanded).expand() == 0

        ad_sc = alphaD()
        ad_sc_expanded = (q() - sm.I*p()) / sm.sqrt(2*hbar)
        assert (ad_sc - ad_sc_expanded).expand() == 0
    
    def test_primed(self):
        rand_poly = get_random_poly(objects=[q(), p(), alpha(), alphaD(), sm.Symbol("x")],
                                    coeffs=[1, sm.Symbol(r"\kappa"), sm.exp(-sm.I/2*sm.Symbol(r"\Gamma"))])
        assert _Primed(rand_poly).atoms(_Primed)
        assert not(_Primed(sm.I*2*sm.Symbol("x")).atoms(_Primed))
        assert (_DePrimed(_Primed(rand_poly)) - rand_poly).expand() == 0
        assert not(_Primed(rand_poly).is_commutative)
        assert (_DePrimed(_Primed(rand_poly)) - rand_poly).expand() == 0

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
        for sub in [None, "1", 1, sm.Number(1), sm.Symbol("1")]:
            obj = moyalstarOp(sub)
            assert isinstance(obj.sub, sm.Symbol)
            assert dill.loads(dill.dumps(obj)) == obj
        
        for base, obj in zip([r"\hat{q}", r"\hat{p}", 
                              r"\hat{a}", r"\hat{a}^{\dagger}",
                              r"\rho"], 
                             [positionOp(), momentumOp(), 
                              annihilateOp(), createOp(),
                              densityOp()]):
            assert isinstance(obj, moyalstarOp)
            assert base in sm.latex(obj)
            
        assert rho() == densityOp()
    
    def test_dagger(self):
        assert Dagger(annihilateOp()) == createOp()
        assert Dagger(createOp()) == annihilateOp()

        for herm_op in [positionOp(), momentumOp(), densityOp()]:
            assert Dagger(herm_op) == herm_op
            
        rand_poly = get_random_poly(objects = (1, sm.Symbol("x"), positionOp(), annihilateOp(),
                                               createOp(), annihilateOp()),
                                    coeffs = list(range(10)) + sm.symbols([]))
        assert (Dagger(Dagger(rand_poly)) - rand_poly).expand() == 0
    
    def test_wigner_transform(self):
        for op, wig in zip([positionOp(), momentumOp(),
                            createOp(), annihilateOp(),
                            densityOp()],
                           [q(), p(), 
                            alphaD(), alpha(), 
                            W()]):
            assert (op.wigner_transform() - wig).expand() == 0
            
@pytest.mark.order(2)
class TestStarProduct():
    
    rand_N = random.randint(0, 100)
    
    x = sm.Symbol("x")
    q = q(rand_N)
    qq = _Primed(q)
    dqq = _DerivativeSymbol(qq)
    p = p(rand_N)
    pp = _Primed(p)
    dpp = _DerivativeSymbol(pp)
    a = alpha(rand_N)
    ad = alphaD(rand_N)
        
    def test_bopp_shift(self):
        q_bopp_right =  Bopp(q(), left=False)
        q_bopp_left = Bopp(q(), left=True)
        p_bopp_right = Bopp(p(), left=False)
        p_bopp_left = Bopp(p(), left=True)
        ddq = _DerivativeSymbol(_Primed(q()))
        ddp = _DerivativeSymbol(_Primed(p()))
        
        for bopped, check in zip([q_bopp_right, 
                                  q_bopp_left, 
                                  p_bopp_right, 
                                  p_bopp_left],
                                 [q() + sm.I*hbar/2*ddp,
                                  q() - sm.I*hbar/2*ddp,
                                  p() - sm.I*hbar/2*ddq,
                                  p() + sm.I*hbar/2*ddq]):
            assert (bopped - check).expand() == 0
            assert not(bopped.is_commutative)
            
    def test_fido(self):
        
        def FIDO(x):
            return _first_index_and_diff_order(x)
        
        try:
            FIDO(self.x+2+self.dqq)
            raise ValueError("Input should be invalid.")
        except:
            pass
        
        assert FIDO(1*self.x*self.q*self.pp) is None
        assert FIDO(self.qq**5) is None
       
        assert FIDO(self.dqq) == (0, self.qq, 1)
        assert FIDO(self.dpp) == (0, self.pp, 1)
        
        assert FIDO(self.dqq**self.rand_N) == (0, self.qq, self.rand_N)
        assert FIDO(self.dpp**self.rand_N) == (0, self.pp, self.rand_N)
    
        assert FIDO(self.qq**5*self.p*self.dqq*self.pp) == (2, self.qq, 1)
        assert FIDO(self.dpp*self.dqq) == (0, self.pp, 1)
        
        random_symbols = [sm.Symbol(r"TEST-{%s}" % n, commutative=False) for n in range(100)]
        random_symbols[self.rand_N] = self.dqq
        assert FIDO(sm.Mul(*random_symbols)) == (self.rand_N, self.qq, 1)
        
    def test_replace_diff(self):
        WW = _Primed(W())
        
        assert _replace_diff(sm.Integer(1)) == 1
        assert _replace_diff(self.x) == self.x
        assert _replace_diff(self.dqq) == sm.Derivative(1, self.qq, evaluate=False)
        
        assert _replace_diff(self.dqq*WW) == sm.Derivative(WW, self.qq)
        assert _replace_diff(self.dpp*WW) == sm.Derivative(WW, self.pp)
        
        assert (_replace_diff(self.dqq**2*self.dpp*WW) 
                == sm.Derivative(sm.Derivative(WW, self.pp), 
                                 self.qq, 2, evaluate=False))
        
        assert (_replace_diff(self.dqq*self.qq*self.pp*WW) 
                == sm.Derivative(self.qq*self.pp*WW, self.qq, evaluate=False))
        
    def test_star_base(self):
        def must_raise_error(bad_A, bad_B):
            try:
                _star_base(bad_A, bad_B)
                raise ValueError("Input should be invalid.")
            except:
                pass    
        for bad_A, bad_B in [[sm.sqrt(self.q), sm.sqrt(self.p)],
                             [sm.Function("foo_A")(self.q, self.p), W()],
                             [self.q**0.2, self.p**1.0000]]:
            must_raise_error(bad_A, bad_B)
        
        q1, p1 = q(self.rand_N+1), p(self.rand_N+1)
        q0, p0, a0, ad0 = self.q, self.p, self.a, self.ad
        for A, B, out in [[q0, q0, q0**2],
                          [p0, p0, p0**2],
                          [q0, p0, p0*q0 + sm.I*hbar/2],
                          [p0, q0, p0*q0 - sm.I*hbar/2],
                          [a0, ad0, (q0**2+p0**2+hbar)/(2*hbar)],
                          [ad0, a0, (q0**2+p0**2-hbar)/(2*hbar)],
                          [q0, p1, q0*p1],
                          [p0, q1, p0*q1]]:
            
            assert (_star_base(A, B) - out).expand() == 0
        
    def test_star(self):
        assert Star() == 1
        assert Star(self.q) == self.q
        for n in range(2, 5):
            assert Star(*[self.q]*n) == self.q**n