import sympy as sm

from .utils.functions import get_objects, _get_primed_objects, _make_prime, _remove_prime
from .utils import objects
from .utils.multiprocessing import _mp_helper

__all__ = ["Bopp",
           "Star"]

class Star():
    def __new__(cls, *args):
        if len(args) == 0:
            return sm.Integer(1)
        
        out = sm.sympify(args[0])
        for arg in args[1:]:
            out = _star_base(out, sm.sympify(arg))
        return out
    
def _star_base(A : sm.Expr, B : sm.Expr) \
    -> sm.Expr:
    """
    The Moyal star-product A(q,p) ★ B(q,p), calculated using the Bopp shift.

    Parameters
    ----------

    A : sympy.Expr
        Left-hand-side operand. Only one of this and `B` may contain a `sympy.Function`; else an exception
        is raised.

    B : sympy.Expr
        Right-hand-side operand, similar to `A`.

    Returns
    -------

    out : sympy.Expr
        The Moyal star-product between `A` and `B`.

    References
    ----------
    
        T. Curtright, D. Fairlie, and C. Zachos, A Concise Treatise On Quantum Mechanics In Phase Space (World Scientific Publishing Company, 2013)    

        https://physics.stackexchange.com/questions/578522/why-does-the-star-product-satisfy-the-bopp-shift-relations-fx-p-star-gx-p
    
    See Also
    --------
    
    moyalstar.bopp : Bopp shift the input expression. 
    
    """
    
    any_phase_space_variable_in_A = bool(A.atoms(objects.q, objects.p))
    any_phase_space_variable_in_B = bool(B.atoms(objects.q, objects.p))
    if (not(any_phase_space_variable_in_A) or 
        not (any_phase_space_variable_in_B)):
        return A*B

    is_function_inside_A = bool(A.atoms(sm.Function))
    is_function_inside_B = bool(B.atoms(sm.Function))
    if is_function_inside_A and is_function_inside_B:
        msg = "One of A and B must be sympy.Function-free to be correctly Bopp shifted."
        raise ValueError(msg)
    
    if is_function_inside_A:
        A = _make_prime(A)
        B = Bopp(B, left=True)
        X = (B * A).expand()
    else:
        A = Bopp(A, left=False)
        B = _make_prime(B)
        X = (A * B).expand()

    # Expanding is necessary to ensure that all arguments of q contain no Add objects.
    
    """
    The ★-product evaluation routine called after Bopp shifting, whence
    the primed objects are no longer needed. This function loops through
    the arguments of the input `X` (generally an `Add` object) and replaces 
    the primed objects by the appropriate, functional Objects, i.e., the unprimed
    variables and `sympy.Derivative`. For the derivative objects, this is recursively 
    done by `_replace_diff`. This function then replaces q' and p' by q and p, respectively.
    """

    X : sm.Expr
    if isinstance(X, sm.Add):
        X_args = X.args
    else:
        X_args = [X]
    
    out = sm.Add(*_mp_helper(X_args, _replace_diff))

    out = _remove_prime(out)
                
    return out.doit().expand()

class Bopp():
    """
    Bopp shift the input quantity for the calculation of the Moyal star-product. 

    `A(q,p)★B(q,p) = A(q + (i/2)*dpp, p - (i/2)*dqq) * B(qq, pp)`

    `A(x,p)★B(x,p) = B(q - (i/2)*dpp, p + (i/2)*dqq) * A(qq, pp)`
    
    In the current version, this operation attempts to remove all `sympy.Derivative`
    in the input expression to ensure a correct Bopp shift. This is why `Star` only
    accepts one 'UndefinedFunction' at maximum, as the algorithm can just Bopp-shift
    the other derivative-free expression. 
            
    Parameters
    ----------

    A : sympy.Expr
        Quantity to be Bopp-shifted, should contain `objects.q` or `objects.p`.

    left : bool, default: False
        Whether the star-product operator is to the left of `A`. 

    Returns
    -------

    out : sympy object
        Bopp-shifted sympy object. 

    References
    ----------
    
        T. Curtright, D. Fairlie, and C. Zachos, A Concise Treatise On Quantum Mechanics In Phase Space (World Scientific Publishing Company, 2013)    

        https://physics.stackexchange.com/questions/578522/why-does-the-star-product-satisfy-the-bopp-shift-relations-fx-p-star-gx-p

    """
    
    def __new__(cls, A : sm.Expr, left : bool = False):
        I, q, p, W = get_objects()
        qq, pp, dqq, dpp = _get_primed_objects()
        
        if bool(A.atoms(sm.Derivative)):
            A = A.doit()
            if bool(A.atoms(sm.Derivative)):
                s = "'A' contains persistent derivative(s), most possibly working on an UndefinedFunction."
                s += "The function has tried to call 'A.doit()' but couldn't get rid of the 'Derivative"
                s += "objecs. This leads to faulty Bopp-shifting which results in incorrect ★-products evaluation."
                raise ValueError(s)
            
        """
        The derivative evaluation attempt in `Bopp` will deal with intermediate
        unevaluated derivative(s) during the ★-product chain in `Star`. Since
        `Bopp` is not called on the operand containing an UndefinedFunction, this
        effectively keeps the derivative(s) operating on expressions containing
        the UndefinedFunction from being evaluated, resulting in a "prettier" output.
        
        The evaluation-prohibition is not applied in the current version, but the above code
        is nevertheless useful to catch errors, so we keep it there.
        """

        sgn = 1
        if left:
            sgn = -1
        out = A.subs({q : q + sgn * I/2 * dpp,
                    p : p - sgn * I/2 * dqq})
        return out.expand()

def _replace_diff(A : sm.Expr) \
    -> sm.Expr:
    """
    Recursively replace the differential operator symbols (dqq and dpp),
    with the appropriate `sympy.Derivative` objects.
    """
    
    fido = _first_index_and_diff_order(A)

    if fido: # no more recursion if fido is None
        cut_idx, diff_var, diff_order = fido
        prefactor = A.args[:cut_idx]
        A_leftover = sm.Mul(*A.args[cut_idx+1:])
        return sm.Mul(*prefactor,
                        sm.Derivative(_replace_diff(A_leftover),
                                      *[diff_var]*diff_order))
        """
        With this code, we can afford to replace any power of the first
        dqq or dpp we encounter, instead of replacing only the base
        and letting the rest of the factors be dealt with in the next recursion
        node, making the recursion more efficient. 
        """
    
    return A

def _first_index_and_diff_order(A : sm.Expr) \
    -> None | tuple[int, objects.qq|objects.pp, int|sm.Number]:
    """
    
    Get the index of the first differential operator appearing
    in the Bopp-shifted expression (dqq or dpp), either qq or pp, and 
    the differential order (the power of dqq or dpp).
    
    Parameters
    ----------
    
    A : sympy.Expr
        A summand in the expanded Bopp-shifted expression to be
        evaluated. `A.args` thus give its factors.
        
    Returns
    -------
    
    idx : int
        The index in `A.args` where the first `dqq` or `dpp` object is contained.
        
    diff_var : `qq` or `pp`
        The differentiation variable. Either `qq` or `pp`.
        
    diff_order : int or sm.Number
        The order of the differentiation contained in the `idx`-th argument of 
        `A`, i.e., the exponent of `dqq` or `dpp` encountered.
    """

    qq, pp, dqq, dpp = _get_primed_objects()

    """
    Everything to the right of the first "derivative operator" symbol
    must be ordered in .args since we have specified the noncommutativity
    of the primed symbols. It does not matter if the unprimed symbols get
    stuck in the middle since the operator does not work on them. What is 
    important is that x' and p' are correctly placed with respect to the
    derivative operators.
    """
    for idx, A_ in enumerate(A.args): 
        
        if A_ == dqq:
            return idx, qq, 1
        if dqq in A_.args:
            # We have dqq**n for n>1. For a Pow object, the second argument gives
            # the exponent; in this case, the differentiation order.
            return idx, qq, A_.args[1]
        
        if A_ == dpp:
            return idx, pp, 1
        if dpp in A_.args:
            return idx, pp, A_.args[1]
        
    return None # This stops the recursion. See _replace_diff.