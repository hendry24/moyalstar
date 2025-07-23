import sympy as sm

from .utils.functions import get_objects, _get_primed_objects, _make_prime, _remove_prime
from .utils import objects
from .utils.multiprocessing import mp_config

from multiprocessing import Pool
import dill
from functools import partial

__all__ = ["bopp",
           "star"]


def star(A : sm.Expr, B : sm.Expr, do : bool = True) \
    -> sm.Expr:
    """
    The Moyal star-product A(x,p) ★ B(x,p), calculated using the Bopp shift.

    Parameters
    ----------

    A : sympy.Expr
        Left-hand-side operand. Only one of this and `B` may contain a `sympy.Function`; else an exception
        is raised.

    B : sympy.Expr
        Right-hand-side operand, similar to `A`.
        
    do : bool
        Whether to do the derivative. Calling `.doit` with to the output wih this argument set to `False`
        is equivalent to the output with this argument set to `True`.

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
    
    any_phase_space_variable_in_A = bool(A.atoms(objects.x, objects.p))
    any_phase_space_variable_in_B = bool(B.atoms(objects.x, objects.p))
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
        B = bopp(B, left=True)
        q = (B * A).expand()
    else:
        A = bopp(A, left=False)
        B = _make_prime(B)
        q = (A * B).expand()

    # Expanding is necessary to ensure that all arguments of q contain no Add objects.
    
    """
    The ★-product evaluation routine called after Bopp shifting, whence
    the primed objects are no longer needed. This function loops through
    the arguments of the input `q` (generally an `Add` object) and replaces 
    the primed objects by the appropriate, functional Objects, i.e., the unprimed
    variables and `sympy.Derivative`. For the derivative objects, this is recursively 
    done by `_replace_diff`. This function then replaces x' and p' by x and p, respectively.
    """

    q : sm.Expr
        
    use_mp = mp_config["enable"] and (len(q.args) >= mp_config["min_num_args"])
    
    if use_mp:
        with Pool(mp_config["num_cpus"]) as pool:
            res = pool.map(_replace_diff_pool_helper,
                           [dill.dumps(qq) for qq in q.args])
            out = sm.Add(*[dill.loads(qq_bytes) for qq_bytes in res])
    
    else:
        out = sm.Add(*[_replace_diff(qq) for qq in q.args])

    out = _remove_prime(out)
    
    if do:
        out = out.doit()
        
    return out

def bopp(q : sm.Expr, left : bool = False) \
    -> sm.Expr:
    """
    Bopp shift the input quantity for the calculation of the Moyal star-product. 

    `A(x,p)★B(x,p) = A(x + (i/2)*dp, p - (i/2)*dx) * B(xx, pp)`

    `A(x,p)★B(x,p) = B(x - (i/2)*ddp, p + (i/2)*ddx) * A(xx, pp)`
            
    Parameters
    ----------

    q : sympy object
        Quantity to be Bopp-shifted.

    left : bool, default: False
        Whether the star-product operator is to the left of `q`. 

    Returns
    -------

    out : sympy object
        Bopp-shifted sympy object. 

    References
    ----------
    
        T. Curtright, D. Fairlie, and C. Zachos, A Concise Treatise On Quantum Mechanics In Phase Space (World Scientific Publishing Company, 2013)    

        https://physics.stackexchange.com/questions/578522/why-does-the-star-product-satisfy-the-bopp-shift-relations-fx-p-star-gx-p

    """
    
    I, x, p, W = get_objects()
    xx, pp, dxx, dpp = _get_primed_objects()

    sgn = 1
    if left:
        sgn = -1
    out = q.subs({x : x + sgn * I/2 * dpp,
                   p : p - sgn * I/2 * dxx})
    return out.expand()

def _replace_diff_pool_helper(q_bytes):
    """
    The package usage involves `sympy.Function`, which the
    package `pickle`, used by `multiprocessing`, cannot pickle.
    As a workaround, here we use `dill` to pickle everything before 
    sending the job to the worker processes. This is the topmost
    function called by a worker process, which loads the bytes input
    by the main process, reconstructing the SymPy objects for 
    `_replace_diff` to work with. Then, the output is pickled once 
    again when sent back to the main process. 
    """
    q = dill.loads(q_bytes)
    return dill.dumps(_replace_diff(q))

def _replace_diff(q : str) \
    -> sm.Expr:
    """
    Recursively replace the differential operator symbols (dxx and dpp),
    with the appropriate `sympy.Derivative` objects.
    """
    
    fido = _first_index_and_diff_order(q)

    if fido: # no more recursion if fido is None
        cut_idx, diff_var, diff_order = fido
        prefactor = q.args[:cut_idx]
        q_leftover = sm.Mul(*q.args[cut_idx+1:])
        return sm.Mul(*prefactor,
                        sm.Derivative(_replace_diff(q_leftover),
                                      *[diff_var]*diff_order))
        """
        With this code, we can afford to replace any power of the first
        dxx or dpp we encounter, instead of replacing only the base
        and letting the rest of the factors be dealt with in the next recursion
        node, making the recursion more efficient. 
        """
    
    return q

def _first_index_and_diff_order(q : sm.Expr) \
    -> None | tuple[int, objects.xx|objects.pp, int|sm.Number]:
    """
    
    Get the index of the first differential operator appearing
    in the Bopp-shifted expression (dxx or dpp), either xx or pp, and 
    the differential order (the power of dxx or dpp).
    
    Parameters
    ----------
    
    q : sympy.Expr
        A summand in the expanded Bopp-shifted expression to be
        evaluated. `q.args` thus give its factors.
        
    Returns
    -------
    
    idx : int
        The index in `q.args` where the first `dxx` or `dpp` object is contained.
        
    diff_var : `xx` or `pp`
        The differentiation variable. Either `xx` or `pp`.
        
    diff_order : int or sm.Number
        The order of the differentiation contained in the `idx`-th argument of 
        `q`, i.e., the exponent of `dxx` or `dpp` encountered.
    """

    xx, pp, dxx, dpp = _get_primed_objects()

    """
    Everything to the right of the first "derivative operator" symbol
    must be ordered in .args since we have specified the noncommutativity
    of the primed symbols. It does not matter if the unprimed symbols get
    stuck in the middle since the operator does not work on them. What is 
    important is that x' and p' are correctly placed with respect to the
    derivative operators.
    """
    for idx, qq in enumerate(q.args): 
        
        if qq == dxx:
            return idx, xx, 1
        if dxx in qq.args:
            # We have dxx**n for n>1. For a Pow object, the second argument gives
            # the exponent; in this case, the differentiation order.
            return idx, xx, qq.args[1]
        
        if qq == dpp:
            return idx, pp, 1
        if dpp in qq.args:
            return idx, pp, qq.args[1]
        
    return None # This stops the recursion. See _replace_diff.