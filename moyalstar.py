import sympy as sm

def get_symbols():
    I = sm.I
    x, p = sm.symbols(r"x p", real=True)
    xx, pp = sm.symbols(r"x' p'", commutative = False)
    ddx, ddp = sm.symbols(r"\partial_{x'} \partial_{p'}", commutative=False)
    return I, x, p, xx, pp, ddx, ddp

def _get_symbols_0():
    x0, p0 = sm.symbols("x_0 p_0", real=True)
    return x0, p0

def make_prime(q):
    x0, p0 = _get_symbols_0()
    I, x, p, xx, pp, ddx, ddp = get_symbols()
    q = q.subs({x:x0, p:p0}) 
    return q.subs({x0 : xx, p0 : pp})

def bopp(q, left=False):
    x0, p0 = _get_symbols_0()
    I, x, p, xx, pp, ddx, ddp = get_symbols()

    if (ddx in q.free_symbols) or (ddp in q.free_symbols):
        raise ValueError("Already Bopp shifted.")
    if not(q.is_commutative):
        msg = "The input is noncommutative. May have already been Bopp shifted."
        msg += "In any case, use noncommutative symbols."
        raise ValueError(msg)

    q = q.subs({x:x0, p:p0})
    sgn = 1
    if left:
        sgn = -1
    out = q.subs({x0 : x + sgn * I/2 * ddp,
                   p0 : p - sgn * I/2 * ddx})
    return sm.expand(out)

def _first_index_and_diff_order(q):
    """
    Here q is one term in the expanded sum containing no sum at all.
    q.args thus give its factors.
    """

    I, x, p, xx, pp, ddx, ddp = get_symbols()

    for idx, qq in enumerate(q.args):
        if qq == ddx:
            return idx, xx, 1
        if ddx in qq.args:
            return idx, xx, qq.args[1]
        
        if qq == ddp:
            return idx, pp, 1
        if ddp in qq.args:
            return idx, pp, qq.args[1]
        
    return None

def _replace_diff(q):
    if isinstance(q, sm.Add):
        msg = "_replace_diff error. Input q must not be a sympy.Add object"
        msg += "since q.args is expected to give its factors."
        raise ValueError(msg)
    
    fido = _first_index_and_diff_order(q)

    if fido:
        cut_idx, diff_var, diff_order = fido
        prefactor = q.args[:cut_idx]
        q_leftover = q.args[cut_idx+1:]
        return sm.Mul(*prefactor,
                        sm.Derivative(_replace_diff(sm.Mul(*q_leftover)),
                                      *[diff_var for _ in range(diff_order)]))
    else:
        return q

def _eval_star(q, do = True):
    I, x, p, xx, pp, ddx, ddp = get_symbols()
    
    q = sm.expand(q)

    out = 0
    for qq in q.args: # each term in the sum.
        out += _replace_diff(qq)

    if do:
        out = out.doit().subs({xx:x, pp:p})
        
    return out

def star(A, B, do = True):
    is_function_inside_A = bool(A.atoms(sm.Function))
    is_function_inside_B = bool(B.atoms(sm.Function))

    if is_function_inside_A and is_function_inside_B:
        msg = "One of A and B must be sympy.Function-free to be correctly Bopp shifted."
        raise ValueError(msg)
    
    if is_function_inside_A:
        A = make_prime(A)
        B = bopp(B, left=True)
        q = sm.expand(B * A)    # Expand gives a sum of terms. Redundancy in _eval.
    else:
        A = bopp(A, left=False)
        B = make_prime(B)
        q = sm.expand(A * B)

    return _eval_star(q, do=do)