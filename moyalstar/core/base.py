import sympy as sm

class moyalstarBase(sm.Symbol):
    """
    Base object for the package, essentially a modified sympy.Symbol supporting accessible
    arguments. A Symbol is used instead of Expr to ensure that the variables are more well-behaved.
    
    """
    
    """
    NOTE:
    Since this package uses pickling for its multiprocessing support, all subclasses of this class
    must have the same call signature for __new__. 
    
    If we want to force a subclass of moyalstarBase to have no argument, a useful workaround would be to write 
    a separate class not subclassing moyalstarBase, whose '__new__' has no argument, then make it return the 
    intended object that is a subclass of moyalstarBase. 
    
    See 'hilbert_ops.densityOp' and 'hilbert_ops.rho', for example. 
    """
    
    def _get_symbol_name_and_assumptions(cls, arg):
        raise NotImplementedError()
    
    def __new__(cls, arg, **assumptions):
        # The `assumptions` kwargs is necessary or the code will break.
        # Presumably, SymPy wants to plug in the base `assumption0` 
        # before adapting to the user-specified input. This applies to
        # all subclass of moyalstarBase.
        
        # Here we override the assumptions fed into sympy.Symbol to force
        # the assumptions that we want.
        name, assumptions = cls._get_symbol_name_and_assumptions(cls, arg)
        
        obj = super().__new__(cls,
                              name = name,
                              **assumptions)
        obj._arg = arg
        """
        By writing it this way, we can store arg as an attribute. We can
        also set what arg is called in the subclass, by defining a property
        then returning arg there.
        """        
        return obj

    def __reduce__(self):
        # Tells pickling package like dill to call self.__class__(self._arg, **self.assumptions0).
        # Dunno if omitting self.assumptions0 will break the code; better be safe here.
        # The comma in (self._arg,) is necessary for SymPy to recognize it as a tuple.
        return (self.__class__, (self._arg,), self.assumptions0)