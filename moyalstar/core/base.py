import sympy as sm

class moyalstarBase(sm.Symbol):
    """
    Base object for the package, essentially a modified sympy.Symbol supporting extra accessible
    arguments. 
    """
    
    def _get_symbol_name_and_assumptions(cls, *custom_args):
        raise NotImplementedError()
    
    def __new__(cls, *custom_args):
        
        name, assumptions = cls._get_symbol_name_and_assumptions(cls, *custom_args)
        
        obj = super().__new__(cls,
                              name = name,
                              **assumptions)
        obj._custom_args = custom_args 
        """
        '_args' is used by SymPy and should not be overriden, or the method
        .atoms which is crucial in this package will not work correctly.
            
        This allows us to store custom_args as accessible attributes. We can
        also set what each custom argument is called in a given subclass, 
        by defining a property then returning the argument.
        """        
        return obj

    def __reduce__(self):
        # This specifies how pickling is done for the object and its subclasses.
        # .assumptions0 is needed by sympy.Symbol
        # See https://docs.python.org/3/library/pickle.html
        return self.__class__, self._custom_args, self.assumptions0