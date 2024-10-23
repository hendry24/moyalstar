# Moyal_star
Moyal star-product with SymPy utilizing the Bopp shift.

### Installation

```
pip install git+https://github.com/hendry24/moyalstar
```

### Tutorial

See ``moyalstar_tutorial.ipynb``.

### References

- T. Curtright, D. Fairlie, and C. Zachos, A Concise Treatise On Quantum Mechanics In Phase Space (World Scientific Publishing Company, 2013)    
- https://physics.stackexchange.com/questions/578522/why-does-the-star-product-satisfy-the-bopp-shift-relations-fx-p-star-gx-p

### Changelog

[0.0.2]
    - Added collection by derivatives in ``core.star``, which calls ``utils.collect_by_diff`` with the module's ``W`` as the function. The
      function can be used to group by another function put into the same term as ``W``. If there is no function, the collection is
      bypassed.

[0.0.1]
    - Added the module.
