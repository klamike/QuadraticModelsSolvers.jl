# QuadraticModelsSolvers

`QuadraticModelsSolvers.jl` is provides the following functions:

- `clarabel`
- `copt`
- `cplex`
- `cuopt`
- `cupdlpx`
- `gurobi`
- `highs`
- `xpress`

Each function is defined only when the corresponding solver package is loaded, using package extensions.

## Example

```julia
using QuadraticModels
using QuadraticModelsSolvers, HiGHS

@info highs(
   QuadraticModel(
      [1.0, 2.0],
      [1, 2],
      [1, 2],
      [4.0, 2.0];
      Arows = [1, 1],
      Acols = [1, 2],
      Avals = [1.0, 1.0],
      lcon = [1.0],
      ucon = [1.0],
      lvar = [0.0, 0.0],
      uvar = [Inf, Inf],
  )
)
```
