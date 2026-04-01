module QuadraticModelsSolvers

export clarabel, copt, cplex, cupdlpx, cuopt, gurobi, highs, xpress

using QuadraticModels

function gurobi(m, k...) error("gurobi is not available; make sure to load the package first, e.g. `using Gurobi`") end
function xpress(m, k...) error("xpress is not available; make sure to load the package first, e.g. `using Xpress`") end
function cplex(m, k...) error("cplex is not available; make sure to load the package first, e.g. `using CPLEX`") end
function highs(m, k...) error("highs is not available; make sure to load the package first, e.g. `using HiGHS`") end
function cuopt(m, k...) error("cuopt is not available; make sure to load the package first, e.g. `using cuOpt`") end
function cupdlpx(m, k...) error("cupdlpx is not available; make sure to load the package first, e.g. `using CuPDLPX`") end
function clarabel(m, k...) error("clarabel is not available; make sure to load the package first, e.g. `using Clarabel`") end
function copt(m, k...) error("copt is not available; make sure to load the package first, e.g. `using COPT`") end

end
