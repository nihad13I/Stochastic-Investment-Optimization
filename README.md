# Stochastic Capital Investment Optimization
## Solving Dynamic Programming Models using JAX and Policy Iteration

### Overview
This project implements a quantitative model to determine the optimal investment strategy for a firm subject to productivity shocks. It uses **Policy Iteration** to solve the infinite-horizon Bellman equation, finding the steady-state capital level that maximizes firm value.

### Key Features
* **Stochastic Modeling:** Incorporates a Markov Chain transition matrix for productivity shocks ($z$).
* **High-Performance Computing:** Leverages **Google JAX** for accelerated numerical operations and matrix solving.
* **Policy Iteration:** An efficient alternative to Value Function Iteration (VFI) that solves for the value function by solving the system of linear equations $V = (I - \beta P)^{-1} R$.
* **Transition Analysis:** Analyzes capital accumulation patterns across different productivity states.

### Mathematical Framework
The model solves the objective:
$V(k, z) = \max_{k'} \{ \pi(k, z) - \Phi(k, k') + \beta E[V(k', z') | z] \}$

Where:
* $\pi(k, z)$: Profit function.
* $\Phi(k, k')$: Investment costs and capital depreciation.
* $\beta$: Discount factor.

### Technical Stack
* **Python**
* **JAX** (XLA-backed numerical computing)
* **NumPy**
