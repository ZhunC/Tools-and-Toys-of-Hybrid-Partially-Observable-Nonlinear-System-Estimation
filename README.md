# Tools-and-Toys-of-Hybrid-Partially-Observable-Nonlinear-System-Estimation
A collection of tools and toys for this particular problem

## 1. Overview

We consider a hybrid dynamical system whose full state consists of discrete components X_D and continuous components X_C. The full state is  
`X = [X_D, X_C]^T`  
where `X_D` has dimension m and `X_C` has dimension n. The state transition dynamics are unknown, and only part of the continuous state is observable.

---

## 2. Observability Structure

Only a subset of components of the continuous state are directly measured:  
`Y \in X_C`.

These observed variables obey a parametric nonlinear relationship:  
`f(Y, theta) = 0`.


---

## 3. Hybrid Interpretation

The discrete state `X_D` represents latent system modes or operating regimes.  
Examples include:

- different physics regimes  
- actuator / sensor modes  
- different parameter sets  
- internal logic of a hybrid machine  

Each discrete mode `m_i` can have its own parameter vector `theta`.

Thus, at any time step, the system may lie on one of several nonlinear curves:  
`f(X_co, theta_{m_i}) = 0`.

---




