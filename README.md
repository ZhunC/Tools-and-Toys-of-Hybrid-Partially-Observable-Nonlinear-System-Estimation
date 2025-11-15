# Tools-and-Toys-of-Hybrid-Partially-Observable-Nonlinear-System-Estimation
A collection of tools and toys for this particular problem

## 1. Overview

We consider a hybrid dynamical system whose full state consists of discrete components $X_D$ and continuous components $X_C$. The full state is  
$X = [X_D, X_C]^T$  
where $X_D$ has dimension m and $X_C$ has dimension n. The state transition dynamics are unknown, and only part of the continuous state is observable.

---

## 2. Observability Structure

Only a subset of components of the states are directly measured:  
$Y \in X$.

These observed variables obey a parametric nonlinear relationship:  
$f(Y, \theta) = 0$.


---

## 3. Hybrid Interpretation

The states $X$ represent latent system modes or operating regimes.  
Examples include:

- different physics regimes  
- actuator / sensor modes  
- different parameter sets  
- internal logic of a hybrid machine  

Each discrete state mode $x_i$ can have its own parameter vector $\theta$.

Thus, at any time step, the system may lie on one of several nonlinear curves:  
$f(Y, \theta(x_i)) = 0$.

---
## 4. Purpose

The purpose of this repo is to compile tools and environments I have used or think will be useful in the investigation of this problem, alongside some of its results.

---
## 5. GenAI Usage

This repo is written with assistance from OpenAI's ChatGPT and Codex, as well as VS Code Copilot. I have used these tools to draft code and notes. I have reviewed and checked all generated content from inspection and unit testing, making edits I feel needed. The codebase architecture, philosophy, as well as mathematical ideas are all from me, not from GenAI. 



