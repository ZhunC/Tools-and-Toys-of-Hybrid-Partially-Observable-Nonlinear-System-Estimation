# Tools-and-Toys-of-Hybrid-Partially-Observable-Nonlinear-System-Estimation
A collection of tools and toys for this particular problem

# Under-Observed Nonlinear Hybrid State Estimation: Problem Description

## 1. Overview

We consider a **hybrid dynamical system** whose full state consists of both discrete and continuous components:

\[
X = 
\begin{bmatrix}
X_D \\
X_C
\end{bmatrix},
\quad
X_D \in \mathbb{R}^4,\;
X_C \in \mathbb{R}^4.
\]

The **state transition dynamics are unknown**, and only part of the continuous state is observable. The goal is to explore how much of the hidden state—both discrete and continuous—can be inferred from limited nonlinear observations.

---

## 2. Observability Structure

Only **two components** of the continuous state are directly measured:

\[
y = 
\begin{bmatrix}
x_{c1} \\
x_{c2}
\end{bmatrix}.
\]

The remaining states in \(X_C\) and all components of \(X_D\) are **hidden**.

The observable components obey a **nonlinear algebraic constraint**:

\[
x_{c2} = a \, e^{-b \, x_{c1}} + c,
\]

where \(a, b, c\) are (possibly mode-dependent) unknown parameters.

---

## 3. Hybrid Interpretation

The discrete state \(X_D\) represents latent system **modes**, switching logic, or operating regimes.  
Examples include:

- different physics regimes  
- actuator/sensor states  
- different parameter sets  
- hidden logic in a hybrid machine  

Each discrete mode may correspond to a different parameter triple:

\[
(a_m,\; b_m,\; c_m),
\qquad m \in \{1,2,3,4\}.
\]

Thus at any time step, the system behaves according to one of several nonlinear curves:

\[
x_{c2} = a_m e^{-b_m x_{c1}} + c_m.
\]

---

## 4. Research Objective

Given only partial and noisy measurements \(y_t = (x_{c1,t}, x_{c2,t})\), we aim to investigate:

1. **Identifiability**  
   Can the underlying discrete mode be inferred from observations?

2. **Parameter recovery**  
   Can we learn \((a_m, b_m, c_m)\) for each mode without labels?

3. **State estimation**  
   How accurately can we reconstruct hidden continuous states?

4. **Hybrid dynamics learning**  
   How can one detect mode transitions from time series data?

5. **Filter design**  
   Could EKF, particle filtering, or mixture models recover the hybrid state?

---

## 5. First Experimental Setup

To begin exploration, we propose a synthetic testbed:

- \(X_D\): dimension 4, acting as a latent discrete mode  
- \(X_C\): dimension 4, with only \(x_{c1}, x_{c2}\) observable  
- 4 discrete modes, each with its own nonlinear relation  
- nonlinear observation constraint:  
  \[
  x_{c2} = a_m e^{-b_m x_{c1}} + c_m
  \]

This synthetic environment allows controlled investigation of:

- partial observability  
- nonlinear sensing  
- mixed-integer state recovery  
- clustering and mode identification  
- filtering over hybrid state spaces  

---

## 6. Goal of the Notebook

This notebook will progressively build:

1. Static (time-independent) examples  
2. Multi-mode nonlinear observation mixtures  
3. Full hybrid dynamical simulations  
4. Sequential and probabilistic state estimation methods  

with the overarching goal of understanding **how much hidden state can be recovered from sparse nonlinear observations** in hybrid systems.

