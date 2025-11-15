# Copilot Instructions for Hybrid PNLS Estimation Toolkit

## Project Overview

This is a research toolkit for **Hybrid Partially-Observable Nonlinear System Estimation**. The core problem:
- Systems with **discrete modes** ($X_D$) and **continuous states** ($X_C$)
- Observable subset $Y \subseteq [X_D, X_C]$ obeying nonlinear relationships $f(Y, \theta) = 0$
- Parameters $\theta$ depend on discrete state (mode-dependent)
- Goal: Develop tools and estimation approaches for this hybrid system class

Reference: See `README.md` for mathematical formulation.

## Architecture & Key Components

### Core State Model: `src/models/model.py`

The foundational `HybridStateModel` (ABC) defines the hybrid state representation and enforces all abstract methods:

**State Structure:**
- `X_D` (int32 vector): Discrete state (modes, regimes) - shape `(n_d,)`
- `X_C` (float64 vector): Continuous state (physics variables) - shape `(n_c,)`
- Both are stored as numpy arrays, initialized to zeros if not provided

**Abstract Methods (Subclasses MUST implement):**
- `check_bounds()`: Diagnostic or corrective bound checking (subclass defines policy)
- `enforce_bounds()`: Apply bound enforcement (clip, wrap, error, etc.)
- `handle_bound_violation(report)`: React to bound violations (called by enforce_bounds)
- `state_dynamics(u, dt)`: Compute next state from current state + control input
- `observation_model(**kwargs)`: Extract observable subset from full state

**Helper Methods:**
- `_check_bounds_diagnostic()`: Standard violation detection (no state modification) - for subclass use
- `set_state(x_d, x_c)` / `get_state()`: State accessors with validation

**Design Philosophy:**
Separates state infrastructure (storage, bounds) from domain-specific policies (checking, dynamics, observations). Subclasses define all behavior—parent provides only utilities.

### Example Implementation: `src/models/4D4C2O_Exponential.py`

Concrete example: **4 Discrete × 4 Continuous × 2 Observable Exponential System**

**System Definition:**
- Discrete modes: 4 modes (unchanged during dynamics)
- Continuous state: 4 hidden variables (x1, x2, x3, x4)
- Observable output: Y = [x1, x2] (first 2 continuous states)
- Exponential constraint: $y_1 = a \cdot e^{-b \cdot y_2} + c$

**Input-State Coupling:**
- Scalar control input u corrupted by Gaussian noise: $u_{noisy} = u + N(\text{bias}, \sigma^2)$
- State update: 
  - $x_1^{next} = u_{noisy}$ (noisy input becomes observable state)
  - $x_2^{next} = \ln\left(\frac{u_{noisy} - c}{a}\right) / (-b)$ (inverse exponential, enforces constraint)
  - $x_3^{next} = x_3$ (unchanged)
  - $x_4^{next} = x_4$ (unchanged)

**Parameter Mapping:**
Parameters depend on discrete state:
- $a = 2.0 + 0.5 \cdot X_D[0]$
- $b = 0.5 + 0.1 \cdot X_D[1]$
- $c = 0.2 + 0.05 \cdot X_D[2]$

**Error Handling:**
Raises `ValueError` if exponential constraint cannot be satisfied (e.g., $a \leq 0$ or $u_{noisy} - c \leq 0$). Sets invalid state to NaN with diagnostic error message.

**Noise Characterization:**
Constructor accepts `noise_bias` and `noise_sigma` parameters; applied in state_dynamics during each step.

### Planned Modules (Currently Empty)

- `src/data/`: Data loading, generation, preprocessing
- `src/estimation/`: Filtering, inference, parameter estimation algorithms
- `src/utils/`: Helper functions, utilities

## Development Practices

### Testing & Structure

- Tests directory (`tests/`) exists but is currently empty
- This is a research/exploratory project; establish testing conventions as modules are added
- Consider pytest for future test organization

### State Type Conventions

Maintain strict type discipline:
- **Discrete state X_D**: Always `np.int32` (represents mode indices)
- **Continuous state X_C**: Always `np.float64` (physics variables)
- **Observable state Y**: Subset of [X_D, X_C], typically float64
- **Bounds**: Can be `np.ndarray`, scalar, or `None` (no bound)

### Documentation Style

- Use docstrings with type hints (as seen in `model.py`, `4D4C2O_Exponential.py`)
- Include concrete examples for abstract methods
- Document parameter mappings and constraints explicitly
- Explain error conditions and handling strategies

## Integration Points & Data Flow

1. **System initialization**: Instantiate concrete `HybridStateModel` subclass with initial state, bounds, noise characterization
2. **State evolution**: Call `state_dynamics(u, dt)` to compute next state
3. **Observation extraction**: Call `observation_model(**kwargs)` to get Y subset
4. **Bound enforcement**: Call `enforce_bounds()` after state updates (policy-dependent)
5. **Extensibility**: Future modules (`estimation/`, `data/`) will consume state models for filtering/inference

## Adding New Example Systems

- Extend `HybridStateModel` in `src/models/`
- Implement all 5 abstract methods with domain-specific logic
- Define clear parameter mappings from state to constraint parameters
- Use `_check_bounds_diagnostic()` helper if standard violation checking applies
- Document the physical/mathematical meaning of states and constraints

---

**Last Updated:** November 15, 2025  
**Maturity:** Early-stage research toolkit; core infrastructure stable, example systems in development, main modules pending
