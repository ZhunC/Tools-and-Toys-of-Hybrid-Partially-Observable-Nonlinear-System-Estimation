from __future__ import annotations
from typing import Optional, Tuple, Dict, Any
import numpy as np
from abc import ABC, abstractmethod


class HybridStateModel(ABC):
    """
    Base class for hybrid (discrete + continuous) state models.

    State:
        X_D : np.ndarray
            Discrete state vector, shape (n_d,), dtype int32
        X_C : np.ndarray
            Continuous state vector, shape (n_c,), dtype float64

    Bounds (optional):
        x_d_min, x_d_max : np.ndarray or scalar or None
        x_c_min, x_c_max : np.ndarray or scalar or None

    Child classes MUST implement (abstract methods):
        - `check_bounds()`: Define bound-checking policy (diagnostic or corrective)
        - `enforce_bounds()`: Define bound-enforcement policy (error, clip, wrap, etc.)
        - `handle_bound_violation(report)`: Define violation handling policy
        - `state_dynamics(u, dt)`: System dynamics (discrete/continuous state evolution)
        - `observation_model(**kwargs)`: Measurement function (extract observable subset)

    Helper methods:
        - `_check_bounds_diagnostic()`: Standard violation detection (for subclass use)

    The core abstraction separates state storage from domain-specific behavior.
    Subclasses define all policies (no parent-class defaults).
    """

    def __init__(
        self,
        x_d_dim: int,
        x_c_dim: int,
        x_d_init: Optional[np.ndarray] = None,
        x_c_init: Optional[np.ndarray] = None,
        x_d_bounds: Optional[Tuple[Optional[np.ndarray], Optional[np.ndarray]]] = None,
        x_c_bounds: Optional[Tuple[Optional[np.ndarray], Optional[np.ndarray]]] = None,
    ) -> None:
        
        # Initialize discrete state
        if x_d_init is None:
            self.X_D = np.zeros(x_d_dim, dtype=np.int32)
        else:
            x_d_init = np.asarray(x_d_init)
            assert x_d_init.shape == (x_d_dim,)
            self.X_D = x_d_init.astype(np.int32)

        # Initialize continuous state
        if x_c_init is None:
            self.X_C = np.zeros(x_c_dim, dtype=np.float64)
        else:
            x_c_init = np.asarray(x_c_init, dtype=np.float64)
            assert x_c_init.shape == (x_c_dim,)
            self.X_C = x_c_init

        # Bounds: each is (min, max) tuple or (None, None)
        self.x_d_min, self.x_d_max = (None, None)
        self.x_c_min, self.x_c_max = (None, None)

        if x_d_bounds is not None:
            self.x_d_min = None if x_d_bounds[0] is None else np.asarray(x_d_bounds[0], dtype=np.int32)
            self.x_d_max = None if x_d_bounds[1] is None else np.asarray(x_d_bounds[1], dtype=np.int32)

        if x_c_bounds is not None:
            self.x_c_min = None if x_c_bounds[0] is None else np.asarray(x_c_bounds[0], dtype=float)
            self.x_c_max = None if x_c_bounds[1] is None else np.asarray(x_c_bounds[1], dtype=float)

    # ------------------------------------------------------------------
    # State accessors
    # ------------------------------------------------------------------

    def set_state(self, x_d: np.ndarray, x_c: np.ndarray) -> None:
        """Set the full state (discrete + continuous)."""
        x_d = np.asarray(x_d, dtype=np.int32)
        x_c = np.asarray(x_c, dtype=float)

        if x_d.shape != self.X_D.shape:
            raise ValueError(f"X_D shape mismatch: expected {self.X_D.shape}, got {x_d.shape}")
        if x_c.shape != self.X_C.shape:
            raise ValueError(f"X_C shape mismatch: expected {self.X_C.shape}, got {x_c.shape}")

        self.X_D = x_d
        self.X_C = x_c

    def get_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return copies of the current state (X_D, X_C)."""
        return self.X_D.copy(), self.X_C.copy()

    # ------------------------------------------------------------------
    # Bound checking logic (abstract - subclasses define policy)
    # ------------------------------------------------------------------

    @abstractmethod
    def check_bounds(self) -> Dict[str, Any]:
        """
        Check bounds for X_D and X_C.

        Subclasses must implement this method. They may:
        - Return a diagnostic report without modifying state
        - Perform in-place corrections (clip, wrap, etc.)
        - Raise an error on violation
        - Call the helper `_check_bounds_diagnostic()` for standard logic

        Returns
        -------
        report : Dict[str, Any]
            Implementation-dependent. Typically includes:
            {
                "x_d_ok": bool,
                "x_c_ok": bool,
                "x_d_violations": np.ndarray or None,
                "x_c_violations": np.ndarray or None,
            }

        Notes
        -----
        Subclasses may delegate to _check_bounds_diagnostic() for the standard
        violation detection, then implement custom handling.
        """
        raise NotImplementedError

    @abstractmethod
    def enforce_bounds(self) -> None:
        """
        Apply bound enforcement policy.

        The exact behavior (error, clip, wrap, mode switch, etc.) is
        defined by the subclass. This method may:
        - Do nothing if bounds are satisfied
        - Modify state to correct violations
        - Raise an exception
        - Trigger other side effects (logging, event handlers, etc.)

        Notes
        -----
        Subclasses may use _check_bounds_diagnostic() as a helper.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Bound checking helper (for subclass use)
    # ------------------------------------------------------------------

    def _check_bounds_diagnostic(self) -> Dict[str, Any]:
        """
        Helper method: Perform standard bounds violation detection. 

        *This naive method checks discrete states within bounds, not from the allowable set.
        *This naive method checks continuous states within bounds.

        Returns a diagnostic report without modifying state.
        Subclasses can call this to get violation information,
        then implement their own handling policy.

        Returns
        -------
        report : Dict[str, Any]
            {
                "x_d_ok": bool,
                "x_c_ok": bool,
                "x_d_violations": np.ndarray or None,  (boolean mask)
                "x_c_violations": np.ndarray or None,  (boolean mask)
            }
        """
        report: Dict[str, Any] = {
            "x_d_ok": True,
            "x_c_ok": True,
            "x_d_violations": None,
            "x_c_violations": None,
        }

        # ----- Discrete bounds -----
        if self.x_d_min is not None or self.x_d_max is not None:
            violations = np.zeros_like(self.X_D, dtype=bool)

            if self.x_d_min is not None:
                violations |= self.X_D < self.x_d_min
            if self.x_d_max is not None:
                violations |= self.X_D > self.x_d_max

            if np.any(violations):
                report["x_d_ok"] = False
                report["x_d_violations"] = violations

        # ----- Continuous bounds -----
        if self.x_c_min is not None or self.x_c_max is not None:
            violations = np.zeros_like(self.X_C, dtype=bool)

            if self.x_c_min is not None:
                violations |= self.X_C < self.x_c_min
            if self.x_c_max is not None:
                violations |= self.X_C > self.x_c_max

            if np.any(violations):
                report["x_c_ok"] = False
                report["x_c_violations"] = violations

        return report

    # ------------------------------------------------------------------
    # Hook for child classes
    # ------------------------------------------------------------------

    @abstractmethod
    def handle_bound_violation(self, report: Dict[str, Any]) -> None:
        """
        Define what happens when bounds are violated.

        `report` is the output of `check_bounds()`.

        Typical policies in subclasses:
            - Raise an error (hard fail)
            - Clip to nearest bound
            - Wrap-around (e.g. angles)
            - Trigger a mode switch in X_D
            - Log and continue

        This method must be implemented by subclasses.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # State dynamics hook
    # ------------------------------------------------------------------

    @abstractmethod
    def state_dynamics(
        self, u: Optional[np.ndarray] = None, dt: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the state dynamics update (one-step prediction).

        The interpretation of X_D and X_C is domain-specific.
        This method computes the next state given current state and control input.

        Parameters
        ----------
        u : np.ndarray, optional
            Control input (dimensions and interpretation depend on the system).
            If None, dynamics proceed with no external control.
        dt : float, optional
            Time step or integration interval (default 1.0).

        Returns
        -------
        x_d_next : np.ndarray
            Updated discrete state, shape (n_d,), dtype int32.
        x_c_next : np.ndarray
            Updated continuous state, shape (n_c,), dtype float64.

        Notes
        -----
        - Subclasses define the system equations (ODEs, transitions, etc.)
        - Return values are not automatically enforced; caller may use enforce_bounds()
        - Common patterns: F(x, u, dt) or x[k+1] = f(x[k], u[k])

        Examples
        --------
        A simple linear continuous system with discrete mode:
            x_d_next = x_d  # mode doesn't change
            x_c_next = x_c + dt * (A @ x_c + B @ u)
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Observation dynamics hook
    # ------------------------------------------------------------------

    @abstractmethod
    def observation_model(self, **kwargs) -> np.ndarray:
        """
        Generate or compute observations from the current state.

        The observation function models the relationship between
        internal state (X_D, X_C) and what is measurable (Y).

        From the problem formulation: Y is a subset of the full state X = [X_D, X_C]
        that obeys a parametric nonlinear relationship f(Y, theta) = 0.
        The parameter vector theta may depend on the discrete state X_D or other
        system-specific factors.

        Parameters
        ----------
        **kwargs
            System-specific keyword arguments (e.g. measurement noise, sensor params).

        Returns
        -------
        y : np.ndarray
            Observation vector (the observable subset of the full state).
            Y can include components from both X_D and X_C.
            Shape and dtype depend on the system.

        Notes
        -----
        - Only a subset of the full state may be observable (dim(Y) <= dim(X_D) + dim(X_C)).
        - The relationship f(Y, theta) = 0 is implicit; extraction is model-specific.
        - Subclasses may add measurement noise or sensor models here.
        - The observation can depend on both discrete and continuous state components.

        Examples
        --------
        Return a subset of continuous state (linear observation):
            return self.X_C[[0, 2]]  # observe components 0 and 2

        Mixed discrete-continuous observation:
            return np.concatenate([self.X_D, self.X_C[:2]])  # mode + first 2 continuous states

        Apply nonlinear measurement function:
            return np.sin(self.X_C)  # nonlinear transformation

        Mode-dependent observation with noise:
            theta = self.get_mode_parameters(self.X_D)
            y = self.X_C[:2]  # partially observable continuous state
            noise = kwargs.get('noise', 0.0)
            return y + noise * np.random.randn(y.shape)
        """
        raise NotImplementedError
