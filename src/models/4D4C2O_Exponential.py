"""
Example hybrid system: 4 Discrete modes, 4 Continuous states, 2 Observable states.

System Description:
    - X_D: 4 discrete states (modes/regimes)
    - X_C: 4 continuous states (hidden dynamics)
    - Y: 2 observable states that obey the exponential relationship:
        y1 = a * exp(-b * y2) + c
    - u: the system input is related to sys state/output in the following manner:
        y1 = u + N(bias, sigma^2)
    
    The parameters (a, b, c) are determined by the 6 hidden states (X_D, X_C).
    Each discrete mode can have different parameter mappings.
"""

from __future__ import annotations
from typing import Optional, Tuple, Dict, Any
import numpy as np
from model import HybridStateModel


class ExponentialSys_4D4C2O(HybridStateModel):
    """
    Hybrid system with 4 discrete modes, 4 continuous states, and 2 observable states.
    
    The observable states follow an exponential constraint: y1 = a * exp(-b * y2) + c,
    where parameters (a, b, c) depend on the current state configuration.
    """

    def __init__(
        self,
        x_d_init: Optional[np.ndarray] = None,
        x_c_init: Optional[np.ndarray] = None,
        x_d_bounds: Optional[Tuple[Optional[np.ndarray], Optional[np.ndarray]]] = None,
        x_c_bounds: Optional[Tuple[Optional[np.ndarray], Optional[np.ndarray]]] = None,
        noise_bias: float = 0.0,
        noise_sigma: float = 0.0,
    ) -> None:
        """
        Initialize the 4D-4C-2O exponential system.

        Parameters
        ----------
        x_d_init : np.ndarray, optional
            Initial discrete state (4 modes). Default: zeros.
        x_c_init : np.ndarray, optional
            Initial continuous state (4 continuous variables). Default: zeros.
        x_d_bounds : Tuple, optional
            Bounds for discrete state (min, max). Default: (0, 3) for 4 modes.
        x_c_bounds : Tuple, optional
            Bounds for continuous state. Default: (-10, 10).
        noise_bias : float, optional
            Mean of Gaussian noise applied to system input. Default: 0.0.
        noise_sigma : float, optional
            Standard deviation of Gaussian noise applied to system input. Default: 0.0.
        """
        # Set default bounds if not provided
        if x_d_bounds is None:
            x_d_bounds = (np.array([0, 0, 0, 0], dtype=np.int32), 
                          np.array([3, 3, 3, 3], dtype=np.int32))
        
        if x_c_bounds is None:
            x_c_bounds = (np.array([-10.0, -10.0, -10.0, -10.0], dtype=np.float64),
                          np.array([10.0, 10.0, 10.0, 10.0], dtype=np.float64))

        # Store noise characterization
        self.noise_bias = noise_bias
        self.noise_sigma = noise_sigma

        # Initialize parent class: 4 discrete, 4 continuous
        super().__init__(
            x_d_dim=4,
            x_c_dim=4,
            x_d_init=x_d_init,
            x_c_init=x_c_init,
            x_d_bounds=x_d_bounds,
            x_c_bounds=x_c_bounds,
        )

        self.a = 0.0
        self.b = 0.0
        self.c = 0.0

    # ------------------------------------------------------------------
    # Bound checking and enforcement
    # ------------------------------------------------------------------

    def check_bounds(self) -> Dict[str, Any]:
        """
        Check bounds for discrete and continuous states.
        
        Uses the standard diagnostic helper and returns the report. Bounds enforcing will happen in the next function.
        """
        return self._check_bounds_diagnostic()

    def enforce_bounds(self) -> None:
        """
        Enforce bounds by clipping state to valid range.
        
        Policy: Soft clipping to nearest bound.
        """
        report = self.check_bounds()
        
        if not report["x_d_ok"]:
            violations = report["x_d_violations"]
            self.X_D[violations & (self.X_D < self.x_d_min)] = self.x_d_min[violations & (self.X_D < self.x_d_min)]
            self.X_D[violations & (self.X_D > self.x_d_max)] = self.x_d_max[violations & (self.X_D > self.x_d_max)]
        
        if not report["x_c_ok"]:
            violations = report["x_c_violations"]
            self.X_C[violations & (self.X_C < self.x_c_min)] = self.x_c_min[violations & (self.X_C < self.x_c_min)]
            self.X_C[violations & (self.X_C > self.x_c_max)] = self.x_c_max[violations & (self.X_C > self.x_c_max)]

    def handle_bound_violation(self, report: Dict[str, Any]) -> None:
        """
        Handle bound violations by logging and clipping.
        
        Policy: Log the violation, then clip to bounds.
        """
        if not report["x_d_ok"]:
            print(f"[Warning] Discrete state violation: {report['x_d_violations']}")
        if not report["x_c_ok"]:
            print(f"[Warning] Continuous state violation: {report['x_c_violations']}")
        
        self.enforce_bounds()

    # ------------------------------------------------------------------
    # State dynamics
    # ------------------------------------------------------------------

    def state_dynamics(
        self, u: float = None, dt: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute state dynamics update (one-step prediction).

        Discrete state: Remains unchanged.
        Continuous state: x1 becomes noisy input, x2 is computed from exponential constraint.
        Other continuous states (x3, x4) remain unchanged.
        
        The input u is corrupted by Gaussian noise: u_noisy = u + N(noise_bias, noise_sigma^2)
        x1_next = u_noisy
        x2_next = ln((u_noisy - c) / a) / (-b)  [inverted exponential relationship]

        Parameters
        ----------
        u : float optional
            Control input (1-dim). If None, use zero input.
        dt : float, optional
            Time step (default 1.0).

        Returns
        -------
        x_d_next : np.ndarray
            Updated discrete state (unchanged).
        x_c_next : np.ndarray
            Updated continuous state (x1, x2 updated; x3, x4 unchanged).
        """
        if u is None:
            u = 0.0
        else:
            u = float(u)

        # Apply noise characterization to input
        # y1 = u + N(bias, sigma^2) as per system specification
        u_noisy = u
        if self.noise_sigma > 0:
            noise = self.noise_bias + self.noise_sigma * np.random.randn()
            u_noisy = u + noise

        # Discrete state doesn't change
        x_d_next = self.X_D.copy()

        # Get current parameters for the exponential relationship
        self.get_parameters()

        # Continuous state: x1 and x2 are updated, x3 and x4 remain unchanged
        x_c_next = self.X_C.copy()
        x_c_next[0] = u_noisy  # x1 becomes the noisy input
        
        # x2 is computed from the inverse of the exponential constraint: y1 = a * exp(-b * y2) + c
        # Solving for y2: y2 = ln((y1 - c) / a) / (-b)
        if self.a > 0 and (u_noisy - self.c) > 0:
            x_c_next[1] = np.log((u_noisy - self.c) / self.a) / (-self.b)
        else:
            # Invalid state progression: cannot satisfy exponential constraint
            x_c_next[1] = np.nan
            raise ValueError(
                f"Invalid state progression: Cannot satisfy exponential constraint y1 = a*exp(-b*y2) + c. "
                f"Parameters: a={self.a:.4f}, b={self.b:.4f}, c={self.c:.4f}, "
                f"u_noisy={u_noisy:.4f}. "
                f"Requires: a > 0 and (u_noisy - c) > 0, but got: a > 0: {self.a > 0}, "
                f"(u_noisy - c) > 0: {(u_noisy - self.c) > 0}"
            )

        return x_d_next.astype(np.int32), x_c_next.astype(np.float64)

    # ------------------------------------------------------------------
    # Observation model
    # ------------------------------------------------------------------

    def get_parameters(self) -> Tuple[float, float, float]:
        """
        Compute parameters (a, b, c) from hidden state.
        
        Mapping from hidden states (X_D[0], X_D[1], X_D[2]) to parameters:
            a = 2.0 + 0.5 * X_D[0]
            b = 0.5 + 0.1 * X_D[1]
            c = 0.2 + 0.05 * X_D[2]
        
        Returns
        -------
        a, b, c : float
            Exponential curve parameters for the relationship: y1 = a * exp(-b * y2) + c
        """
        self.a = 2.0 + 0.5 * self.X_D[0]
        self.b = 0.5 + 0.1 * self.X_D[1]
        self.c = 0.2 + 0.05 * self.X_D[2]
        
        return float(self.a), float(self.b), float(self.c)

    def observation_model(self, **kwargs) -> np.ndarray:
        """
        Generate 2 observable states from hidden state.

        The observable states are simply the first two continuous state components:
            y = [x_c[0], x_c[1]] = [y1, y2]
        
        where y1 and y2 obey the exponential constraint:
            y1 = a * exp(-b * y2) + c
        
        Parameters
        ----------
        **kwargs
            Optional 'noise' key for additive Gaussian measurement noise (std dev).

        Returns
        -------
        y : np.ndarray
            2-element observation vector [y1, y2] = X_C[0:2].
        """
        y = self.X_C[0:2].copy()

        return y
