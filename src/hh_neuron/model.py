"""
Core implementation of the Hodgkin-Huxley neuron model.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple
import numpy as np
from numpy.typing import NDArray

@dataclass
class HHParameters:
    """Parameters for the Hodgkin-Huxley model."""
    
    # Membrane capacitance (μF/cm²)
    C_m: float = 1.0
    
    # Maximum conductances (mS/cm²)
    g_Na: float = 120.0  # Sodium
    g_K: float = 36.0    # Potassium
    g_L: float = 0.3     # Leak
    
    # Reversal potentials (mV)
    E_Na: float = 50.0   # Sodium
    E_K: float = -77.0   # Potassium
    E_L: float = -54.387 # Leak
    
    # Temperature parameters
    T: float = 6.3       # Temperature (°C)
    Q10_gates: float = 3.0  # Q10 for gate kinetics
    Q10_cond: float = 1.0   # Q10 for conductances

    def get_temperature_scale(self, T0: float = 6.3) -> Tuple[float, float]:
        """Calculates temperature scaling factors."""
        phi_gates = self.Q10_gates ** ((self.T - T0) / 10.0)
        phi_cond = self.Q10_cond ** ((self.T - T0) / 10.0)
        return phi_gates, phi_cond


class HHNeuron:
    """Hodgkin-Huxley neuron model implementation."""
    
    def __init__(self, parameters: Optional[HHParameters] = None):
        self.p = parameters or HHParameters()
    
    @staticmethod
    def _alpha_n(V: float) -> float:
        """Rate constant α_n for K⁺ activation."""
        return 0.01 * (V + 55.0) / (1.0 - np.exp(-(V + 55.0) / 10.0))
    
    @staticmethod
    def _beta_n(V: float) -> float:
        """Rate constant β_n for K⁺ activation."""
        return 0.125 * np.exp(-(V + 65.0) / 80.0)
    
    @staticmethod
    def _alpha_m(V: float) -> float:
        """Rate constant α_m for Na⁺ activation."""
        return 0.1 * (V + 40.0) / (1.0 - np.exp(-(V + 40.0) / 10.0))
    
    @staticmethod
    def _beta_m(V: float) -> float:
        """Rate constant β_m for Na⁺ activation."""
        return 4.0 * np.exp(-(V + 65.0) / 18.0)
    
    @staticmethod
    def _alpha_h(V: float) -> float:
        """Rate constant α_h for Na⁺ inactivation."""
        return 0.07 * np.exp(-(V + 65.0) / 20.0)
    
    @staticmethod
    def _beta_h(V: float) -> float:
        """Rate constant β_h for Na⁺ inactivation."""
        return 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))
    
    def steady_state_gates(self, V: float) -> Dict[str, Tuple[float, float]]:
        """
        Calculate steady-state values and time constants for all gates.
        
        Returns:
            Dict with keys 'm', 'h', 'n', each containing (x_inf, tau_x)
        """
        results = {}
        
        # Na⁺ activation (m)
        am, bm = self._alpha_m(V), self._beta_m(V)
        tau_m = 1.0 / (am + bm)
        m_inf = am * tau_m
        results['m'] = (m_inf, tau_m)
        
        # Na⁺ inactivation (h)
        ah, bh = self._alpha_h(V), self._beta_h(V)
        tau_h = 1.0 / (ah + bh)
        h_inf = ah * tau_h
        results['h'] = (h_inf, tau_h)
        
        # K⁺ activation (n)
        an, bn = self._alpha_n(V), self._beta_n(V)
        tau_n = 1.0 / (an + bn)
        n_inf = an * tau_n
        results['n'] = (n_inf, tau_n)
        
        return results
    
    def initialize_states(self, V0: float = -65.0) -> Dict[str, float]:
        """Initializes gate states to their steady-state values at V0."""
        states = self.steady_state_gates(V0)
        return {
            'V': V0,
            'm': states['m'][0],
            'h': states['h'][0],
            'n': states['n'][0]
        }
    
    def compute_currents(self, V: float, m: float, h: float, n: float) -> Dict[str, float]:
        """Computes ionic currents given voltage and gate states."""
        _, phi_cond = self.p.get_temperature_scale()
        
        # Compute conductances
        g_Na = phi_cond * self.p.g_Na * (m**3) * h
        g_K = phi_cond * self.p.g_K * (n**4)
        g_L = self.p.g_L
        
        # Compute currents
        I_Na = g_Na * (V - self.p.E_Na)
        I_K = g_K * (V - self.p.E_K)
        I_L = g_L * (V - self.p.E_L)
        
        return {
            'I_Na': I_Na,
            'I_K': I_K,
            'I_L': I_L,
            'I_total': I_Na + I_K + I_L
        }
    
    def _dxdt(self, t: float, y: NDArray, I_inj: Callable[[float], float]) -> NDArray:
        """
        Computes time derivatives for all state variables.
        
        Args:
            t: Time (ms)
            y: State vector [V, m, h, n]
            I_inj: Injected current function (μA/cm²)
            
        Returns:
            Array of derivatives [dV/dt, dm/dt, dh/dt, dn/dt]
        """
        V, m, h, n = y
        phi_gates, _ = self.p.get_temperature_scale()
        
        # Gate dynamics
        am, bm = self._alpha_m(V), self._beta_m(V)
        ah, bh = self._alpha_h(V), self._beta_h(V)
        an, bn = self._alpha_n(V), self._beta_n(V)
        
        dmdt = phi_gates * (am * (1 - m) - bm * m)
        dhdt = phi_gates * (ah * (1 - h) - bh * h)
        dndt = phi_gates * (an * (1 - n) - bn * n)
        
        # Membrane potential dynamics
        currents = self.compute_currents(V, m, h, n)
        dVdt = (-currents['I_total'] + I_inj(t)) / self.p.C_m
        
        return np.array([dVdt, dmdt, dhdt, dndt])
    
    def simulate(self, 
                tmax: float,
                dt: float,
                I_inj: Callable[[float], float],
                V0: float = -65.0,
                method: str = "euler") -> Dict[str, NDArray]:
        """
        Simulates the neuron model.
        
        Args:
            tmax: Simulation duration (ms)
            dt: Time step (ms)
            I_inj: Injected current function (μA/cm²)
            V0: Initial membrane potential (mV)
            method: Integration method ('euler' or 'rk4')
            
        Returns:
            Dictionary containing simulation results
        """
        # Initialize time points
        nsteps = int(np.round(tmax/dt)) + 1
        t = np.linspace(0.0, tmax, nsteps)
        
        # Initialize state variables
        states = self.initialize_states(V0)
        V = np.empty(nsteps)
        m = np.empty_like(V)
        h = np.empty_like(V)
        n = np.empty_like(V)
        
        V[0] = states['V']
        m[0] = states['m']
        h[0] = states['h']
        n[0] = states['n']
        
        # Initialize current records
        I_Na = np.empty_like(V)
        I_K = np.empty_like(V)
        I_L = np.empty_like(V)
        I = np.empty_like(V)
        
        # Initial currents
        currents = self.compute_currents(V[0], m[0], h[0], n[0])
        I_Na[0] = currents['I_Na']
        I_K[0] = currents['I_K']
        I_L[0] = currents['I_L']
        I[0] = I_inj(0.0)
        
        def step_euler(y: NDArray, ti: float, dt: float) -> NDArray:
            """Euler integration step."""
            return y + dt * self._dxdt(ti, y, I_inj)
        
        def step_rk4(y: NDArray, ti: float, dt: float) -> NDArray:
            """4th-order Runge-Kutta integration step."""
            k1 = self._dxdt(ti, y, I_inj)
            k2 = self._dxdt(ti + dt/2, y + dt*k1/2, I_inj)
            k3 = self._dxdt(ti + dt/2, y + dt*k2/2, I_inj)
            k4 = self._dxdt(ti + dt, y + dt*k3, I_inj)
            return y + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
        
        # Choose integration method
        step_func = step_rk4 if method == "rk4" else step_euler
        
        # Run simulation
        y = np.array([V[0], m[0], h[0], n[0]], dtype=float)
        for i in range(nsteps-1):
            # Update state
            y = step_func(y, t[i], dt)
            V[i+1], m[i+1], h[i+1], n[i+1] = y
            
            # Record currents
            currents = self.compute_currents(V[i+1], m[i+1], h[i+1], n[i+1])
            I_Na[i+1] = currents['I_Na']
            I_K[i+1] = currents['I_K']
            I_L[i+1] = currents['I_L']
            I[i+1] = I_inj(t[i+1])
        
        return {
            't': t,
            'V': V,
            'm': m,
            'h': h,
            'n': n,
            'I_Na': I_Na,
            'I_K': I_K,
            'I_L': I_L,
            'I': I
        }
