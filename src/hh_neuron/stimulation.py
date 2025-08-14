from typing import Callable, Optional
import numpy as np

class CurrentInjector:
    
    @staticmethod
    def step(t_on: float, t_off: float, amplitude: float) -> Callable[[float], float]:
        """
        Creates a step current injection.
        
        Args:
            t_on: Start time (ms)
            t_off: End time (ms)
            amplitude: Current amplitude (μA/cm²)
            
        Returns:
            Function that computes current at any time t
        """
        def I(t: float) -> float:
            return amplitude if (t_on <= t <= t_off) else 0.0
        return I
    
    @staticmethod
    def ramp(t_on: float, t_off: float, max_amplitude: float) -> Callable[[float], float]:
        """
        Creates a ramp current injection.
        
        Args:
            t_on: Start time (ms)
            t_off: End time (ms)
            max_amplitude: Maximum current amplitude (μA/cm²)
            
        Returns:
            Function that computes current at any time t
        """
        def I(t: float) -> float:
            if t < t_on or t > t_off:
                return 0.0
            return max_amplitude * (t - t_on) / (t_off - t_on)
        return I
    
    @staticmethod
    def sine(frequency: float, amplitude: float, 
            t_on: Optional[float] = None, 
            t_off: Optional[float] = None) -> Callable[[float], float]:
        """
        Creates a sinusoidal current injection.
        
        Args:
            frequency: Oscillation frequency (Hz)
            amplitude: Peak current amplitude (μA/cm²)
            t_on: Optional start time (ms)
            t_off: Optional end time (ms)
            
        Returns:
            Function that computes current at any time t
        """
        def I(t: float) -> float:
            if t_on is not None and t < t_on:
                return 0.0
            if t_off is not None and t > t_off:
                return 0.0
            return amplitude * np.sin(2 * np.pi * frequency * t / 1000.0)
        return I
    
    @staticmethod
    def noise(mean: float, std: float, dt: float,
             t_on: Optional[float] = None,
             t_off: Optional[float] = None,
             seed: Optional[int] = None) -> Callable[[float], float]:
        """
        Creates a noisy current injection using Gaussian white noise.
        
        Args:
            mean: Mean current (μA/cm²)
            std: Standard deviation of current (μA/cm²)
            dt: Time step (ms)
            t_on: Optional start time (ms)
            t_off: Optional end time (ms)
            seed: Optional random seed
            
        Returns:
            Function that computes current at any time t
        """
        if seed is not None:
            np.random.seed(seed)
            
        def I(t: float) -> float:
            if t_on is not None and t < t_on:
                return 0.0
            if t_off is not None and t > t_off:
                return 0.0
            return mean + std * np.random.normal()
        return I
