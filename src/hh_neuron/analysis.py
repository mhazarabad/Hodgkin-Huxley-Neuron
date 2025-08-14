from typing import Dict, List, Optional, Tuple
import numpy as np
from numpy.typing import NDArray

class SpikeAnalyzer:
    """Utility class for analyzing spike trains and neuron dynamics."""
    
    def __init__(self, threshold: float = 0.0, refractory_period: float = 2.0):
        """
        
        Args:
            threshold: Voltage threshold for spike detection (mV)
            refractory_period: Minimum time between spikes (ms)
        """
        self.threshold = threshold
        self.refractory_period = refractory_period
    
    def detect_spikes(self, t: NDArray, V: NDArray) -> NDArray:
        """
        Detects action potentials in voltage trace.
        
        Args:
            t: Time points (ms)
            V: Membrane potential (mV)
            
        Returns:
            Array of spike time indices
        """
        # Find threshold crossings
        above = V > self.threshold
        crossings = np.where(np.logical_and(~above[:-1], above[1:]))[0] + 1
        
        # Find peaks after crossings
        spikes = []
        last_spike_time = -np.inf
        
        for idx in crossings:
            # Look for local maximum within 2ms window
            window = (t >= t[idx]) & (t <= t[idx] + 2.0)
            if not np.any(window):
                continue
                
            peak_idx = idx + np.argmax(V[window])
            
            # Enforce refractory period
            if t[peak_idx] - last_spike_time >= self.refractory_period:
                spikes.append(peak_idx)
                last_spike_time = t[peak_idx]
        
        return np.array(spikes, dtype=int)
    
    def get_spike_properties(self, t: NDArray, V: NDArray, 
                           spike_idx: NDArray) -> List[Dict[str, float]]:
        """
        Extracts properties of each spike.
        
        Args:
            t: Time points (ms)
            V: Membrane potential (mV)
            spike_idx: Indices of spike peaks
            
        Returns:
            List of dictionaries containing spike properties
        """
        properties = []
        
        for idx in spike_idx:
            # Find start of spike (last crossing of resting potential before peak)
            rest_cross = np.where(V[:idx] < -65.0)[0]
            start_idx = rest_cross[-1] if len(rest_cross) > 0 else 0
            
            # Find end of spike (first crossing of resting potential after peak)
            rest_cross = np.where(V[idx:] < -65.0)[0]
            end_idx = idx + rest_cross[0] if len(rest_cross) > 0 else len(V)-1
            
            props = {
                'time': t[idx],
                'peak': V[idx],
                'width': t[end_idx] - t[start_idx],
                'rise_time': t[idx] - t[start_idx],
                'fall_time': t[end_idx] - t[idx],
                'threshold': V[start_idx]
            }
            properties.append(props)
        
        return properties
    
    def get_firing_rate(self, t: NDArray, spike_idx: NDArray, 
                       window_ms: Optional[float] = None) -> float:
        """
        Calculates firing rate.
        
        Args:
            t: Time points (ms)
            spike_idx: Indices of spike peaks
            window_ms: Optional time window for rate calculation (ms)
            
        Returns:
            Firing rate in Hz
        """
        if len(spike_idx) < 2:
            return 0.0 if len(spike_idx) == 0 else 1000.0 / window_ms
            
        if window_ms is None:
            window_ms = t[spike_idx[-1]] - t[spike_idx[0]]
            
        return (len(spike_idx) - 1) * 1000.0 / window_ms
    
    def get_isi_stats(self, t: NDArray, spike_idx: NDArray) -> Dict[str, float]:
        """
        Calculates inter-spike interval statistics.
        
        Args:
            t: Time points (ms)
            spike_idx: Indices of spike peaks
            
        Returns:
            Dictionary with ISI statistics
        """
        if len(spike_idx) < 2:
            return {
                'mean': float('nan'),
                'std': float('nan'),
                'cv': float('nan'),
                'min': float('nan'),
                'max': float('nan')
            }
            
        isi = np.diff(t[spike_idx])
        
        return {
            'mean': float(np.mean(isi)),
            'std': float(np.std(isi)),
            'cv': float(np.std(isi) / np.mean(isi)),
            'min': float(np.min(isi)),
            'max': float(np.max(isi))
        }
    
    def get_fi_curve(self, neuron, current_range: NDArray,
                    t_on: float = 50.0, duration: float = 400.0,
                    tmax: float = 500.0, dt: float = 0.02) -> Tuple[NDArray, NDArray]:
        """
        Generates frequency-current (F-I) curve.
        
        Args:
            neuron: HHNeuron instance
            current_range: Array of current amplitudes (μA/cm²)
            t_on: Stimulus onset time (ms)
            duration: Stimulus duration (ms)
            tmax: Total simulation time (ms)
            dt: Time step (ms)
            
        Returns:
            Tuple of (currents, firing_rates)
        """
        from .stimulation import CurrentInjector
        
        rates = []
        for I_amp in current_range:
            # Run simulation
            I = CurrentInjector.step(t_on, t_on + duration, I_amp)
            res = neuron.simulate(tmax, dt, I, method="rk4")
            
            # Detect spikes during stimulus
            spikes = self.detect_spikes(res['t'], res['V'])
            spike_times = res['t'][spikes]
            in_stim = spike_times[(spike_times >= t_on) & 
                                (spike_times <= t_on + duration)]
            
            # Calculate rate
            rate = self.get_firing_rate(res['t'], in_stim, window_ms=duration)
            rates.append(rate)
            
        return current_range, np.array(rates)
