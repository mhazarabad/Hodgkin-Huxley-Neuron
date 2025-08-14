from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

class NeuronVisualizer:
    
    @staticmethod
    def plot_voltage_trace(t: np.ndarray, V: np.ndarray, I: Optional[np.ndarray] = None,
                          title: str = "Membrane Potential") -> Tuple[Figure, Axes]:
        """
        Plots membrane potential over time.
        
        Args:
            t: Time points (ms)
            V: Membrane potential (mV)
            I: Optional injected current (μA/cm²)
            title: Plot title
            
        Returns:
            Figure and Axes objects
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(t, V, label="V (mV)")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Membrane Potential (mV)")
        
        if I is not None:
            ax2 = ax.twinx()
            ax2.plot(t, I, '--', color='gray', label="I (μA/cm²)")
            ax2.set_ylabel("Current Density (μA/cm²)")
            
            # Combine legends
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        else:
            ax.legend(loc='upper right')
            
        ax.set_title(title)
        plt.tight_layout()
        return fig, ax
    
    @staticmethod
    def plot_phase_plane(V: np.ndarray, n: np.ndarray,
                        title: str = "Phase Plane") -> Tuple[Figure, Axes]:
        """
        Plots phase plane trajectory (V vs n).
        
        Args:
            V: Membrane potential (mV)
            n: Potassium activation variable
            title: Plot title
            
        Returns:
            Figure and Axes objects
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(V, n)
        ax.set_xlabel("Membrane Potential (mV)")
        ax.set_ylabel("K⁺ Activation (n)")
        ax.set_title(title)
        plt.tight_layout()
        return fig, ax
    
    @staticmethod
    def plot_gate_variables(t: np.ndarray, m: np.ndarray, h: np.ndarray, n: np.ndarray,
                          title: str = "Gate Variables") -> Tuple[Figure, Axes]:
        """
        Plots all gate variables over time.
        
        Args:
            t: Time points (ms)
            m: Na⁺ activation
            h: Na⁺ inactivation
            n: K⁺ activation
            title: Plot title
            
        Returns:
            Figure and Axes objects
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(t, m, label='m (Na⁺ activation)')
        ax.plot(t, h, label='h (Na⁺ inactivation)')
        ax.plot(t, n, label='n (K⁺ activation)')
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Gate Variable")
        ax.set_title(title)
        ax.legend()
        plt.tight_layout()
        return fig, ax
    
    @staticmethod
    def plot_currents(t: np.ndarray, I_Na: np.ndarray, I_K: np.ndarray, I_L: np.ndarray,
                     title: str = "Ionic Currents") -> Tuple[Figure, Axes]:
        """
        Plots all ionic currents over time.
        
        Args:
            t: Time points (ms)
            I_Na: Sodium current (μA/cm²)
            I_K: Potassium current (μA/cm²)
            I_L: Leak current (μA/cm²)
            title: Plot title
            
        Returns:
            Figure and Axes objects
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(t, I_Na, label='I_Na')
        ax.plot(t, I_K, label='I_K')
        ax.plot(t, I_L, label='I_L')
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Current Density (μA/cm²)")
        ax.set_title(title)
        ax.legend()
        plt.tight_layout()
        return fig, ax
    
    @staticmethod
    def plot_fi_curve(currents: np.ndarray, rates: np.ndarray,
                     title: str = "F-I Curve") -> Tuple[Figure, Axes]:
        """
        Plots the frequency-current relationship.
        
        Args:
            currents: Injected current amplitudes (μA/cm²)
            rates: Corresponding firing rates (Hz)
            title: Plot title
            
        Returns:
            Figure and Axes objects
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(currents, rates, 'o-')
        ax.set_xlabel("Injected Current (μA/cm²)")
        ax.set_ylabel("Firing Rate (Hz)")
        ax.set_title(title)
        plt.tight_layout()
        return fig, ax
