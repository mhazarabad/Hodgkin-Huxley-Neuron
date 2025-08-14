import numpy as np
from src.hh_neuron.model import HHNeuron, HHParameters
from src.hh_neuron.stimulation import CurrentInjector
from src.hh_neuron.analysis import SpikeAnalyzer
from src.hh_neuron.visualization import NeuronVisualizer
import matplotlib.pyplot as plt

def main():
    # Create neuron
    params = HHParameters(T=6.3)  # Original temperature
    neuron = HHNeuron(params)
    
    # Create step current stimulus
    I = CurrentInjector.step(t_on=10.0, t_off=90.0, amplitude=10.0)
    
    # Simulate
    res = neuron.simulate(tmax=100.0, dt=0.02, I_inj=I, V0=-65.0, method="rk4")
    
    # Analyze spikes
    analyzer = SpikeAnalyzer()
    spikes = analyzer.detect_spikes(res['t'], res['V'])
    
    # Visualize results
    viz = NeuronVisualizer()
    
    # Plot membrane potential
    viz.plot_voltage_trace(
        res['t'], res['V'], res['I'],title="HH Neuron - Step Current Response"
    )
    
    # Plot gate variables
    viz.plot_gate_variables(res['t'], res['m'], res['h'], res['n'])
    
    # Plot ionic currents
    viz.plot_currents(res['t'], res['I_Na'], res['I_K'], res['I_L'])
    
    # Generate and plot F-I curve
    currents = np.linspace(0.0, 20.0, 21)
    currents, rates = analyzer.get_fi_curve(neuron, currents)
    viz.plot_fi_curve(currents, rates)
    plt.show()

if __name__ == "__main__":
    main()
