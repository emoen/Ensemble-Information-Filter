import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from EnsembleKalmanFilter import EnsembleKalmanFilter 
from EnsembleInformationFilter import EnsembleInformationFilter


# Simulation parameters
n_steps = 40
true_state = np.array([-5.0, 5.0])  # Initial true state [position, velocity]
measurements = []
true_states = []
enkf_estimates = []
enif_estimates = []

# Create filter instances
enkf = EnsembleKalmanFilter()
enif = EnsembleInformationFilter()

# Set up the animation
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(-5, 15)
ax.set_ylim(-5, 15)
ax.set_xlabel('Position')
ax.set_ylabel('Velocity')

# Plot objects
ensemble_scatter_enkf = ax.scatter([], [], alpha=0.2, label='EnKF Ensemble')
ensemble_scatter_enif = ax.scatter([], [], alpha=0.2, label='EnIF Ensemble', color='orange')
true_point = ax.scatter([], [], color='red', s=100, label='True State')
measured_point = ax.scatter([], [], color='green', s=100, label='Measurement')
enkf_point = ax.scatter([], [], color='blue', s=100, label='EnKF Estimate')
enif_point = ax.scatter([], [], color='orange', s=100, label='EnIF Estimate')

ax.legend()
ax.grid(True)

# Add a text object to display the step counter
#step_counter = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=12, verticalalignment='top')


def animate(frame):
    global true_state
    F = np.array([[1, enkf.dt], [0, 1]])
    true_state = F @ true_state
    true_states.append(true_state)

    measurement = true_state[0] + np.random.normal(0, 0.2)
    measurements.append(measurement)

    # EnKF steps
    enkf.predict()
    enkf.update(measurement)
    enkf_estimate = enkf.get_state_estimate()
    enkf_estimates.append(enkf_estimate)

    # EnIF steps
    enif.predict()
    enif.update(measurement)
    enif_estimate = enif.get_state_estimate()
    enif_estimates.append(enif_estimate)

    # Update plots
    ensemble_scatter_enkf.set_offsets(enkf.state_ensemble.T)
    ensemble_scatter_enif.set_offsets(enif.state_ensemble.T)
    true_point.set_offsets([true_state[0], true_state[1]])
    measured_point.set_offsets([measurement, 0])
    enkf_point.set_offsets([enkf_estimate[0], enkf_estimate[1]])
    enif_point.set_offsets([enif_estimate[0], enif_estimate[1]])

    # Update the step counter
    #step_counter.set_text(f'Step: {frame + 1}/{n_steps}')

    return ensemble_scatter_enkf, ensemble_scatter_enif, true_point, measured_point, enkf_point, enif_point


# Create animation
anim = FuncAnimation(fig, animate, frames=n_steps, interval=500, blit=True)
plt.title('Comparison: EnKF vs EnIF')
#plt.show()

anim.save('enkf_vs_enif_tracking.gif', writer='pillow')