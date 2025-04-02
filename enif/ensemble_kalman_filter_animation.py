import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from EnsembleKalmanFilter import EnsembleKalmanFilter 


# Simulation parameters
n_steps = 100
true_state = np.array([-15.0, 2.0])  # Initial true state [position, velocity]
measurements = []
true_states = []
estimated_states = []

# Create EnKF instance
enkf = EnsembleKalmanFilter()

# Set up the animation
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(-5, 15)
ax.set_ylim(-5, 15)
ax.set_xlabel('Position')
ax.set_ylabel('Velocity')

# Plot objects
ensemble_scatter = ax.scatter([], [], alpha=0.2, label='Ensemble Members')
true_point = ax.scatter([], [], color='red', s=100, label='True State')
measured_point = ax.scatter([], [], color='green', s=100, label='Measurement')
estimated_point = ax.scatter(
    [], [], color='blue', s=100, label='Estimated State')

ax.legend()
ax.grid(True)


def animate(frame):
    # Simulate true state
    F = np.array([[1, enkf.dt],
                  [0, 1]])
    true_state = F @ true_states[-1] if true_states else np.array([-5.0, 2.0])
    true_states.append(true_state)

    # Generate noisy measurement (only position)
    measurement = true_state[0] + np.random.normal(0, 0.2)
    measurements.append(measurement)

    # EnKF steps
    enkf.predict()
    enkf.update(measurement)
    estimated_state = enkf.get_state_estimate()
    estimated_states.append(estimated_state)

    # Update plots
    ensemble_scatter.set_offsets(enkf.state_ensemble.T)
    true_point.set_offsets([true_state[0], true_state[1]])
    measured_point.set_offsets([measurement, estimated_state[1]])
    estimated_point.set_offsets([estimated_state[0], estimated_state[1]])

    return ensemble_scatter, true_point, measured_point, estimated_point


# Create animation
anim = FuncAnimation(fig, animate, frames=n_steps,
                     interval=50, blit=True)
plt.title('Ensemble Kalman Filter Tracking')
#plt.show()

# Optionally save the animation
anim.save('enkf_tracking.gif', writer='pillow')
