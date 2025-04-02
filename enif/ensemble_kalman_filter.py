import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class EnsembleKalmanFilter:
    def __init__(self, n_ensemble=10, process_noise=0.1, measurement_noise=0.2):
        self.n_ensemble = n_ensemble
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

        # Initial state ensemble (position and velocity)
        self.state_ensemble = np.zeros((2, n_ensemble))
        # Initialize with random positions and velocities
        self.state_ensemble[0, :] = np.random.normal(
            0, 1, n_ensemble)  # positions
        self.state_ensemble[1, :] = np.random.normal(
            0, 0.1, n_ensemble)  # velocities

        # Time step
        self.dt = 0.1

    def predict(self):
        """Predict step: propagate ensemble forward in time"""
        # State transition matrix for constant velocity model
        F = np.array([[1, self.dt],
                     [0, 1]])

        # Propagate each ensemble member
        for i in range(self.n_ensemble):
            # Apply state transition
            self.state_ensemble[:, i] = F @ self.state_ensemble[:, i]

            # Add process noise
            process_noise = np.random.normal(0, self.process_noise, 2)
            self.state_ensemble[:, i] += process_noise

    def update(self, measurement):
        """Update step: correct ensemble using measurement"""
        # Measurement matrix (we only measure position)
        H = np.array([[1, 0]])

        # Generate perturbed measurements
        perturbed_measurements = measurement + \
            np.random.normal(0, self.measurement_noise, self.n_ensemble)

        # Calculate Kalman gain
        PHt = np.cov(self.state_ensemble) @ H.T
        S = H @ PHt + self.measurement_noise
        K = PHt / S

        # Update each ensemble member
        for i in range(self.n_ensemble):
            # Innovation: difference between perturbed measurement and predicted measurement
            innovation = perturbed_measurements[i] - \
                H @ self.state_ensemble[:, i]

            # Update state
            self.state_ensemble[:, i] += K.flatten() * innovation

    def get_state_estimate(self):
        """Return mean of ensemble as state estimate"""
        return np.mean(self.state_ensemble, axis=1)


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
