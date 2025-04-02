import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class EnsembleInformationFilter:
    def __init__(self, n_ensemble=10, process_noise=0.1, measurement_noise=0.2):
        self.n_ensemble = n_ensemble
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

        # Initial state ensemble (position and velocity)
        self.state_ensemble = np.zeros((2, n_ensemble))
        self.state_ensemble[0, :] = np.random.normal(0, 1, n_ensemble)  # positions
        self.state_ensemble[1, :] = np.random.normal(0, 0.1, n_ensemble)  # velocities

        # Precision matrix (inverse covariance)
        self.precision_matrix = np.eye(2)  # Start with identity precision matrix

        # Time step
        self.dt = 0.1

    def predict(self):
        """Predict step: propagate ensemble forward in time"""
        F = np.array([[1, self.dt],
                      [0, 1]])  # State transition matrix

        for i in range(self.n_ensemble):
            self.state_ensemble[:, i] = F @ self.state_ensemble[:, i]
            process_noise = np.random.normal(0, self.process_noise, 2)
            self.state_ensemble[:, i] += process_noise

        # Update precision matrix (inverse covariance propagation)
        Q = self.process_noise * np.eye(2)  # Process noise covariance
        self.precision_matrix = np.linalg.inv(F @ np.linalg.inv(self.precision_matrix) @ F.T + Q)

    def update(self, measurement):
        """Update step: correct ensemble using measurement"""
        H = np.array([[1, 0]])  # Measurement matrix (only position)

        # Generate perturbed measurements
        perturbed_measurements = measurement + np.random.normal(0, self.measurement_noise, self.n_ensemble)

        # Update precision matrix with measurement information
        R = self.measurement_noise  # Measurement noise covariance
        H_precision_Ht = H @ np.linalg.inv(self.precision_matrix) @ H.T
        innovation_precision = 1 / (H_precision_Ht + R)
        kalman_gain = np.linalg.inv(self.precision_matrix) @ H.T * innovation_precision

        # Update ensemble members
        for i in range(self.n_ensemble):
            innovation = perturbed_measurements[i] - H @ self.state_ensemble[:, i]
            self.state_ensemble[:, i] += kalman_gain.flatten() * innovation

        # Update precision matrix
        self.precision_matrix += (1 / R) * (H.T @ H)

    def get_state_estimate(self):
        """Return mean of ensemble as state estimate"""
        return np.mean(self.state_ensemble, axis=1)


# Simulation parameters
n_steps = 100
true_state = np.array([-5.0, 2.0])  # Initial true state [position, velocity]
measurements = []
true_states = []
estimated_states = []

# Create EnIF instance
enif = EnsembleInformationFilter()

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
estimated_point = ax.scatter([], [], color='blue', s=100, label='Estimated State')

ax.legend()
ax.grid(True)


def animate(frame):
    # Simulate true state
    F = np.array([[1, enif.dt],
                  [0, 1]])
    true_state = F @ true_states[-1] if true_states else np.array([-5.0, 2.0])
    true_states.append(true_state)

    # Generate noisy measurement (only position)
    measurement = true_state[0] + np.random.normal(0, 0.2)
    measurements.append(measurement)

    # EnIF steps
    enif.predict()
    enif.update(measurement)
    estimated_state = enif.get_state_estimate()
    estimated_states.append(estimated_state)

    # Update plots
    ensemble_scatter.set_offsets(enif.state_ensemble.T)
    true_point.set_offsets([true_state[0], true_state[1]])
    measured_point.set_offsets([measurement, estimated_state[1]])
    estimated_point.set_offsets([estimated_state[0], estimated_state[1]])

    return ensemble_scatter, true_point, measured_point, estimated_point


# Create animation
anim = FuncAnimation(fig, animate, frames=n_steps, interval=50, blit=True)
plt.title('Ensemble Information Filter Tracking')
#plt.show()

# Optionally save the animation
anim.save('enif_tracking.gif', writer='pillow')