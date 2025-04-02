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
        self.state_ensemble[0, :] = np.random.normal(0, 1, n_ensemble)  # positions
        self.state_ensemble[1, :] = np.random.normal(0, 0.1, n_ensemble)  # velocities

        self.dt = 0.1

    def predict(self):
        F = np.array([[1, self.dt], [0, 1]])
        for i in range(self.n_ensemble):
            self.state_ensemble[:, i] = F @ self.state_ensemble[:, i]
            process_noise = np.random.normal(0, self.process_noise, 2)
            self.state_ensemble[:, i] += process_noise

    def update(self, measurement):
        H = np.array([[1, 0]])
        perturbed_measurements = measurement + np.random.normal(0, self.measurement_noise, self.n_ensemble)
        PHt = np.cov(self.state_ensemble) @ H.T
        S = H @ PHt + self.measurement_noise
        K = PHt / S
        for i in range(self.n_ensemble):
            innovation = perturbed_measurements[i] - H @ self.state_ensemble[:, i]
            self.state_ensemble[:, i] += K.flatten() * innovation

    def get_state_estimate(self):
        return np.mean(self.state_ensemble, axis=1)


class EnsembleInformationFilter:
    def __init__(self, n_ensemble=10, process_noise=0.1, measurement_noise=0.2):
        self.n_ensemble = n_ensemble
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

        # Initial state ensemble (position and velocity)
        self.state_ensemble = np.zeros((2, n_ensemble))
        self.state_ensemble[0, :] = np.random.normal(0, 1, n_ensemble)  # positions
        self.state_ensemble[1, :] = np.random.normal(0, 0.1, n_ensemble)  # velocities

        self.precision_matrix = np.eye(2)
        self.dt = 0.1

    def predict(self):
        F = np.array([[1, self.dt], [0, 1]])
        for i in range(self.n_ensemble):
            self.state_ensemble[:, i] = F @ self.state_ensemble[:, i]
            process_noise = np.random.normal(0, self.process_noise, 2)
            self.state_ensemble[:, i] += process_noise
        Q = self.process_noise * np.eye(2)
        self.precision_matrix = np.linalg.inv(F @ np.linalg.inv(self.precision_matrix) @ F.T + Q)

    def update(self, measurement):
        H = np.array([[1, 0]])
        perturbed_measurements = measurement + np.random.normal(0, self.measurement_noise, self.n_ensemble)
        R = self.measurement_noise
        H_precision_Ht = H @ np.linalg.inv(self.precision_matrix) @ H.T
        innovation_precision = 1 / (H_precision_Ht + R)
        kalman_gain = np.linalg.inv(self.precision_matrix) @ H.T * innovation_precision
        for i in range(self.n_ensemble):
            innovation = perturbed_measurements[i] - H @ self.state_ensemble[:, i]
            self.state_ensemble[:, i] += kalman_gain.flatten() * innovation
        self.precision_matrix += (1 / R) * (H.T @ H)

    def get_state_estimate(self):
        return np.mean(self.state_ensemble, axis=1)


# Simulation parameters
n_steps = 100
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

    return ensemble_scatter_enkf, ensemble_scatter_enif, true_point, measured_point, enkf_point, enif_point


# Create animation
anim = FuncAnimation(fig, animate, frames=n_steps, interval=500, blit=True)
plt.title('Comparison: EnKF vs EnIF')
plt.show()

anim.save('enkf_vs_enif_tracking.gif', writer='pillow')