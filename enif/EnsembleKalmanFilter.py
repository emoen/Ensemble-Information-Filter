import numpy as np

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