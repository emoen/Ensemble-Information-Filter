import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
from matplotlib.animation import FuncAnimation

# Set random seed for reproducibility
np.random.seed(42)

# True probability of heads
true_p = 0.7

# Generate 100 coin flips
n_flips = 100
flips = np.random.binomial(1, true_p, n_flips)

# Initial parameters for Beta distribution
alpha_prior = 1
beta_prior = 1

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 6))
x = np.linspace(0, 1, 200)

# Initialize line objects
prior_line, = ax.plot(x, beta.pdf(x, alpha_prior, beta_prior),
                      'b--', label='Prior', alpha=0.5)
posterior_line, = ax.plot([], [], 'r-', label='Posterior')
true_p_line = ax.axvline(x=true_p, color='g', linestyle='--',
                         label='True Probability')

# Set up the plot
ax.set_xlim(0, 1)
ax.set_ylim(0, 10)
ax.set_xlabel('Probability of Heads')
ax.set_ylabel('Density')
ax.set_title('Bayesian Update of Coin Bias')
ax.legend()

# Text display for counts
text = ax.text(0.02, 9, '', transform=ax.transData)


def update(frame):
    # Update parameters
    alpha = alpha_prior + np.sum(flips[:frame+1])
    beta_param = beta_prior + frame + 1 - np.sum(flips[:frame+1])

    # Update posterior line
    posterior = beta.pdf(x, alpha, beta_param)
    posterior_line.set_data(x, posterior)

    # Update text
    heads = np.sum(flips[:frame+1])
    tails = frame + 1 - heads
    text.set_text(f'Flips: {frame+1}\nHeads: {heads}\nTails: {tails}\n' +
                  f'Posterior: Beta({alpha:.0f}, {beta_param:.0f})')

    return posterior_line, text


# Create animation
anim = FuncAnimation(fig, update, frames=n_flips,
                     interval=50, blit=True)

plt.show()

# Optionally save the animation (uncomment to save)
# anim.save('bayesian_update.gif', writer='pillow')
