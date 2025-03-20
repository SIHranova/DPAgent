#%%
 
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import inv

# Set random seed for reproducibility
np.random.seed(42)

### 1. Define LDS Parameters ###
T = 100  # Number of time steps
true_a = 0.9  # True transition coefficient
true_d = 0.5  # True bias term
sigma_q = 0.3  # Process noise std dev
sigma_r = 0.5  # Observation noise std dev

# Initialize the true state and observations
x_true = np.zeros(T)
y_obs = np.zeros(T)

# Initial state
x_true[0] = 1 # np.random.randn()

# Simulate LDS
for t in range(1, T):
    x_true[t] = true_a * x_true[t-1] + true_d + np.random.normal(0, sigma_q)
    y_obs[t] = x_true[t] + np.random.normal(0, sigma_r)

### 2. Kalman Filter Implementation ###
def kalman_filter(y, a, d, sigma_q, sigma_r):
    """ Run Kalman Filter for state inference """
    T = len(y)
    x_pred = np.zeros(T)
    P_pred = np.zeros(T)
    x_filt = np.zeros(T)
    P_filt = np.zeros(T)

    # Initial state estimate and covariance
    x_filt[0] = 0  # Assume zero initial mean
    P_filt[0] = 1  # Large initial variance

    for t in range(1, T):
        # Prediction step
        x_pred[t] = a * x_filt[t-1] + d
        P_pred[t] = a**2 * P_filt[t-1] + sigma_q**2

        # Kalman Gain
        K_t = P_pred[t] / (P_pred[t] + sigma_r**2)

        # Update step
        x_filt[t] = x_pred[t] + K_t * (y[t] - x_pred[t])
        P_filt[t] = (1 - K_t) * P_pred[t]

    return x_filt, P_filt


### 3. Kalman Smoother Implementation ###
def kalman_smoother(x_filt, P_filt, a):
    """ Run Kalman Smoother for backward smoothing """
    T = len(x_filt)
    x_smooth = np.copy(x_filt)
    P_smooth = np.copy(P_filt)

    for t in range(T-2, -1, -1):
        G_t = P_filt[t] * a / (a**2 * P_filt[t] + sigma_q**2)
        x_smooth[t] += G_t * (x_smooth[t+1] - a * x_filt[t] - d)
        P_smooth[t] += G_t**2 * (P_smooth[t+1] - a**2 * P_filt[t] - sigma_q**2)

    return x_smooth, P_smooth

### 4. EM Algorithm for Learning (Maximizing Likelihood) ###
def em_algorithm(y, num_iter=50,sigma_q_est=0.2, sigma_r_est=0.5, infer_variance=True):
    """ Expectation-Maximization algorithm to estimate a and d """
    # Initialize parameters
    a_est = np.random.uniform(0.5, 1.5)  # Random initial guess
    d_est = np.random.uniform(-1, 1)
    # sigma_q_est = 0.5
    # sigma_r_est = 0.5

    for _ in range(num_iter):
        # E-step: Kalman smoother to get expected state statistics
        x_filt, P_filt = kalman_filter(y, a_est, d_est, sigma_q_est, sigma_r_est)
        x_smooth, P_smooth = kalman_smoother(x_filt, P_filt, a_est)

        # Compute sufficient statistics
        Ex = x_smooth[:-1]
        Ex_next = x_smooth[1:]
        Exx = Ex @ Ex
        Ex_next_x = Ex @ Ex_next
        Ex_sum = np.sum(Ex)
        Ex_next_sum = np.sum(Ex_next)

        # M-step: Update parameters using closed-form expressions
        a_est = (Ex_next_x - d_est*Ex.sum()) / Exx
        d_est = (Ex_next_sum - a_est * Ex_sum) / (T-1)
        if infer_variance:
            sigma_q_est = np.sqrt(np.mean((Ex_next - a_est * Ex - d_est)**2))
            sigma_r_est = np.sqrt(np.mean((y - x_smooth)**2))

    return a_est, d_est, sigma_q_est, sigma_r_est, x_smooth

### 5. run em
### with inferring variances
a_est, d_est, sigma_q_est, sigma_r_est, x_est = em_algorithm(y_obs)#,sigma_q_est=np.sqrt(y_obs.var()/10), sigma_r_est=np.sqrt(y_obs.var()/2))
plt.figure(figsize=(12, 5))

# True vs estimated states
plt.plot(x_true, label="True State", linestyle="dashed", color="black")
plt.plot(x_est, label="Estimated State", linestyle="solid", color="red")
plt.hlines(y=x_true.mean(), xmin=0, xmax=100)
plt.scatter(range(T), y_obs, label="Observations", color="blue", s=10, alpha=0.6)

plt.xlabel("Time Step")
plt.ylabel("State / Observation")
plt.title("LDS State Estimation and Learning via EM")
plt.legend()
plt.show()

##### wITH TRUE VARIANCES
a_est, d_est, sigma_q_est, sigma_r_est, x_est = em_algorithm(y_obs, sigma_q_est = sigma_q, sigma_r_est = sigma_r)#plt.figure(figsize=(12, 5))

plt.plot(x_true, label="True State", linestyle="dashed", color="black")
plt.plot(x_est, label="Estimated State", linestyle="solid", color="red")
plt.hlines(y=x_true.mean(), xmin=0, xmax=100)
plt.scatter(range(T), y_obs, label="Observations", color="blue", s=10, alpha=0.6)

plt.xlabel("Time Step")
plt.ylabel("State / Observation")
plt.title("LDS State Estimation and Learning via EM")
plt.legend()
plt.show()
print(f"Learned Parameters: a = {a_est:.3f}, d = {d_est:.3f}, sigma_q = {sigma_q_est:.3f}, sigma_r = {sigma_r_est:.3f}")
print(f"True Parameters:    a = {true_a:.3f}, d = {true_d:.3f}, sigma_q = {sigma_q:.3f}, sigma_r = {sigma_r:.3f}")

##### with random variance initialization
a_est, d_est, sigma_q_est, sigma_r_est, x_est = em_algorithm(y_obs)#,sigma_q_est=sigma_q, sigma_r_est=sigma_r)
plt.figure(figsize=(12, 5))
plt.plot(x_true, label="True State", linestyle="dashed", color="black")
plt.plot(x_est, label="Estimated State", linestyle="solid", color="red")
plt.hlines(y=x_true.mean(), xmin=0, xmax=100)
plt.scatter(range(T), y_obs, label="Observations", color="blue", s=10, alpha=0.6)

plt.xlabel("Time Step")
plt.ylabel("State / Observation")
plt.title("LDS State Estimation and Learning via EM")
plt.legend()
plt.show()


##### With cleverly initialized variance
a_est, d_est, sigma_q_est, sigma_r_est, x_est = em_algorithm(y_obs,sigma_q_est=np.sqrt(y_obs.var()/10), sigma_r_est=np.sqrt(y_obs.var()/2))
plt.plot(x_true, label="True State", linestyle="dashed", color="black")
plt.plot(x_est, label="Estimated State", linestyle="solid", color="red")
plt.hlines(y=x_true.mean(), xmin=0, xmax=100)
plt.scatter(range(T), y_obs, label="Observations", color="blue", s=10, alpha=0.6)

plt.xlabel("Time Step")
plt.ylabel("State / Observation")
plt.title("LDS State Estimation and Learning via EM")
plt.legend()
plt.show()
print(f"Learned Parameters: a = {a_est:.3f}, d = {d_est:.3f}, sigma_q = {sigma_q_est:.3f}, sigma_r = {sigma_r_est:.3f}")
print(f"True Parameters:    a = {true_a:.3f}, d = {true_d:.3f}, sigma_q = {sigma_q:.3f}, sigma_r = {sigma_r:.3f}")


# Print learned parameters
print(f"Learned Parameters: a = {a_est:.3f}, d = {d_est:.3f}, sigma_q = {sigma_q_est:.3f}, sigma_r = {sigma_r_est:.3f}")
print(f"True Parameters:    a = {true_a:.3f}, d = {true_d:.3f}, sigma_q = {sigma_q:.3f}, sigma_r = {sigma_r:.3f}")


### Plot results of oonly filtering given true a and d
x_filt, P_filt = kalman_filter(y_obs, true_a, true_d, sigma_q, sigma_r)
plt.plot(x_true, label="True State", linestyle="dashed", color="black")
plt.scatter(range(T), y_obs, label="Observations", color="blue", s=10, alpha=0.6)
plt.plot(x_filt, label="Filtered X")
plt.ylim([-0.3,7])
plt.xlim([-1,101])

