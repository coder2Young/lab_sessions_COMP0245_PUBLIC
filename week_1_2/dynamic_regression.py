import numpy as np
import time
import os
import matplotlib.pyplot as plt
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, SinusoidalReference 
from scipy import stats

# TODO Compute Adjusted R-squared
def compute_adjusted_r2(tau_mes_all, tau_pred_all, num_params):
    n = len(tau_mes_all) # num of data
    tss = np.sum((tau_mes_all - np.mean(tau_mes_all)) ** 2) # total sum of sqrd
    rss = np.sum((tau_mes_all - tau_pred_all) ** 2) # residual sum of sqrd
    r_squared = 1 - (rss / tss)
    adjusted_rsqd = 1 - (1 - r_squared) * (n - 1) / (n - num_params - 1)
    return r_squared, adjusted_rsqd

# TODO Compute F-statistic
def compute_f_statistic(r_squared, n, num_params):
    f_stat = (r_squared / num_params) / ((1 - r_squared) / (n - num_params - 1))
    return f_stat

# TODO Compute confidence intervals for parameters
def compute_confidence_intervals(a, regressor_all, tau_mes_all, confidence=0.95):
    n, p = regressor_all.shape
    tau_pred_all = regressor_all @ a
    residuals = tau_mes_all - tau_pred_all
    residual_var = np.var(residuals, ddof=p)
    
    # Covariance matrix of the parameters
    cov_a = np.linalg.pinv(regressor_all.T @ regressor_all) * residual_var
    
    # Standard errors of the parameters
    se_a = np.sqrt(abs(np.diag(cov_a)))
    
    # t-distribution critical value for the confidence interval
    t_critical = stats.t.ppf((1 + confidence) / 2., n - p)
    
    # Confidence intervals for the parameters
    ci_lower = a - t_critical * se_a
    ci_upper = a + t_critical * se_a
    
    return ci_lower, ci_upper

# TODO Compute confidence intervals for predictions
def compute_prediction_confidence_intervals(a, regressor_all, tau_mes_all, confidence=0.95):
    n, p = regressor_all.shape
    tau_pred_all = regressor_all @ a
    residuals = tau_mes_all - tau_pred_all
    residual_var = np.var(residuals, ddof=p)
    
    # Covariance matrix of predictions
    se_pred = np.sqrt(abs(np.diag(regressor_all @ np.linalg.pinv(regressor_all.T @ regressor_all) @ regressor_all.T) * residual_var))
    
    # t-distribution critical value for the confidence interval
    t_critical = stats.t.ppf((1 + confidence) / 2., n - p)
    
    # Confidence intervals for the predictions
    ci_pred_lower = tau_pred_all - t_critical * se_pred
    ci_pred_upper = tau_pred_all + t_critical * se_pred
    
    return ci_pred_lower, ci_pred_upper

def main():
    # Configuration for the simulation
    conf_file_name = "pandaconfig.json"  # Configuration file for the robot
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    sim = pb.SimInterface(conf_file_name, conf_file_path_ext=cur_dir)  # Initialize simulation interface

    # Get active joint names from the simulation
    ext_names = sim.getNameActiveJoints()
    ext_names = np.expand_dims(np.array(ext_names), axis=0)  # Adjust the shape for compatibility

    source_names = ["pybullet"]  # Define the source for dynamic modeling

    # Create a dynamic model of the robot
    dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names, source_names, False, 0, cur_dir)
    num_joints = dyn_model.getNumberofActuatedJoints()

    # Print initial joint angles
    print(f"Initial joint angles: {sim.GetInitMotorAngles()}")

    # Sinusoidal reference
    # Specify different amplitude values for each joint
    amplitudes = [np.pi/4, np.pi/6, np.pi/4, np.pi/4, np.pi/4, np.pi/4, np.pi/4]  # Example amplitudes for joints
    # Specify different frequency values for each joint
    frequencies = [0.4, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4]  # Example frequencies for joints

    # Convert lists to NumPy arrays for easier manipulation in computations
    amplitude = np.array(amplitudes)
    frequency = np.array(frequencies)
    ref = SinusoidalReference(amplitude, frequency, sim.GetInitMotorAngles())  # Initialize the reference
    
    
    # Simulation parameters
    time_step = sim.GetTimeStep()
    current_time = 0
    max_time = 10  # seconds/ modified, was 10
    stable_time = 0.5 # estimated time the control system reach stable
    
    # Command and control loop
    cmd = MotorCommands()  # Initialize command structure for motors
    # PD controller gains
    kp = 1000
    kd = 100

    # Initialize data storage
    tau_mes_all = []
    regressor_all = []

    # Data collection loop
    while current_time < max_time:
        # Measure current state
        q_mes = sim.GetMotorAngles(0)
        qd_mes = sim.GetMotorVelocities(0)
        qdd_mes = sim.ComputeMotorAccelerationTMinusOne(0)
        
        # Compute sinusoidal reference trajectory
        q_d, qd_d = ref.get_values(current_time)  # Desired position and velocity
        
        # Control command
        cmd.tau_cmd = feedback_lin_ctrl(dyn_model, q_mes, qd_mes, q_d, qd_d, kp, kd)
        sim.Step(cmd, "torque")

        # Get measured torque
        tau_mes = sim.GetMotorTorques(0)
        if current_time >= stable_time: # drop the unstable data
            tau_mes_all.append(tau_mes)

        if dyn_model.visualizer: 
            for index in range(len(sim.bot)):  # Conditionally display the robot model
                q = sim.GetMotorAngles(index)
                dyn_model.DisplayModel(q)  # Update the display of the robot model

        # Exit logic with 'q' key
        keys = sim.GetPyBulletClient().getKeyboardEvents()
        qKey = ord('q')
        if qKey in keys and keys[qKey] and sim.GetPyBulletClient().KEY_WAS_TRIGGERED:
            break

        # TODO Compute regressor and store it
        if current_time >= stable_time: # drop the unstable data
            cur_regressor = dyn_model.ComputeDynamicRegressor(q_mes, qd_mes, qdd_mes)
            regressor_all.append(cur_regressor)
        current_time += time_step
        # Optional: print current time
        # print(f"Current time in seconds: {current_time:.2f}")

    # TODO After data collection, stack all the regressor and all the torquen and compute the parameters 'a'  using pseudoinverse
    regressor_all = np.vstack(regressor_all)
    tau_mes_all = np.hstack(tau_mes_all)

    pseudo_inv_regressor_all = np.linalg.pinv(regressor_all)
    a = pseudo_inv_regressor_all @ tau_mes_all
    tau_pred_all = regressor_all @ a

    # TODO compute the metrics for the linear model
    tau_err_all = tau_mes_all - tau_pred_all
    mse = np.mean(tau_err_all ** 2)
    print(f"Mean Squared Error (MSE) of torque predictions: {mse:.4f}")
    # Number of parameters (a)
    num_params = a.shape[0]

    r_squared, adjusted_rsqd = compute_adjusted_r2(tau_mes_all, tau_pred_all, num_params)
    f_stat = compute_f_statistic(r_squared, len(tau_mes_all), num_params)

    print(f"R-squared: {r_squared:.4f}")
    print(f"Adjusted R-squared: {adjusted_rsqd:.4f}")
    print(f"F-statistic: {f_stat:.4f}")

    # Compute confidence intervals for the parameters
    ci_params_lower, ci_params_upper = compute_confidence_intervals(a, regressor_all, tau_mes_all)
    #print(f"Confidence intervals for parameters: Lower {ci_params_lower}, Upper {ci_params_upper}")

    # Compute confidence intervals for predictions
    ci_pred_lower, ci_pred_upper = compute_prediction_confidence_intervals(a, regressor_all, tau_mes_all)
    #print(f"Confidence intervals for predictions: Lower {ci_pred_lower}, Upper {ci_pred_upper}")

    # TODO Plot the torque prediction error for each joint (optional)
    for i in range(num_joints):
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        
        axs[0].plot(tau_mes_all[i::num_joints], label='Measured Torque')
        axs[0].plot(tau_pred_all[i::num_joints], label='Predicted Torque')
        # Plot the confidence interval as a shaded area
        axs[0].fill_between(
            time_step,
            ci_pred_lower[i::num_joints],  # Lower bound of the confidence interval
            ci_pred_upper[i::num_joints],  # Upper bound of the confidence interval
            color='gray',  # Shade color
            alpha=0.9,  # Transparency
            label='Confidence Interval'
        )
        axs[0].set_title(f'Joint {i+1} Torque Prediction')
        axs[0].set_xlabel('Time Step')
        axs[0].set_ylabel('Torque')
        axs[0].legend()
        axs[0].grid()

        axs[1].plot(tau_err_all[i::num_joints], label='Torque Prediction Error')
        axs[1].set_title(f'Joint {i+1} Torque Prediction Error')
        axs[1].set_xlabel('Time Step')
        axs[1].set_ylabel('Error')
        axs[1].legend()
        axs[1].grid()

        # Adjust layout to prevent overlap
        plt.tight_layout()
        # Show the plots
        plt.show()

if __name__ == '__main__':
    main()
