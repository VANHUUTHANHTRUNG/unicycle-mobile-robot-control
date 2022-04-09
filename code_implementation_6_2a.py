import numpy as np
import matplotlib.pyplot as plt
from visualize_mobile_robot import sim_mobile_robot

# Constants and Settings
Ts = 0.01  # Update simulation every 10ms
t_max = np.pi  # total simulation duration in seconds

# Set initial state
init_state = np.array([-1., -0.5, np.pi / 8])  # px, py, theta
IS_SHOWING_2DVISUALIZATION = True

# Define Field size for plotting (should be in tuple)
field_x = (-2.5, 2.5)
field_y = (-2, 2)

# Robot physical characteristics
L = .21
R = .1
w_max = 10

# Params 2a    ###  Conditions
k_rho = 2  # k_rho > 0
k_alpha = 4  # k_alpha > k_rho
k_beta = -2  # k_beta < 0


# MAIN SIMULATION COMPUTATION
# ---------------------------------------------------------------------
def simulate_control():
    sim_iter = round(t_max / Ts)  # Total Step for simulation

    # Initialize robot's state (Single Integrator)
    robot_state = init_state.copy()  # numpy array for [px, py, theta]
    desired_state = np.array([0, 0, 0])  # numpy array for goal / the desired [px, py, theta]
    current_input = np.array([0., 0.])  # initial numpy array for [v, w]i

    # Store the value that needed for plotting: total step number x data length
    state_history = np.zeros((sim_iter, len(robot_state)))
    goal_history = np.zeros((sim_iter, len(desired_state)))
    input_history = np.zeros((sim_iter, len(current_input)))
    w_left_history = np.zeros((sim_iter, 1))
    w_right_history = np.zeros((sim_iter, 1))

    # Polar coordinates
    polar_state = np.array([.0, .0, .0])
    polar_state_history = np.zeros((sim_iter, len(robot_state)))

    if IS_SHOWING_2DVISUALIZATION:  # Initialize Plot
        sim_visualizer = sim_mobile_robot('unicycle')  # Omnidirectional Icon
        # sim_visualizer = sim_mobile_robot( 'unicycle' ) # Unicycle Icon
        sim_visualizer.set_field(field_x, field_y)  # set plot area
        sim_visualizer.show_goal(desired_state)

    for it in range(sim_iter):
        # record current state at time-step t
        state_history[it] = robot_state
        goal_history[it] = desired_state
        polar_state_history[it] = polar_state

        # IMPLEMENTATION OF CONTROLLER
        # ------------------------------------------------------------
        # Compute the control input
        rho = np.sqrt(robot_state[0] ** 2 + robot_state[1] ** 2)
        alpha = - robot_state[2] + np.arctan2(-robot_state[1], -robot_state[0])
        beta = - robot_state[2] - alpha

        # Error as polar state:  [rho, alpha, beta].T
        e = np.array([rho,
                      alpha,
                      beta])
        polar_state = e

        # # Linearize the closed-loop system at (0,0,0)
        # K = np.array([[-k_rho, 0, 0],
        #               [0, -(k_alpha - k_rho), -k_beta],
        #               [0, -k_rho, 0]])
        #
        # polar_control = K @ e

        # Coordinate transformation polar -> cartesian
        v = k_rho * rho
        w = k_alpha * alpha + k_beta * beta

        # # Consider rotational limit
        w_left = (2 * v - w * L) / (2 * R)
        w_right = (2 * v + w * L) / (2 * R)

        w_left = np.min([w_left, w_max]) if w_left >= 0 else np.max([w_left, -w_max])
        w_right = np.min([w_right, w_max]) if w_right >= 0 else np.max([w_right, -w_max])

        # Update feasible input
        current_input[0] = (w_left + w_right) * R / 2
        current_input[1] = (w_right - w_left) * R / L

        # ------------------------------------------------------------

        # record the computed input at time-step t
        input_history[it] = current_input
        polar_state_history[it] = polar_state
        w_left_history[it] = w_left
        w_right_history[it] = w_right

        if IS_SHOWING_2DVISUALIZATION:  # Update Plot
            sim_visualizer.update_time_stamp(it * Ts)
            sim_visualizer.update_goal(desired_state)
            sim_visualizer.update_trajectory(state_history[:it + 1])  # up to the latest data

        # --------------------------------------------------------------------------------
        # Update new state of the robot at time-step t+1
        # using discrete-time model of single integrator dynamics for omnidirectional robot
        theta = robot_state[2]
        B = np.array([[np.cos(theta), 0], [np.sin(theta), 0], [0, 1]])
        robot_state = robot_state + Ts * (B @ current_input)  # will be used in the next iteration
        robot_state[2] = ((robot_state[2] + np.pi) % (2 * np.pi)) - np.pi  # ensure theta within [-pi pi]

        # Update desired state if we consider moving goal position
        # desired_state = desired_state + Ts*(-1)*np.ones(len(robot_state))

    # End of iterations
    # ---------------------------
    # return the stored value for additional plotting or comparison of parameters
    return state_history, goal_history, input_history, w_left_history, w_right_history, polar_state_history


if __name__ == '__main__':
    # Call main computation for robot simulation
    state_history, goal_history, input_history, w_left_history, w_right_history, polar_state_history = simulate_control()

    # ADDITIONAL PLOTTING
    # ----------------------------------------------
    t = [i * Ts for i in range(round(t_max / Ts))]

    # Plot historical data of control input
    fig2 = plt.figure(2)
    ax = plt.gca()
    ax.plot(t, input_history[:, 0], label='v [m/s]')
    ax.plot(t, input_history[:, 1], label='w [rad/s]')
    ax.set(xlabel="t [s]", ylabel="control input", title='Control input')
    plt.legend()
    plt.grid()

    # Plot historical data of state
    fig3 = plt.figure(3)
    ax = plt.gca()
    ax.plot(t, state_history[:, 0], label='px [m]')
    ax.plot(t, state_history[:, 1], label='py [m]')
    ax.plot(t, state_history[:, 2], label='theta [rad]')
    ax.plot(t, goal_history[:, 0], ':', label='goal px [m]')
    ax.plot(t, goal_history[:, 1], ':', label='goal py [m]')
    ax.plot(t, goal_history[:, 2], ':', label='goal theta [rad]')
    ax.set(xlabel="t [s]", ylabel="state", title='State plot')
    plt.legend()
    plt.grid()

    # Plot wheel speed
    fig4 = plt.figure(4)
    ax = plt.gca()
    ax.plot(t, w_left_history, label='w_left')
    ax.plot(t, w_right_history, label='w_right')
    ax.set(xlabel="t [s]", ylabel="rotational speed [rad/s]", title='Wheel speed plot')
    plt.legend()
    plt.grid()

    fig5 = plt.figure(5)
    ax = plt.gca()
    ax.plot(t, polar_state_history[:, 0], label='rho [m]')
    ax.plot(t, polar_state_history[:, 1], label='alpha [rad]')
    ax.plot(t, polar_state_history[:, 2], label='beta [rad]')
    ax.set(xlabel="t [s]", ylabel="state", title='State in polar coordinate')
    plt.legend()
    plt.grid()

    plt.show()
