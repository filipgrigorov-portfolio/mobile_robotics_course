import numpy as np

def normal(x, var):
    norm = 1.0 / np.sqrt(2.0 * np.pi * var * var)
    return norm * np.exp(-0.5 * (x * x / var * var))

def sample(var):
    return 0.5 * np.random.uniform(-var, var)

def inverse_motion_model(u):
    '''
        Computes the motion embedding off the odometry.

        input1: x_t-1 = [x, y, theta]
        input2: x_t = [x', y', theta']
        output: delta_rot1, delta_trans, delta_rot2

        [rot_1, trans, rot_2]:
        theta_start + rot_1 = theta_next
        theta_start + rot_1 + rot_2 = theta_end

        rot_1 = theta_next - theta_start
        rot_2 = thetha_end - rot_1 - theta_start
    '''

    y_diff = u[1][1] - u[0][1]
    x_diff = u[1][0] - u[0][0]
    theta_next = np.arctan2(y_diff, x_diff)
    rot1 = theta_next - u[0][2]
    trans = np.sqrt(x_diff * x_diff + y_diff * y_diff)
    rot2 = u[1][2] - u[0][2] - rot1
    return np.around(rot1, 4), np.around(trans, 4), np.around(rot2, 4)

def compute_motion_probs(pose_prev, pose_next, u, alphas):
    # The actual poses
    rot1, trans, rot2 = inverse_motion_model(u)

    rot1_squared = rot1 * rot1
    trans_squared = trans * trans
    rot2_squared = rot2 * rot2

    x_prev = pose_prev - pose_prev
    x_next = pose_next - pose_next
    # The mean expected poses
    rot1_exp, trans_exp, rot2_exp = inverse_motion_model(np.array([ x_prev, x_next ]))

    p1 = normal(rot1 - rot1_exp, alphas[0] * rot1_squared + alphas[1] * trans_squared)
    p2 = normal(trans - trans_exp, 
        alphas[2] * trans_squared + alphas[3] * rot1_squared + alphas[3] * rot2_squared)
    p3 = normal(rot2 - rot2_exp, alphas[0] * rot2_squared + alphas[1] * trans_squared)

    return p1, p2, p3

# Hyptohessis (pose or mean expected pose) and odometry (actual value)
def motion_model_odometry(pose, u, alphas):
    '''
        Computes the posterior p(x_t | u_t, x_t-1).

        input1: x_t-1 = ( x, y, theta ).T
        input2: x_t = ( x', y', theta' ).T
        input3: u_t=( x_bar_t-1, x_bar_t ).T
        input4: noise alpha = ( alpha1, alpha2, alpha3, alpha4 ); These are robot-specific error parameters.

        output: prob = p(x_t | u_t, x_t-1)
    '''
    p1, p2, p3 = compute_motion_probs(pose[0], pose[1], u, alphas)

    return p1 * p2 * p3

# Do I need to transform from map coordinates to robot coordinates?????

def motion_model_odometry_with_map(pose_prev, u, gridmap, alphas):
    # Compute probability of going there (x_t), for each position (x, y) within grid
    rows, cols = gridmap.shape[0], gridmap.shape[1]
    angle_step = np.pi * 0.5
    nangles = 4
    for y in range(0, rows):
        for x in range(0, cols):
            thetas = np.linspace(0, nangles * angle_step, nangles)
            for theta in thetas:
                # Integrate/sum over each final theta (orientation)
                pose_next = np.array([ x, y, theta ])
                p1, p2, p3 = compute_motion_probs(pose_prev, pose_next, u, alphas)
                gridmap[y][x] += p1 * p2
            gridmap[y][x] /= nangles
    return gridmap

def sample_motion_model_odometry(pose_prev, u, alphas):
    rot1, trans, rot2 = inverse_motion_model(u)

    rot1_squared = rot1 * rot1
    trans_squared = trans * trans
    rot2_squared = rot2 * rot2

    rot1 -= sample(alphas[0] * rot1_squared + alphas[1] * trans_squared)
    trans -= sample(
        alphas[2] * trans_squared + alphas[3] * rot1_squared + alphas[3] * rot2_squared)
    rot2 -= sample(alphas[0] * rot2_squared + alphas[1] * trans_squared)

    x_t = pose_prev[0] + trans * np.cos(pose_prev[2] + rot1)
    y_t = pose_prev[1] + trans * np.sin(pose_prev[2] + rot1)
    theta_t = pose_prev[2] + rot1 + rot2

    return np.array([ x_t, y_t, theta_t ])
