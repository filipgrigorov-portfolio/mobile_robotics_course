#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

# icp_known_corresp: performs icp given that the input datasets
def icp_known_corresp(q, p, q_indices, p_indices):
    q = q[:, q_indices]
    p = p[:, p_indices]

    #print(f'q: {q.shape}')
    #print(f'p: {p.shape}')

    # Isolate the rotation
    mu_q = compute_mean(q)
    mu_p = compute_mean(p)
    
    cross_covariance = compute_cross_covariance(q, p, mu_q, mu_p)

    rotation, translation = compute_R_t(cross_covariance, mu_q, mu_p)

    #print('R: ', rotation.shape)
    #print('t: ', translation.shape)

    # Compute the new positions of the points after
    # applying found rotation and translation to them
    p_decentered = p - mu_p
    p_rotated = rotation.dot(p_decentered)
    p_transformed = p_rotated + mu_q + translation

    #print(f'q: {q.shape}')
    #print(f'p_transformed: {p_transformed.shape}')

    error = compute_error(q, p_transformed)

    return p_transformed, error

# compute_cross_covariance: compute matrix W to use in SVD
def compute_cross_covariance(q, p, mu_q, mu_p):
    q_shifted = q - mu_q
    p_shifted = p - mu_p
    return q_shifted.dot(p_shifted.T)

    
# compute_R_t: compute rotation matrix and translation vector
# based on the SVD as presented in the lecture
def compute_R_t(cross_covariance, mu_q, mu_p):
    u, d, v = np.linalg.svd(cross_covariance)
    rotation = u.dot(v.T)
    # [2 x 100] - [2 x 2].[2 x 100] = [2 x 100]
    return rotation, mu_q - rotation.dot(mu_p)


# compute_mean: compute mean value for a [M x N] matrix
def compute_mean(mat):
    return np.expand_dims(np.ceil(np.mean(mat, axis=1)), axis=1)


# compute_error: compute the icp error
def compute_error(q, p_transformed):
    return np.linalg.norm(q - p_transformed)


# simply show the two lines
def show_figure(line1, line2):
    plt.figure()
    plt.scatter(line1[0], line1[1], marker='o', s=2, label='Q')
    plt.scatter(line2[0], line2[1], s=1, label='P')
    
    plt.xlim([-8, 8])
    plt.ylim([-8, 8])
    plt.legend()  
    
    plt.show()
    

# initialize figure
def init_figure():
    fig = plt.gcf()
    fig.show()
    fig.canvas.draw()
    
    line1_fig = plt.scatter([], [], marker='o', s=2, label='Line 1')
    line2_fig = plt.scatter([], [], marker='o', s=1, label='Line 2')
    # plt.title(title)
    plt.xlim([-8, 8])
    plt.ylim([-8, 8])
    plt.legend()
    
    return fig, line1_fig, line2_fig


# update_figure: show the current state of the lines
def update_figure(fig, line1_fig, line2_fig, Line1, Line2, hold=False):
    line1_fig.set_offsets(Line1.T)
    line2_fig.set_offsets(Line2.T)
    if hold:
        plt.show()
    else:
        fig.canvas.flush_events()
        fig.canvas.draw()
        plt.pause(0.5)

'''
Use the result of your code from the first question, to implement the full ICP algorithm.

When the point correspondences are not available. You will need to iteratively find the point correspondences and using these perform the ICP updates.

A starting point for this exercise is given as follows.

Make you algorithm stop after convergence.

Hint: The NearestNeighbors functions of sklearn library can be useful in this task.
'''
def run_unknown_correspondances():
    Data = np.load('icp_data.npz')
    Line1 = Data['LineGroundTruth']
    Line2 = Data['LineMovedNoCorresp']

    MaxIter = 100
    Epsilon = 0.001
    E = np.inf

    # show figure
    ex.show_figure(Line1, Line2)
        
    for i in range(MaxIter):

        # TODO: find correspondences of points
        # point with index QInd(1, k) from Line1 corresponds to
        # point with index PInd(1, k) from Line2
        QInd = ...
        PInd = ...

        # update Line2 and error
        # Now that you know the correspondences, use your implementation
        # of icp with known correspondences and perform an update
        EOld = E
        [Line2, E] = ex.icp_known_corresp(Line1, Line2, QInd, PInd)
            
        print('Error value on ' + str(i) + ' iteration is: ', E)

        # TODO: perform the check if we need to stop iterating
        ...

def run_known_correspondances():
    import ex1 as ex
    import numpy as np
    import matplotlib.pyplot as plt

    data = np.load('icp_data.npz')
    Q = data['LineGroundTruth']
    P = data['LineMovedCorresp']
        
    # Show the initial positions of the lines
    ex.show_figure(Q, P)

    # We assume that the there are 1 to 1 correspondences for this data
    # Assign labels
    Q_indices = np.arange(len(Q[0]))
    P_indices = np.arange(len(P[0]))

    # Perform icp given the correspondences
    [P_transformed, error] = ex.icp_known_corresp(Q, P, Q_indices, P_indices)

    # Show the adjusted positions of the lines
    ex.show_figure(Q, P_transformed)

    # print the error
    print('Error value is: ', error)

if __name__ == '__main__':
    run_known_correspondances()
