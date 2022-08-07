#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import bresenham as bh

# laser properties
start_angle = -1.5708
angular_res = 0.0087270
max_range = 30

def plot_gridmap(gridmap):
    plt.figure(figsize=(10, 10))
    plt.imshow(gridmap, cmap='Greys',vmin=0, vmax=1)
    
def init_gridmap(size, res):
    gridmap = np.zeros([int(np.ceil(size / res)), int(np.ceil(size / res))])
    return gridmap

def world2map(pose, gridmap, map_res):
    origin = np.array(gridmap.shape) / 2
    new_pose = np.zeros((pose.shape[0], pose.shape[1]))
    new_pose[:, 0] = np.round(pose[:, 0] / map_res) + origin[0]
    new_pose[:, 1] = np.round(pose[:, 1] / map_res) + origin[1]
    return new_pose.astype(np.int)

def compute_transform_by(pose):
    c = np.cos(pose[2])
    s = np.sin(pose[2])
    tr = np.array([
        [c, -s, pose[0]], 
        [s, c, pose[1]], 
        [0, 0, 1]
    ])
    return tr    

def ranges2points(ranges):
    # rays within range
    num_beams = ranges.shape[0]
    idx = (ranges < max_range) & (ranges > 0)

    # 2D points: num_beams of points per pose (with theta increment)
    angles = np.linspace(start_angle, start_angle + (num_beams * angular_res), num_beams)[idx]
    points = np.array([np.multiply(ranges[idx], np.cos(angles)), np.multiply(ranges[idx], np.sin(angles))])
    
    # homogeneous points (easier application of transform; could represent inf points eg. [1, 1, 0])
    points_hom = np.append(points, np.ones((1, points.shape[1])), axis=0)
    return points_hom

def ranges2cells(r_ranges, w_pose, gridmap, map_res):
    # ranges to points
    r_points = ranges2points(r_ranges)
    #print(f'r_points (3 x num_beans): {r_points}\n')
    w_P = compute_transform_by(w_pose)
    #print(f'w_P: {w_P}\n')
    # 3 x 322 : x, y, z
    w_points = np.matmul(w_P, r_points)
    #print(f'w_points (3 x num_beans): {w_points}\t{w_points.shape}\n')

    # covert to map frame
    #m_points = world2map(w_points, gridmap, map_res)

    origin = np.array(gridmap.shape) / 2

    m_points = np.zeros((w_points.shape[0], w_points.shape[1]))
    m_points[0, :] = np.round(w_points[0, :] / map_res) + origin[0]
    m_points[1, :] = np.round(w_points[1, :] / map_res) + origin[1]
    m_points = m_points.astype(np.int)

    #print(f'm_points (3 x num_beans): {m_points}\n\n')
    m_points = m_points[0 : 2, :]
    #print(f'm_points (3 x num_beans): {m_points}\n')

    return m_points

def poses2cells(w_pose, gridmap, map_res):
    # covert to map frame
    m_pose = world2map(w_pose, gridmap, map_res)
    return m_pose  

def bresenham(x0, y0, x1, y1):
    l = np.array(list(bh.bresenham(x0, y0, x1, y1)))
    return l
    
def prob2logodds(p):
    inv_p = 1.0 - p
    return round(np.log(p / inv_p), 2) if inv_p != 0.0 else 0.5
    
def logodds2prob(l):
    return round(1.0 - (1.0 / (1 + np.exp(l))), 2)
    
def inv_sensor_model(cell, endpoint, prob_occ, prob_free):
    line = bresenham(cell[0], cell[1], endpoint[0], endpoint[1])

    probs = []
    for idx in range(0, len(line) - 1):
        given_cell = line[idx]
        probs.append([given_cell[0], given_cell[1], prob_free])
    probs.append([endpoint[0], endpoint[1], prob_occ])

    return np.array(probs)

# Each z_t represents a full circle of the lidar with an angle attached to it
def grid_mapping_with_known_poses(ranges_raw, poses_raw, occ_gridmap, map_res, prob_occ, prob_free, prior):
    # log odds of prior (const)
    odds_prior = prob2logodds(prior)

    T = ranges_raw.shape[0]

    assert(T == poses_raw.shape[0])

    poses_cells = poses2cells(poses_raw, occ_gridmap, map_res)

    # for each explored cell (pose)
    for t in range(0, T):
        x_t = poses_raw[t]
        x_cell = poses_cells[t]
        z_t = ranges_raw[t]

        #print(f'{occ_gridmap.shape}\n')
        #print(f'pose: {x_t}\n\n{x_cell}\n\n{z_t}')

        # find cells to pose and range: [2 X 322]
        z_cells = ranges2cells(z_t, x_t, occ_gridmap, map_res).T

        # for each measurement cell: 322 x 2
        for z_cell in z_cells:
            endpoint = np.array([z_cell[0], z_cell[1]])

            # find measurement map cells along the ray trace, along with occupied probability
            sensor_model = inv_sensor_model(x_cell, endpoint, prob_occ, prob_free)

            # update map cells along ray trace with occupied probability
            for i in range(len(sensor_model)):
                row = int(sensor_model[i][0])
                col = int(sensor_model[i][1])

                odds_old = prob2logodds(occ_gridmap[row][col])
                odds_current = prob2logodds(sensor_model[i][-1])

                odds = odds_old + odds_current + odds_prior
                prob = logodds2prob(odds)
                occ_gridmap[row][col] = prob

    return occ_gridmap
