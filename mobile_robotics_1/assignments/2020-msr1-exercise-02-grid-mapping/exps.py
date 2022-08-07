#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import bresenham as bh

# Laser properties:
class laser:
    start_angle = -1.5708
    angular_res = 0.0087270 # angle increments while lidar rotates
    max_range = 30

def rad2deg(angle_rad):
    return 180.0 * angle_rad / np.pi

def deg2rad(angle_degree):
    return np.pi * angle_degree / 180.0

def plot_gridmap(gridmap):
    plt.figure()
    plt.imshow(gridmap, cmap='Greys', vmin=0, vmax=1)
    
def init_gridmap(size, res):
    # M -> size x size with (size / res) cells
    gridmap = np.zeros([int(np.ceil(size / res)), int(np.ceil(size / res))])
    return gridmap

def world2map(pose, gridmap, map_res):
    '''
        Projects world pose onto 2D grid.
    '''
    origin = np.array(gridmap.shape) / 2
    new_pose = np.zeros((2), dtype=np.int) # [x', y']
    new_pose[0] = np.round(pose[0] / map_res) + origin[0]
    new_pose[1] = np.round(pose[1] / map_res) + origin[1]
    return new_pose

def vec2transform(pose):
    cosine = np.cos(pose[2])
    sine = np.sin(pose[2])
    # rotation (theta degrees ccw) + translation
    transform = np.array(
        [
            [cosine, -sine, pose[0]], 
            [sine, cosine, pose[1]], 
            [0, 0, 1]
        ])
    return transform  

def ranges2points(ranges):
    # rays within range (If beam of ranges is within the interval)
    num_beams = ranges.shape[0]
    in_range = (ranges < laser.max_range) & (ranges > 0)
    
    # 2D points
    # angle_i = angle_i-1 + num_beans * angle_res
    #angles = np.linspace(laser.start_angle, laser.start_angle + (num_beams * laser.angular_res), num_beams)
    angles = np.zeros_like(ranges)
    for i in range(0, len(angles)):
        angles[i].fill(laser.start_angle + i * laser.angular_res)
    angles = angles[in_range]

    # point_i = [ range_val * cos(angle), range_val * sin(angle) ] = [ x, y ]
    points = np.array([ np.multiply(ranges[in_range], np.cos(angles)), np.multiply(ranges[in_range], np.sin(angles)) ])
    
    # homogeneous points
    # p_i goes from [ x, y ] to [ x, y, 1 ]
    points_hom = np.append(points, np.ones((1, points.shape[1])), axis=0)

    return points_hom


def ranges2cells(r_ranges, world_pose, gridmap, map_res):
    '''
        Converts the raw measurements acquired by the robot (ranges_raw) into the correspoding cells of the gridmap.
    '''
    # ranges to points
    r_points = ranges2points(r_ranges)
    
    # transform the beam according to the pose orientation
    w_P = vec2transform(world_pose)
    # [3 x 241489]
    w_points = np.matmul(w_P, r_points)

    # [686 x n x 3]
    

    #raise

    # covert to map frame
    m_points = world2map(w_points, gridmap, map_res)
    m_points = m_points[0 : 2, :]

    print(m_points.shape)
    raise('debug')
    return m_points

def pose2cell(world_pose, gridmap, map_res):
    '''
        Converts the raw poses of the robot (poses_raw) into the correspoding cells of the gridmap (to map frame).
    '''
    map_pose = world2map(world_pose, gridmap, map_res)
    return map_pose  

def bresenham(x0, y0, x1, y1):
    '''
        Returns all the cells along a straigh line between two points in the gridmap.
    '''
    line_coords = np.array( list( bh.bresenham(x0, y0, x1, y1) ) )
    return line_coords
    
def prob2logodds(p):
    return round(np.log(p / (1.0 - p)), 2)
    
def logodds2prob(l):
    return round(1.0 - (1.0 / (1 + np.exp(l))), 2)
    
def inv_sensor_model(cell, measurement, prob_occ, prob_free):
    '''
        @ arg: cell: pose measured in cell in world coordinates (m_i)
        @ arg: measurement: range value of 2d laser scan in world coordinates (z_t_i)
    '''
    # compute logits
    logit_occ = prob2logodds(prob_occ)
    logit_free = prob2logodds(prob_free)

    # compute inverse logit prob
    logit_prob = logit_free if cell in measurement else logit_occ
    return logodds2prob(logit_prob)

# poses of (686, 3) and ranges of (686, 361)
def grid_mapping_with_known_poses(ranges_raw, poses_raw, occ_gridmap, map_res, prob_occ, prob_free, prior):
    assert(ranges_raw.shape[0] == poses_raw.shape[0])

    # Compute cells for poses and ranges
    ranges_cells = ranges2cells(ranges_raw, poses_raw, occ_gridmap, map_res)
    print(ranges_cells.shape)

    poses_cells = np.array([ pose2cell(world_pose, occ_gridmap, map_res) for world_pose in poses_raw ])
    print(poses_cells.shape)

    raise('debug')

    #print(f'Sample pose: { poses_cells[0] }\n')
    #print(f'Sample range beam: {ranges_cells[0]}\n')

    n = poses_cells.shape[0]
    assert(n == ranges_cells.shape[0])

    rows = occ_gridmap.shape[0]
    cols = occ_gridmap.shape[1]

    for t in range(n):
        x_t = poses_cells[t]
        z_t = ranges_cells[t]

        for row in range(rows):
            for col in range(cols):
                cell_i = (row, col)
                prob_prev = occ_gridmap[row][col]
                l_prev = prob2logodds(prob_prev)

                cells_along_line = bresenham(x_t[0], x_t[1], z_t[0], z_t[1])

                if cell_i in cells_along_line:
                    l_next = l_prev + inv_sensor_model(cell_i, z_t, prob_occ, prob_free) + prior
                else:
                    l_next = l_prev

                occ_gridmap[row][col] = logodds2prob(l_next)

    return occ_gridmap
