import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import utils as u

import pdb

def get_circle_set(TA, TB):
    circle_points_list = []
    circle_points_num = circle_points.shape[0]
    for idx in range(circle_points_num):
        T_cc_ = circle_points[idx, :]
        AB = (TB - TA) / np.linalg.norm(TB - TA)
        # Tcc_TA = TA - T_cc_
        # Tcc_TB = TB - T_cc_
        V_ = np.cross(TA - T_cc_, TB - T_cc_) 
        V_ = V_ / np.linalg.norm(V_)

        # theta_start = np.arctan2((TA - T_cc_)[1], (TA - T_cc_)[0])
        # theta_end = np.arctan2((TB - T_cc_)[1], (TB - T_cc_)[0])
        # print(np.degrees(theta_start), np.degrees(theta_end))

        # arc_points = []
        # arc_num = 0
        circle_points_ = u.get_transformed_circle(r, V_, T_cc_)
        # points_num = circle_points_.shape[0]
        # for idx in range(points_num):
        #     circle_point = circle_points_[idx, :]
        #     dot_product = np.dot(circle_point - TA, AB)

        #     if dot_product < 0:
        #         arc_points.append(circle_point)
        #         arc_num += 1

        # circle_points_list.append(np.array(arc_points).reshape((arc_num, 3)))
        circle_points_list.append(circle_points_)

    return circle_points_list 




if __name__ == '__main__':
    '''
    计算圆心坐标
    '''
    # init data
    T = np.array([0, 0, 2])  
    A, B, C = u.get_guided_L_cord(r=1) 
    TA, TB, TC, alpha, beta, gamma = u.cords2insc(A, B, C, T)
    cc_r, r = u.T_points2circle_r(TB, TC, beta)

    # calculate V and T
    V = (TB-TC)/np.linalg.norm(TB-TC)
    T_cc = (TB + TC)/2 

    circle_points = u.get_transformed_circle(cc_r, V, T_cc)


    BC_circle_points_list = get_circle_set(TB, TC) 
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1, 1, 1])  # 设置三个坐标轴尺度一致

    u.scatter_points(ax, np.stack([T, TA, TB, TC], axis=0), 'g')

    u.scatter_points(ax, circle_points, 'g')

    for circle_points in BC_circle_points_list:
        u.plot_points(ax, circle_points, 'r')
    # u.plot_points(ax, BC_circle_points_list[0], 'r')

    plt.show()

    