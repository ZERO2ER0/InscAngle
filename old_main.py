import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# from data.gen_data import cords2inc
from utils import get_transform_matrix

import pdb

# def cal_arc_points(CC, r, A, B):
#     # Calculate vectors CA and CB
#     CA = A - CC
#     CB = B - CC

#     # Calculate cross product of CA and CB
#     cross = np.cross(CA, CB)

#     # Normalize cross product
#     cross_norm = cross / np.linalg.norm(cross)

#     # Calculate angles of CA and CB
#     theta_CA = np.arctan2(CA[1], CA[0])
#     theta_CB = np.arctan2(CB[1], CB[0])

#     # Generate angles for the arc
#     theta = np.linspace(theta_CA, theta_CB, 100)

#     # Calculate points on the arc
#     # X = CC[0] + r * np.cos(theta) * np.cos(theta)
#     # Y = CC[1] + r * np.cos(theta) * np.sin(theta)
#     # Z = CC[2] + r * np.sin(theta) * cross_norm[2]
#     # arc_points = np.stack([X, Y, Z], axis = 1)
#     # arc_points = CC[:, np.newaxis] + r * (np.cos(theta)[:, np.newaxis] * np.vstack((np.cos(theta), np.sin(theta))).T) + r * (np.sin(theta)[:, np.newaxis] * cross_norm[np.newaxis, :2])
#     # arc_points = CC[:, np.newaxis] + r * (np.cos(theta)[:, np.newaxis] * np.vstack((np.cos(theta), np.sin(theta))).T) + r * (np.sin(theta)[:, np.newaxis] * cross_norm)
#     # arc_points = CC[:, np.newaxis] + r * (np.cos(theta)[:, np.newaxis] * np.vstack((np.cos(theta), np.sin(theta))).T) + r * (np.sin(theta)[:, np.newaxis] * cross_norm[np.newaxis, :])
#     # arc_points = CC[:, np.newaxis] + r * (np.cos(theta)[:, np.newaxis] * np.vstack((np.cos(theta), np.sin(theta))).T) + r * (np.sin(theta)[:, np.newaxis] * cross_norm.T)
#     arc_points = CC[:, np.newaxis] + r * np.cos(theta)[:, np.newaxis] * np.vstack((np.cos(theta), np.sin(theta))).T + r * np.sin(theta)[:, np.newaxis] * cross_norm

#     return arc_points
def cal_arc_points(Cc, r, A, B, rad):
    # pdb.set_trace()
    # 计算圆面法线
    v1 = A - Cc
    v2 = B - Cc
    radius = np.linalg.norm(v1)
    # print(v1, v2)
    normal = np.cross(v1, v2)
    normal /= np.linalg.norm(normal)
    
    # print(r)
    theta = np.linspace(0, 2 * np.pi, 100)
    # x = Cc[0] + r * np.cos(theta)
    # y = Cc[1] + r * np.sin(theta)
    # z = Cc[2] + np.zeros_like(x)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    z = np.zeros_like(x)

    # points = np.vstack([x, y, z])
    # return points
    # pdb.set_trace()
    # # 将圆面法线应用到所有点
    rotation_matrix = np.eye(3) - normal[:, np.newaxis] * normal
    rotated_points = np.dot(rotation_matrix, np.vstack([x, y, z]))

    points = rotated_points + Cc.reshape((3,1))

    return np.transpose(points, (1,0))


    # # 计算圆的半径
    # # radius = np.sqrt(np.sum((point1 - center) ** 2))
    # CA = A - Cc
    # CB = B - Cc
    # # pdb.set_trace()
    # # V = CA/np.linalg.norm(CA)
    # # 圆面的垂直向量
    # Vc = np.cross(CA, CB) / np.linalg.norm(np.cross(CA, CB))
    # # 随机单位向量
    # r_vec = np.random.randn(3)
    # # r_vec = r_vec / np.linalg.norm(r_vec)
    # # 随机向量到垂直向量的投影，随机向量-垂直向量 = 随机向量到圆面的投影
    # r_vec_ = np.dot(r_vec, Vc) / np.linalg.norm(Vc)
    # r_vec__ = r_vec - r_vec_
    # # r_vec -= np.dot(r_vec, Vc) * Vc / np.linalg.norm(Vc)
    # # 随机向量-垂直向量 = 随机向量到圆面的投影
    # r_vec__ /= np.linalg.norm(r_vec__)

    # theta_CA = np.arctan2(CA[1], CA[0])
    # theta_CB = np.arctan2(CB[1], CB[0])
    # # print(np.linalg.norm(CA), r)
    # # pdb.set_trace()

    # # theta = np.linspace(theta_CA, theta_CA + 2*rad, 100)
    # theta = np.linspace(0, 2 * np.pi, 100)
    # # pdb.set_trace()
    # # arc_points = []
    # # arc_points = np.array([Cc[0] + r * np.cos(theta), Cc[1] + r * np.sin(theta), np.full_like(theta, Cc[2])]).T
    # points = []
    # for angle in theta:
    #     cos_theta = np.cos(angle)
    #     sin_theta = np.sin(angle)
    #     P = Cc + r * cos_theta * r_vec__ + r * sin_theta * np.cross(np.cross(CA, CB), r_vec__)
    #     points.append(P)

    # # Convert points to NumPy array
    # points = np.array(points)

    return points

def plot_points(ax, points, color='r'):
    ax.plot(points[:, 0], points[:, 1], points[:, 2], color=color)
    ax.set_box_aspect([1, 1, 1])  # 设置三个坐标轴尺度一致


def scatter_points(ax, points, color='r'):
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color=color)
    ax.set_box_aspect([1, 1, 1])  # 设置三个坐标轴尺度一致
    

def cal_circle_points(A, B, r, sample_num=100):
    # Calculate midpoint
    M = (A + B) / 2

    # Calculate vector AB
    AB = B - A
    normal = AB / np.linalg.norm(AB)

    # Generate random vector perpendicular to AB
    # V = np.random.randn(3)
    # # print(V, np.linalg.norm(V))
    # # V = np.array([0.5, 1., 1.])
    # V -= np.dot(V, AB) / np.linalg.norm(AB)
    # V /= np.linalg.norm(V)

    # Generate angles
    theta = np.linspace(0, 2*np.pi, sample_num)

    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = np.zeros_like(x)

    rotation_matrix = np.eye(3) - normal[:, np.newaxis] * normal
    pdb.set_trace()
    rotated_points = np.dot(rotation_matrix, np.vstack([x, y, z]))
    points = rotated_points + M.reshape((3,1))

    return np.transpose(np.vstack([x, y, z]), (1,0))


    # Generate points on the circle
    # points = []
    # for angle in theta:
    #     cos_theta = np.cos(angle)
    #     sin_theta = np.sin(angle)
    #     P = M + r * cos_theta * V + r * sin_theta * np.cross(AB, V)
    #     points.append(P)

    # # Convert points to NumPy array
    # points = np.array(points)
    return points

def cal_circle_points_(A, B, r, sample_num=100):
    # Calculate midpoint
    M = (A + B) / 2

    # Calculate vector AB
    AB = B - A

    # Generate random vector perpendicular to AB
    # V = np.random.randn(3)
    # np.array([1,1,1])
    V -= np.dot(V, AB) * AB / np.linalg.norm(AB)
    V /= np.linalg.norm(V)

    # Generate angles
    theta = np.linspace(0, 2*np.pi, sample_num)

    # Generate points on the circle
    points = []
    for angle in theta:
        cos_theta = np.cos(angle)
        sin_theta = np.sin(angle)
        P = M + r * cos_theta * V + r * sin_theta * np.cross(AB, V)
        points.append(P)

    # Convert points to NumPy array
    points = np.array(points)
    return points

def get_guided_L_cord(r=1):
    # 引导灯坐标
    p1 = np.array([       0,  np.sqrt(3) * r / 3, 0])
    p2 = np.array([-r / 2.0, -np.sqrt(3) * r / 6, 0])
    p3 = np.array([ r / 2.0, -np.sqrt(3) * r / 6, 0])

    return p1, p2, p3

def insc2d_r(A, B, insc):
    l = np.linalg.norm(B-A)
    d = l / 2 / np.tan(insc)
    r = l / 2 / np.sin(insc)

    return d, r

def cords2insc(A, B, C, T):

    TA = T + A
    TB = T + B
    TC = T + C

    alpha = cords2inc(TA, TB)
    beta  = cords2inc(TB, TC)
    gamma = cords2inc(TC, TA)

    return TA, TB, TC, alpha, beta, gamma

def cords2inc(vec0, vec1):
    # vec0 = vec0 * vec_expansion
    # vec1 = vec1 * vec_expansion
    dot_product = np.dot(vec0, vec1)
    mag0 = np.linalg.norm(vec0)
    mag1 = np.linalg.norm(vec1)
    inc_rad = np.arccos(dot_product / (mag0 * mag1))
    return inc_rad


if __name__ == '__main__':
    # init data
    T = np.array([0, 0, 1])  
    A, B, C = get_guided_L_cord(r=1) 
    TA, TB, TC, alpha, beta, gamma = cords2insc(A, B, C, T)

    # 
    d_b, r_b = insc2d_r(TB, TC, beta)

    # # fig = plt.figure()
    circle_points = cal_circle_points(TB, TC, d_b)
    print(circle_points[0])
    print(TC, TB)
    # arc_points = cal_arc_points(circle_points[2], r_b, TB, TC, beta)
    # print(circle_points[0])

    # pdb.set_trace()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter_points(ax, np.stack([TB, TC], axis=0), 'g')

    scatter_points(ax, circle_points, 'b')
    # plot_points(ax, arc_points, 'r')
    plt.show()
    
    # pdb.set_trace()
    