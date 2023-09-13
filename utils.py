import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

import pdb



def get_transform_matrix(V, T, normal = np.array([0, 0, 1])):
    # 计算旋转轴
    rotation_axis = np.cross(normal, V)

    # 计算旋转角度
    cos_theta = np.dot(normal, V)
    sin_theta = np.linalg.norm(rotation_axis)
    theta = np.arctan2(sin_theta, cos_theta)

    # 使用旋转轴和旋转角度构建旋转矩阵
    rotation_matrix = Rotation.from_rotvec(theta * rotation_axis).as_matrix()


    transform_matrix = np.eye(4)  # 创建一个4x4的单位矩阵
    transform_matrix[:3, :3] = rotation_matrix  # 将旋转矩阵放入变换矩阵的左上角
    transform_matrix[:3, 3] = T  # 将平移向量放入变换矩阵的右上角
    return transform_matrix

def get_rotation_matrix(V, normal = np.array([0, 0, 1])):
    # 计算旋转轴
    rotation_axis = np.cross(normal, V)

    # 计算旋转角度
    cos_theta = np.dot(normal, V)
    sin_theta = np.linalg.norm(rotation_axis)
    theta = np.arctan2(sin_theta, cos_theta)

    # 使用旋转轴和旋转角度构建旋转矩阵
    rotation_matrix = Rotation.from_rotvec(theta * rotation_axis).as_matrix()
    
    return rotation_matrix

def _get_circle_points(r, theta_start = 0, theta_end = 2*np.pi, sample_num = 100):
    # Generate angles
    theta = np.linspace(theta_start, theta_end, sample_num)

    # Generate points on the circle
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = np.zeros_like(x)

    points = np.column_stack([x, y, z])

    return points

def get_transformed_circle(r, V, T, theta_start = 0, theta_end = 2*np.pi, sample_num = 100):

    _circle_points = _get_circle_points(r, theta_start, theta_end, sample_num)
    # 将坐标点扩展为齐次坐标，添加额外的维度为1
    homogeneous_points = np.column_stack((_circle_points, np.ones((sample_num, 1))))

    transforme_matrix = get_transform_matrix(V, T)

    transformed_circle = np.dot(transforme_matrix, homogeneous_points.T).T
    return transformed_circle[:, :3]

def get_transformed_circle_1(r, V, T, theta_start = 0, theta_end = 2*np.pi, sample_num = 100):

    circle_points = _get_circle_points(r, theta_start, theta_end, sample_num)
    # 将坐标点扩展为齐次坐标，添加额外的维度为1
    # homogeneous_points = np.column_stack((_circle_points, np.ones((sample_num, 1))))

    rotation_matrix = get_rotation_matrix(V)

    rotation_circle = np.dot(rotation_matrix, circle_points.T).T
    # pdb.set_trace()
    # a = rotation_circle[0, :]
    # b = rotation_circle[49, :]
    # c = rotation_circle[99, :]
    # d = np.cross(b - a, c - b)
    # e = d / np.linalg.norm(d)
    # # print(e)
    # pdb.set_trace()
    transformed_circle = rotation_circle + T

    # a = transformed_circle[0, :]
    # b = transformed_circle[49, :]
    # c = transformed_circle[99, :]

    # d = np.cross(b - a, c - b)
    # e = d / np.linalg.norm(d)
    # # print(e)

    return transformed_circle

def scatter_points(ax, points, color='r'):
    # ax.set_box_aspect([1, 1, 1])  # 设置三个坐标轴尺度一致
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color=color)


def plot_points(ax, points, color='r'):
    # ax.set_box_aspect([1, 1, 1])  # 设置三个坐标轴尺度一致
    ax.plot(points[:, 0], points[:, 1], points[:, 2], color=color)


def get_guided_L_cord(r=1):
    # 引导灯坐标
    p1 = np.array([       0,  np.sqrt(3) * r / 3, 0])
    p2 = np.array([-r / 2.0, -np.sqrt(3) * r / 6, 0])
    p3 = np.array([ r / 2.0, -np.sqrt(3) * r / 6, 0])

    return p1, p2, p3

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


def T_points2circle_r(TA, TB, circle_angle):

    l = np.linalg.norm(TB-TA)
    cc_r = l / 2 / np.tan(circle_angle)
    r = l / 2 / np.sin(circle_angle)

    return cc_r, r


if __name__ == '__main__':
    r = 1
    V = np.array([1, 0, 0])
    T = np.array([2, 0, 0])
    # T = np.array([ 0.,-0.28867513,  2.        ])
    circle_points = get_transformed_circle_1(r, V, T)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter_points(ax, circle_points, 'g')
    plt.show()


    




