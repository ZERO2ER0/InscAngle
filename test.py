# import numpy as np
# import matplotlib.pyplot as plt
# import pdb

# from old_main import get_guided_L_cord, cords2insc, insc2d_r, cal_circle_points

# T = np.array([0, 0, 1])
# A, B, C = get_guided_L_cord(r=1)
# TA, TB, TC, alpha, beta, gamma = cords2insc(A, B, C, T)

# d_b, r_b = insc2d_r(TB, TC, beta)

# # print(A, B, C, TA, TB, TC, alpha, beta, gamma, d_b, r_b)ß

# circle_points = cal_circle_points(TB, TC, d_b)
# print(circle_points[0])
# print(TC, TB)
# # 已知数据
# Cx, Cy, Cz = circle_points[0][0], circle_points[0][1], circle_points[0][2]  # 圆心坐标
# P1x, P1y, P1z = TB[0], TB[1], TB[2]  # 第一个已知点坐标
# P2x, P2y, P2z = TC[0], TC[1], TC[2]  # 第二个已知点坐标

# # 计算圆面法线
# v1 = np.array([P1x - Cx, P1y - Cy, P1z - Cz])
# v2 = np.array([P2x - Cx, P2y - Cy, P2z - Cz])
# normal = np.cross(v1, v2)
# normal /= np.linalg.norm(normal)
# pdb.set_trace()
# # 计算圆的半径
# R = np.sqrt((P1x - Cx)**2 + (P1y - Cy)**2 + (P1z - Cz)**2)
# # print(R)
# # 生成圆周上的点坐标
# num_points = 100  # 生成100个点
# theta = np.linspace(0, 2 * np.pi, num_points)
# x = Cx + R * np.cos(theta)
# y = Cy + R * np.sin(theta)
# z = Cz + np.zeros_like(x)

# # 将圆面法线应用到所有点
# rotation_matrix = np.eye(3) - normal[:, np.newaxis] * normal
# rotated_points = np.dot(rotation_matrix, np.vstack([x, y, z]))

# # 绘制圆
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(rotated_points[0], rotated_points[1], rotated_points[2])

# # 设置坐标轴标签
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')

# # 显示图形
# plt.show()


