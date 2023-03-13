import numpy as np
from scipy.integrate import odeint, solve_ivp
import matplotlib.pyplot as plt
import math
import pandas as pd
import seaborn as sns
import csv
from math import sin,cos,pi, degrees, radians


dp90 = pd.read_csv('/Users/notyahir/Desktop/Data/10_Data_Fixed2A.csv', header=None)
dp90.head()

time, x1A, y1A, x2A, y2A = (dp90[0]-2.35), dp90[1]+75, dp90[2]-548, dp90[3]+50, dp90[4]-615


def coupledODEs(t,x):
    # constants
    m1 = 0.132
    m2 = m1
    l1 = 558
    l2 = l1
    g = 9810

    # equations
    a = x
    aprime = [0, 0, 0, 0]

    aprime[0] = a[2]

    aprime[1] = a[3]

    aprime[2] = ((-m1 * l1 * (a[3] ** 2) * math.sin(a[0] - a[1]) * math.cos(a[0] - a[1])) + (m2 * g * math.sin(a[1]) * \
                                                                                             math.cos(a[0] - a[1])) - (
                             m2 * l2 * (a[3] ** 2) * math.sin(a[0] - a[1])) - ((m1 + m2) * g * math.sin(a[0]))) \
                / (l1 * (m1 + m2) - m2 * l1 * (math.cos(a[0] - a[1]) ** 2))

    aprime[3] = ((m2 * l2 * (a[3]**2)) * math.sin(a[0]-a[1]) * math.cos(a[0]-a[1]) + g * math.sin(a[0]) *\
             math.cos(a[0]-a[1])*(m1+m2) + l1 * a[2]**2 * math.sin(a[0]-a[1]) * (m1+m2) - g * math.sin(a[1]) *\
             (m1 + m2)) / (l2*(m1+m2)-m2 * l2 * math.cos(a[0]-a[1])**2)
    return aprime


def coupledODEsApproxed(t,x):
    # constants
    m1 = 0.132
    m2 = m1
    l1 = 558
    l2 = l1
    g = 9810

    # equations
    b = x
    bprime = [0, 0, 0, 0]

    bprime[0] = b[2]

    bprime[1] = b[3]

    bprime[2] = 2*(g/l1)*((1/2)*b[1]-b[0])

    bprime[3] = (b[0]-b[1])*(2*(g/l1))
    return bprime


# t = np.linspace(0,10,100)
# solution = odeint(coupledODEs, a0, t)
# # print(solution)


t_span = (0, 10,)
initial = [0.2356194, 0, 0, 0]

solution_better_real = solve_ivp(coupledODEs, t_span, y0=initial, vectorized=True, rtol=1e-6,atol=1e-6)
solution_better_approx = solve_ivp(coupledODEsApproxed, t_span, y0=initial, vectorized=True, rtol=1e-6,atol=1e-6)


# Plotting (x,y) for the first and second masses
l = 558
x1, y1 = l*np.sin(solution_better_real.y[0]), -l*np.cos(solution_better_real.y[0])
x2, y2 = l*np.sin(solution_better_real.y[1])+x1, -l*np.cos(solution_better_real.y[1])+y1

x1AP, y1AP = l*np.sin(solution_better_approx.y[0]), -l*np.cos(solution_better_approx.y[0])
x2AP, y2AP = l*np.sin(solution_better_approx.y[1])+x1AP, -l*np.cos(solution_better_approx.y[1])+y1AP
plt.figure()
plt.plot(solution_better_real.t, x1, color='red')


plt.plot(solution_better_approx.t,x1AP, color='green')
plt.plot(time,x1A, color='blue')
plt.ylabel("x1 Position (mm)")
plt.xlabel("Time (s)")
plt.show()
