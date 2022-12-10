# some code reference from https://zhuanlan.zhihu.com/p/157858995

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score


def decision_function_output(i):
    global m, b
    return np.sum(
        [alpha[j] * target[j] * kernel(point[j], point[i])
         for j in range(m)]) - b


def svm_output(alphas, target, kernel, X_train, x_test, b):
    result = (alphas * target) @ kernel(X_train, x_test) - b
    return result


def linear_kernel(x, y, b=1):
    # linear_kernel
    result = x @ y.T + b
    return result


def gaussian_kernel(x, y, sigma=1):
    # gaussian_kernel
    if np.ndim(x) == 1 and np.ndim(y) == 1:
        result = np.exp(-(np.linalg.norm(x - y, 2))**2 / (2 * sigma**2))
    elif (np.ndim(x) > 1 and np.ndim(y) == 1) or (np.ndim(x) == 1
                                                  and np.ndim(y) > 1):
        result = np.exp(-(np.linalg.norm(x - y, 2, axis=1)**2) /
                        (2 * sigma**2))
    elif np.ndim(x) > 1 and np.ndim(y) > 1:
        result = np.exp(-(np.linalg.norm(
            x[:, np.newaxis] - y[np.newaxis, :], 2, axis=2)**2) /
                        (2 * sigma**2))
    return result


def get_error(i1):
    if 0 < alpha[i1] < C:
        return errors[i1]
    else:
        return decision_function_output(i1) - target[i1]


C = 20
b = 0
target = email_dataset.iloc[500:1500, :]["LABEL"]
target[target == 0] = -1
target = np.array(target)
point = np.array(email_dataset.iloc[500:1500, 400:1400])
m, n = np.shape(point)
tol = 0.01
eps = 0.01
alpha = [0 for _ in range(len(point))]
kernel = linear_kernel
errors = svm_output(alpha, target, kernel, point, point, b) - target


def takeStep(i1, i2):
    global b
    if i1 == i2:
        return 0
    alph1 = alpha[i1]
    y1 = target[i1]
    E1 = get_error(i1)
    alph2 = alpha[i2]
    y2 = target[i2]
    E2 = get_error(i2)
    s = y1 * y2
    # Compute L, H
    if (y1 != y2):
        L = max(0, alph2 - alph1)
        H = min(C, C + alph2 - alph1)
    elif (y1 == y2):
        L = max(0, alph1 + alph2 - C)
        H = min(C, alph1 + alph2)
    if L == H:
        return 0
    k11 = kernel(point[i1], point[i1])
    k12 = kernel(point[i1], point[i2])
    k22 = kernel(point[i2], point[i2])
    eta = 2 * k12 - k11 - k22
    if eta < 0:
        a2 = alph2 - y2 * (E1 - E2) / eta
        if a2 < L:
            a2 = L
        elif a2 > H:
            a2 = H
    else:
        f1 = y1 * (E1 + b) - alph1 * k11 - s * alph2 * k12
        f2 = y2 * (E2 + b) - s * alph1 * k12 - alph2 * k22
        L1 = alph1 + s * (alph2 - L)
        H1 = alph1 + s * (alph2 - H)
        Lobj = L1 * f1 + L * f2 + 0.5 * (L1**2) * k11 + 0.5 * (
            L**2) * k22 + s * L * L1 * k12
        Hobj = H1 * f1 + H * f2 + 0.5 * (H1**2) * k11 + 0.5 * (
            H**2) * k22 + s * H * H1 * k12

        if Lobj < Hobj - eps:
            a2 = H
        elif Lobj > Hobj + eps:
            a2 = L
        else:
            a2 = alph2

    if a2 < 1e-8:
        a2 = 0.0
    elif a2 > (C - 1e-8):
        a2 = C

    if (np.abs(a2 - alph2) < eps * (a2 + alph2 + eps)):
        return 0

    a1 = alph1 + s * (alph2 - a2)

    #更新 bias b的值 Equation (J20)
    b1 = E1 + y1 * (a1 - alph1) * k11 + y2 * (a2 - alph2) * k12 + b
    #equation (J21)
    b2 = E2 + y1 * (a1 - alph1) * k12 + y2 * (a2 - alph2) * k22 + b
    # Set new threshoold based on if a1 or a2 is bound by L and/or H
    if 0 < a1 and a1 < C:
        b_new = b1
    elif 0 < a2 and a2 < C:
        b_new = b2
    #Average thresholds if both are bound
    else:
        b_new = (b1 + b2) * 0.5
    #update model threshold
    b = b_new

    #优化完了，更新差值矩阵的对应值
    #同时更新差值矩阵其它值
    errors[i1] = 0
    errors[i2] = 0
    #更新差值 Equation (12)
    for i in range(m):
        if 0 < alpha[i] < C:
            errors[i] += y1 * (a1 - alph1) * kernel(
                point[i1], point[i]) + y2 * (a2 - alph2) * kernel(
                    point[i2], point[i]) + b - b_new
    alpha[i1] = a1
    alpha[i2] = a2
    return 1


def examineExample(i2):
    y2 = target[i2]
    alph2 = alpha[i2]
    E2 = get_error(i2)
    r2 = E2 * y2
    if (r2 < -tol and alph2 < C) or (r2 > tol and alph2 > 0):
        if (len(alpha) - alpha.count(0) - alpha.count(C) > 1):
            if errors[i2] > 0:
                i1 = np.argmin(errors)
            elif errors[i2] <= 0:
                i1 = np.argmax(errors)
            if takeStep(i1, i2):
                return 1
        rand = random.randint(0, len(alpha))
        for i in range(len(alpha)):
            if alpha[(i + rand) % len(alpha)] == 0 or alpha[(i + rand) %
                                                            len(alpha)] == C:
                continue
            if takeStep((i + rand) % len(alpha), i2):
                return 1
        rand = random.randint(0, len(alpha))
        for i in range(len(alpha)):
            i1 = (i + rand) % len(alpha)
            if takeStep(i1, i2):
                return 1
    return 0


def train():
    numChanged = 0
    examineAll = 1
    loopnum = 0
    totaloop = 100
    while (numChanged > 0 or examineAll):
        numChanged = 0
        if loopnum == totaloop:
            break
        else:
            loopnum += 1
            print(f"\rLoops:{loopnum}/{totaloop}", end="")
        if examineAll:
            for i in range(len(point)):
                numChanged != examineExample(i)
        else:
            for i in range(len(alpha)):
                if alpha[i] != 0 or alpha[i] != C:
                    numChanged += examineExample(i)
        if examineAll == 1:
            examineAll = 1
        elif numChanged == 0:
            examineAll = 1
    print(" Finish!")
    return -1


train()

output = svm_output(alpha, target, kernel, point, point, b)
res = []
for i in output:
    if i > 0:
        res.append(1)
    else:
        res.append(-1)
print(target)
print(res)

print(precision_score(target, res))
print(recall_score(target, res))
print(f1_score(target, res))
