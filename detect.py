# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# %%
def create_exp(N, A, B):
    expo = A*np.exp(-B*np.arange(0,N))
    return expo
def create_lin(N, a, b):
    lin = a*np.arange(0,N) + b
    return lin
def concat_noise(C, m = 0, s = 1):
    # import numpy as np
    concat = []
    for c in C:
        concat.extend(c)
    X = np.arange(0,len(concat))
    Y = concat +  np.random.normal(loc=m, scale=s, size=len(concat))
    return X, Y

def RMSE(Ypred, Yreal):
    rmse = np.sqrt(np.sum(np.square(Ypred-Yreal))/np.shape(Ypred)[0])
    return rmse
def fit_and_pred(X, y):
    Xarr = np.array(X).reshape(-1,1)
    yarr = np.array(y).reshape(-1,1)
    reg = LinearRegression().fit(Xarr, yarr)
    pred = reg.predict(Xarr)
    rmse = RMSE(pred, yarr)
    return rmse
def sliding_window_pred(X, y, window, lag):
    Xmax = np.shape(X)[0]-window+1
    start = np.arange(0, Xmax, lag)
    score = [fit_and_pred(X[s:s+window], y[s:s+window]) 
            for s in start]    
    return score


# %% 
expo = create_exp(N = 15000, A = 10 , B = 0.0001)
lin = create_lin(N = 15000, a = -0.00005 , b = expo[-1]-0.01*expo[-1])
print(expo)
print(lin)

# %% low noise
X, Y = concat_noise(C = [expo,lin], m = 0, s = 0.001)

# %%
plt.scatter(X,Y)
plt.show()

plt.scatter(X[0:1000],Y[0:1000])
plt.show()

# %%
score = sliding_window_pred(X = X, y = Y, 
                            window = 100, 
                            lag = 10)
plt.bar(np.arange(0,np.shape(score)[0]),score)
plt.show()


# %%
score = sliding_window_pred(X = X, y = Y,
                            window = 10, 
                            lag = 1)
plt.bar(np.arange(0,np.shape(score)[0]),score)
plt.show()


# %%
score = sliding_window_pred(X = X, y = Y, 
                            window = 1000, 
                            lag = 100)
plt.bar(np.arange(0,np.shape(score)[0]),score)
plt.show()


# %% more noisy data
X, Y = concat_noise(C = [expo,lin], m = 0, s = 0.1)

# %%
plt.scatter(X,Y)
plt.show()

plt.scatter(X[0:1000],Y[0:1000])
plt.show()

# %%
score = sliding_window_pred(X = X, y = Y, 
                            window = 2787, 
                            lag = 77)
plt.bar(np.arange(0,np.shape(score)[0]),score)
plt.show()


# %%

for i in [2111, 3111, 4811, 5877, 6854]:
    for j in [11, 59, 111, 337, 511, 777]:
        score = sliding_window_pred(X = X, y = Y, 
                                    window = i, 
                                    lag = j)
        deltascore = np.round(np.max(np.diff(score))/0.001,3)
        print("\n for window : {} and lag : {} \n delta max diff score is : {}".format(i, j, deltascore))

# %%
score = sliding_window_pred(X = X, y = Y, 
                            window = 6854, 
                            lag = 111)
plt.bar(np.arange(0,np.shape(score)[0]),score)
plt.show()



plt.scatter(np.arange(0,np.shape(np.diff(score))[0]),np.diff(score))
plt.show()

# %%
score = sliding_window_pred(X = X, y = Y, 
                            window = 3111, 
                            lag = 337)
plt.bar(np.arange(0,np.shape(score)[0]),score)
plt.show()

plt.scatter(np.arange(0,np.shape(np.diff(score))[0]),np.diff(score))
plt.show()


# %% still more noisy data
X, Y = concat_noise(C = [expo,lin], m = 0, s = 0.3)

# %%
plt.scatter(X,Y)
plt.show()

plt.scatter(X[0:1000],Y[0:1000])
plt.show()


# %%

for i in [2111, 3111, 4811, 5877, 6854]:
    for j in [11, 59, 111, 337, 511, 777, 1279, 2447]:
        score = sliding_window_pred(X = X, y = Y, 
                                    window = i, 
                                    lag = j)
        deltascore = np.round(np.max(np.diff(score)-np.mean(np.diff(score)))/0.001,3)
        print("\n for window : {} and lag : {} \n delta max diff score is : {}".format(i, j, deltascore))

# %%
score = sliding_window_pred(X = X, y = Y, 
                            window = 5877, 
                            lag = 1279)
plt.bar(np.arange(0,np.shape(score)[0]),score)
plt.show()



plt.scatter(np.arange(0,np.shape(np.diff(score))[0]),np.diff(score))
plt.show()

# %%
score = sliding_window_pred(X = X, y = Y, 
                            window = 10000, 
                            lag = 5000)
plt.bar(np.arange(0,np.shape(score)[0]),score)
plt.show()

plt.scatter(np.arange(0,np.shape(np.diff(score))[0]),np.diff(score)-np.mean(np.diff(score)))
plt.show()
