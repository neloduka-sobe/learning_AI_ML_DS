#!/usr/bin/env python3
import numpy as np

train_x = np.array([1.0,2.0])
train_y = np.array([200.0,400.0])

def gradient(a_x, a_y, w, b):

	m = a_x.shape[0]
	dj_dw = 0
	dj_db = 0
	
	for i in range(m):
		f_wb = w * a_x[i] + b
		dj_dw_i = (f_wb - a_y[i]) * a_x[i]
		dj_db_i = f_wb - a_y[i] 
		dj_db += dj_db_i
		dj_dw += dj_dw_i
	dj_dw = dj_dw / m 
	dj_db = dj_db / m 
	
	return dj_dw, dj_db

def gradient_descent(x, y, w, b, alpha, num_iters, gradient_function):

	for i in range(num_iters):
		dj_dw, dj_db = gradient_function(x, y, w, b)

		b = b - alpha * dj_db
		w = w - alpha * dj_dw
	return w, b

w = 0
b = 0
tmp_alpha = 1.0e-2

iterations = 100000
w_final, b_final = gradient_descent(train_x, train_y, w, b, tmp_alpha, iterations, gradient)	
print(f"(w,b) found by gradient descent: ({w_final},{b_final})")
