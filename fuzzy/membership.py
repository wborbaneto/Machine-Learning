import numpy as np
import matplotlib.pyplot as plt


def triangle(points=None, res=100):
	""" Given a,b,c points returns triangular functions.
		Parameters
		----------
		points : np.array([][], type = float64), default = example
			An array containing the points for the functions.
		res : int, default = 100
			Number of desired point in the function

		Returns
		-------
		xx,yy :  np.array([][], type = float64)
			Arrays containing the x and y values in each row.

		Examples
		-------
		1. Triangle
		# Three Triangles with its a,b,c points. Each column represents a
		# triangle and each row represents one point.
		p = [[0.00, 0.25, 0.50], #[a_1, b_1, c_1]
			[0.25, 0.50, 0.75],  #[a_2, b_2, c_2]
			[0.50, 0.75, 1.00]]  #[a_3, b_3, c_1]

		x,y = triangle(points=p, res=100)
		# Plot only first triangle
		plt.plot(x[0,:], y[0,:])
		# Plot all triangles
		plt.plot(x,y)

		"""
	# Holders for xx and yy values.
	xx = np.zeros((res,))
	yy = np.zeros((res,))

	for p in points:
		# Creating the x and y values for the function.
		x = np.linspace(p[0], p[2], res)
		y = np.zeros(x.shape)

		# Sectioning the first part (fp -> raising) and second part(sp ->
		# falling) for the triangle.
		fp = np.logical_and(p[0] < x, x <= p[1])
		sp = np.logical_and(p[1] < x, x < p[2])

		# Creating a linear function for each part.
		y[fp] = (x[fp] - p[0]) / (p[1] - p[0])
		y[sp] = (p[2] - x[sp]) / (p[2] - p[1])

		# Saving each triangle in a row.
		xx = np.c_[xx, x]
		yy = np.c_[yy, y]
	return xx, yy


def trapeze(points=None, res=100):
	""" Given a,b,c,d points returns trapezoidal functions.
		Parameters
		----------
		points : np.array([][], type = float64), default = example
			An array containing the points for the functions.
		res : int, default = 100
			Number of desired point in the function

		Returns
		-------
		xx,yy :  np.array([][], type = float64)
			Arrays containing the x and y values in each row.

		Examples
		-------
		1. Trapeze
		# Three Trapezes with its a,b,c points. Each column represents a
		# triangle and each row represents one point.
		p = [[0.00, 0.10, 0.30, 0.40], #[a_1, b_1, c_1, d_1]
			[0.30, 0.40, 0.60, 0.70],  #[a_2, b_2, c_2, d_2]
			[0.60, 0.70, 0.90, 1.00]]  #[a_3, b_3, c_3, d_3]

		x,y = trapeze(points=p, res=100)
		# Plot only first trapeze
		plt.plot(x[0,:], y[0,:])
		# Plot all trapezes
		plt.plot(x,y)

	"""
	# Holders for xx and yy values.
	xx = np.zeros((res,))
	yy = np.zeros((res,))

	for p in points:
		# Creating the x and y values for the function.
		x = np.linspace(p[0], p[3], res)
		y = np.zeros(x.shape)

		# Sectioning the first part (fp -> raising), second part(sp ->
		# constant) and third part (tp -> falling) for the trapeze
		fp = np.logical_and(p[0] < x, x <= p[1])
		sp = np.logical_and(p[1] < x, x <= p[2])
		tp = np.logical_and(p[2] < x, x < p[3])

		# Creating a linear function for each part.
		y[fp] = (x[fp] - p[0]) / (p[1] - p[0])
		y[sp] = 1
		y[tp] = (p[3] - x[tp]) / (p[3] - p[2])

		# Saving each trapeze in a row.
		xx = np.c_[xx, x]
		yy = np.c_[yy, y]
	return xx, yy


def gaussian(points, res=100):
	""" Given mean and deviation points returns gaussian functions.
		Parameters
		----------
		points : np.array([][], type = float64), default = example
			An array containing the mean and deviation for the functions.
		res : int, default = 100
			Number of desired point in the function

		Returns
		-------
		xx,yy :  np.array([][], type = float64)
			Arrays containing the x and y values in each row.

		Examples
		-------
		1. Bell
		# Three bells with its mean and variance values. Each column
		represents a bell and each row represents its mean and variance.
		p = [[0.25, 0.1], # [mean_1, variance_1]
			[0.50, 0.1],  # [mean_2, variance_2]
			[0.75, 0.1]]  # [mean_3, variance_3]

		x,y = gaussian(points=p, res=100)
		# Plot only first gaussian.
		plt.plot(x[0,:], y[0,:])
		# Plot all gaussians.
		plt.plot(x,y)

	"""
	# Holders for xx and yy values.
	xx = np.zeros((res,))
	yy = np.zeros((res,))

	# Creating the x values for the function.
	x = np.linspace(0, 1, res)

	for p in points:
		# Calculating gaussian function.
		y = np.exp(-np.power(x - p[0], 2.) / (2 * np.power(p[1], 2.)))

		# Saving each gaussian in a row.
		xx = np.c_[xx, x]
		yy = np.c_[yy, y]
	return xx, yy


def membership_func(points=None, type='triangle'):
	"""Return membership functions given its points and type.
		Parameters
		----------
		points : np.array([][], type = float64), default = example
			An array containing the points for the functions.
		type : str, default = 'triangle'
			Type of the desired membership function.

		Returns
		-------
		membership_function :  np.array([][], type = float64)
			Desired functions saved in an array

		Examples
		-------
		1. Triangle
		# Three Triangles with its a,b,c points. Each column represents a
		# triangle and each row represents one point.
		p = [[0.00, 0.25, 0.50], #[a_1, b_1, c_1]
			[0.25, 0.50, 0.75],  #[a_2, b_2, c_2]
			[0.50, 0.75, 1.00]]  #[a_3, b_3, c_3]

		membership_func(points = p,type='triangle')

		2. Trapeze
		# Three Trapezes with its a,b,c,d points. Each column represents a
		# trapeze and each row represents one point.
		p = [[0.00, 0.10, 0.30, 0.40], #[a_1, b_1, c_1, d_1]
			[0.30, 0.40, 0.60, 0.70],  #[a_2, b_2, c_2, d_2]
			[0.60, 0.70, 0.90, 1.00]]  #[a_3, b_3, c_3, d_3]

		membership_func(points = p,type='trapeze')

		3. Bell
		# Three bells with its mean and variance values. Each column
		represents a bell and each row represents its mean and variance.
		p = [[0.25, 0.1], # [mean_1, variance_1]
			[0.50, 0.1],  # [mean_2, variance_2]
			[0.75, 0.1]]  # [mean_3, variance_3]

		membership_func(points = p,type='bell')

	"""
	if type == 'triangle':
		points = points if points is not None else [[0.00, 0.25, 0.50],
													[0.25, 0.50, 0.75],
													[0.50, 0.75, 1.00]]
		return triangle(points)

	elif type == 'trapeze':
		points = points if points is not None else [[0.00, 0.10, 0.30, 0.40],
													[0.30, 0.40, 0.60, 0.70],
													[0.60, 0.70, 0.90, 1.00]]
		return trapeze(points, res=100)

	elif type == 'bell':
		points = points if points is not None else [[0.25, 0.1],
													[0.50, 0.1],
													[0.75, 0.1]]
		return gaussian(points, res=100)


#if "__name__" == "__main__":

p = [[0.0, 0.2, 0.4],
	[0.2, 0.4, 0.6],
	[0.4, 0.6, 0.8],
	[0.6, 0.8, 1.0]]

plt.figure(1)
x_trig, y_trig = membership_func(points=p, type='triangle')
plt.plot(x_trig, y_trig)
plt.grid()

p = [[0.0, 0.1, 0.2, 0.3],
	[0.2, 0.3, 0.4, 0.5],
	[0.5, 0.6, 0.7, 0.8],
	[0.7, 0.8, 0.9, 1.0]]

plt.figure(2)
x_trap, y_trap = membership_func(points=p, type='trapeze')
plt.plot(x_trap, y_trap)
plt.grid()

plt.figure(3)
# Example using the standard values embedded into the function
x_gauss, y_gauss = membership_func(type='bell')
plt.plot(x_gauss, y_gauss)
plt.grid()

