from matplotlib import pyplot as plt


def plot401kInterest(a, t):
	#@a: The amount of each monthly payment
	ys = [(a * sum([1.05**t_j for t_j in range(t_i)])) for t_i in range(t)]
	xs = [i for i in range(len(ys))]
	print("Final value: {} vs invested: {}".format(ys[-1], a*t))
	plt.plot(xs, ys)
	plt.show()

def plotLoanRepayment(a, t):
	ys = [ (12000 - 3000 * sum([1.04**t_j for t_j in range(t_i)])) for t_i in range(t) ]
	xs = [i for i in range(len(ys))]
	print("Final value: {} vs invested: {}".format(ys[-1], a*t))
	plt.plot(xs, ys)
	plt.show()

plot401kInterest(1500, 36)
#plotLoanRepayment(4000, 12)




