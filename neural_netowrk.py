import numpy as np
import sys
import os

class NeuralNetwork:
	def __init__(self):
		self.synaptic_weights =np.random.random((3,1))
		print "Initial value of weights = \n"+str(self.synaptic_weights)

	def sigmoid(self,x):
		return 1/(1+np.exp(-x))
	
	def sigmoidDerivative(self,x):
		return x*(1-x)
	
	def train(self,training_set_inputs,training_set_outputs,iterations):
		for iteration in range(iterations):
				print "iterations = "+str(iteration)
				output = self.think(training_set_inputs)
				print "Output = \n"+str(output)
				error = training_set_outputs-output
				print "Error = \n"+str(error)
				adjustment = np.dot(training_set_inputs.T,error*self.sigmoidDerivative(output))
				print "Adjustment = \n"+str(adjustment)
				self.synaptic_weights+=adjustment
				print "Updated Weights = \n" + str(self.synaptic_weights)
				os.system('clear')
	
	def think(self,inputs):
		return self.sigmoid(np.dot(inputs,self.synaptic_weights))

def main(argv):
	neural_network = NeuralNetwork()
	training_sets_input = np.array([[0,0,1],[1,1,1],[1,0,1],[0,1,1]])
	training_sets_output = np.array([[0,1,1,0]]).T

	neural_network.train(training_sets_input,training_sets_output,10000)

	print neural_network.think(np.array([[1,0,0]]))

    
if __name__=='__main__':
	main(sys.argv)


