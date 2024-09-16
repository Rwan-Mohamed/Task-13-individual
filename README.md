# chapter 1:Using neural nets to recognize handwritten digits

## Introduction:
Neural networks approach the problem in a different way. The idea is to take a large number of handwritten digits, known as training examples,neural network uses the examples to automatically infer rules for recognizing handwritten digits. Furthermore, by increasing the number of training examples, the network can learn more about handwriting, and so improve its accuracy

## Perceptrons:
it is a type of artificial neuron.
### how the perceptrons work:
the perceptron has three inputs, x1,x2,x3.In general it could have more or fewer inputs. Rosenblatt proposed a simple rule to compute the output. He introduced weights, w1,w2,…, real numbers expressing the importance of the respective inputs to the output. The neuron's output, 0 or 1, is determined by whether the weighted sum ∑jwjxj is less than or greater than some threshold value. Just like the weights, the threshold is a real number which is a parameter of the neuron. To put it in more precise algebraic terms:
### output={0 if ∑jwjxj≤ threshold
###         1 if ∑jwjxj> threshold }

That's the basic mathematical model. A way you can think about the perceptron is that it's a device that makes decisions by weighing up evidence and by varying the weights and the threshold, we can get different models of decision-making
if we have two layers,the first layer of perceptrons - is making three very simple decisions, by weighing the input evidence. What about the perceptrons in the second layer? Each of those perceptrons is making a decision by weighing up the results from the first layer of decision-making. In this way a perceptron in the second layer can make a decision at a more complex and more abstract level than perceptrons in the first layer etc
### output={0 if w⋅x+b≤0
###         1 if w⋅x+b>0} 
 where w and x are vectors whose components are the weights and inputs and we use bias instead of threshold where b≡−threshold.You can think of the bias as a measure of how easy it is to get the perceptron to output a 1.
 
--> Another way perceptrons can be used is to compute the elementary logical functions we usually think of as underlying computation, functions such as AND, OR, and NAND as we can do NAND gate by having a perceptron with two inputs, each with weight −2, and an overall bias of 3 so we can use networks of perceptrons to compute any logical function at all as the NAND gate is universal for computation, that is, we can build any computation up out of NAND gates.

## Sigmoid neurons:
Sigmoid neurons are similar to perceptrons, but modified so that small changes in their weights and bias cause only a small change in their output.Just like a perceptron, the sigmoid neuron has inputs, x1,x2,…. But instead of being just 0 or 1, these inputs can also take on any values between 0and 1.So, for instance, 0.638 is a valid input for a sigmoid neuron. Also just like a perceptron, the sigmoid neuron has weights for each input, w1,w2,…,and an overall bias,b.But the output is not 0or 1.Instead, it's *σ(w⋅x+b)* , where σ is called the sigmoid function.

--->Incidentally, σ is sometimes called the logistic function, and this new class of neurons called logistic neurons. It's useful to remember this terminology, since these terms are used by many people working with neural nets. However, we'll stick with the sigmoid terminology., and is defined by:
### σ(z)≡1/1+(e^−z)
To understand the similarity to the perceptron model, suppose z≡w⋅x+bis a large positive number. Then e−z≈0and so σ(z)≈1 . In other words, when z=w⋅x+bis large and positive, the output from the sigmoid neuron is approximately 1, just as it would have been for perceptron. Suppose on the other hand that z=w⋅x+b is very negative. Then e−z→∞ , and σ(z)≈0. So when z=w⋅x+b is very negative, the behaviour of a sigmoid neuron also closely approximates a perceptron. It's only when w⋅x+b is of modest size that there's much deviation from the perceptron model.

How should we interpret the output from a sigmoid neuron? Obviously, one big difference between perceptrons and sigmoid neurons is that sigmoid neurons don't just output 0 or 1. They can have as output any real number between 0 and 1, so values such as 0.173… and 0.689 are legitimate outputs. This can be useful, for example, if we want to use the output value to represent the average intensity of the pixels in an image input to a neural network. But sometimes it can be a nuisance. Suppose we want the output from the network to indicate either "the input image is a 9" or "the input image is not a 9". Obviously, it'd be easiest to do this if the output was a 0 or a 1, as in a perceptron. But in practice we can set up a convention to deal with this, for example, by deciding to interpret an output of at least 0.5 as indicating a "9", and any output less than 0.5 as indicating "not a 9"  

## A simple network to classify handwritten digits:
To recognize individual digits we will use a three-layer neural network:The input layer of the network contains neurons encoding the values of the input pixels. As discussed in the next section, our training data for the network will consist of many 28 by 28 pixel images of scanned handwritten digits, and so the input layer contains 784=28×28 neurons. For simplicity I've omitted most of the 784 input neurons in the diagram above. The input pixels are greyscale, with a value of 0.0 representing white, a value of 1.0 representing black, and in between values representing gradually darkening shades of grey.The second layer of the network is a hidden layer. We denote the number of neurons in this hidden layer by n, and we'll experiment with different values for n. The example shown illustrates a small hidden layer, containing just n=15 neurons.The output layer of the network contains 10 neurons. If the first neuron fires, i.e., has an output ≈1, then that will indicate that the network thinks the digit is a 0. If the second neuron fires then that will indicate that the network thinks the digit is a 1. And so on. A little more precisely, we number the output neurons from 0 through 9, and figure out which neuron has the highest activation value. If that neuron is, say, neuron number 6, then our network will guess that the input digit was a 6. And so on for the other output neurons.

## cost function:
Now that we have a design for our neural network, how can it learn to recognize digits? The first thing we'll need is a data set to learn from - a so-called training data set. We'll use the MNIST data set, which contains tens of thousands of scanned images of handwritten digits, together with their correct classifications. MNIST's name comes from the fact that it is a modified subset of two data sets

---> *The MNIST data* comes in two parts. The first part contains 60,000 images to be used as training data. These images are scanned handwriting samples from 250 people, half of whom were US Census Bureau employees, and half of whom were high school students. The images are greyscale and 28 by 28 pixels in size. The second part of the MNIST data set is 10,000 images to be used as test data. Again, these are 28 by 28 greyscale images. We'll use the test data to evaluate how well our neural network has learned to recognize digits.
To make this a good test of performance, the test data was taken from a different set of 250 people than the original training data (albeit still a group split between Census Bureau employees and high school students).

What we'd like is an algorithm which lets us find weights and biases so that the output from the network approximates y(x) for all training inputs x. To quantify how well we're achieving this goal we define a cost function
### C(w,b)≡1/2n∑x∥y(x)−a∥^2
We'll call C the quadratic cost function; it's also sometimes known as the mean squared error or just MSE. Inspecting the form of the quadratic cost function, we see that C(w,b) is non-negative, since every term in the sum is non-negative. Furthermore, the cost C(w,b) becomes small, i.e., C(w,b)≈0, precisely when y(x) is approximately equal to the output, a, for all training inputs, x.

## Learning with gradient descent:
*Goal:* The aim is to minimize the cost function (or loss function), which measures the difference between the predicted output and the actual target. For a neural network, this is usually done through iterative optimization.
*Gradient Descent Process:*
1.Start with an initial guess for the network's weights (randomly initialized).
2.For each iteration, the weights are adjusted in the direction that decreases the cost function the most, based on the gradient (the slope or rate of change) of the cost function with respect to the weights.
3.This adjustment is proportional to the learning rate, a hyperparameter that controls the size of the step taken in each iteration.
*Mathematical Description:*
The update rule for the weights w is given by:
### 𝑤←𝑤−𝜂∇𝐶(𝑤)
Where:
         η is the learning rate.
         ∇C(w) is the gradient of the cost function with respect to the weights.
*Importance of the Gradient:* The gradient gives information on how the cost changes when the weights are changed. By following the direction of the negative gradient, we move toward reducing the cost.
*Challenges:*
1.Choosing the Learning Rate: If the learning rate is too small, learning will be slow; if too large, the model may fail to converge and overshoot the minimum.
2.Local Minima: Gradient descent may get stuck in local minima or saddle points instead of finding the global minimum, especially in complex cost landscapes like those in neural networks.

## Backpropagation:
The backpropagation algorithm, essential for training deep networks, is briefly introduced. It efficiently computes gradients of the cost function with respect to the weights by applying the chain rule of calculus. This allows for weight updates throughout the entire network by propagating the error backward from the output layer to earlier layers.
