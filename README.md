# chapter 1:Using neural nets to recognize handwritten digits

## Introduction:
Neural networks approach the problem in a different way. The idea is to take a large number of handwritten digits, known as training examples,neural network uses the examples to automatically infer rules for recognizing handwritten digits. Furthermore, by increasing the number of training examples, the network can learn more about handwriting, and so improve its accuracy

## Perceptrons:
it is a type of artificial neuron.
### how the perceptrons work:
the perceptron has three inputs, x1,x2,x3.In general it could have more or fewer inputs. Rosenblatt proposed a simple rule to compute the output. He introduced weights, w1,w2,…, real numbers expressing the importance of the respective inputs to the output. The neuron's output, 0 or 1, is determined by whether the weighted sum ∑jwjxj is less than or greater than some threshold value. Just like the weights, the threshold is a real number which is a parameter of the neuron. To put it in more precise algebraic terms:
# output={0 if ∑jwjxj≤ threshold
#         1 if ∑jwjxj> threshold }
