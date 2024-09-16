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

