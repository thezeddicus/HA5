#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import random
class Memento:
    def __init__(self, state):
        self.__state = state

    @property
    def state(self):
        return self.__state

def get_random_vector(n):
    return [random.random() * 2 - 1 for _ in range(n)]

class NeuralNetwork():
    def __init__(self, weights_count):
        self.weights_count = weights_count
        self.weights = get_random_vector(self.weights_count)

    def train(self):
        random_vector = get_random_vector(self.weights_count)
        for i in range(self.weights_count):
            self.weights[i] += random_vector[i]

    def predict(self, x_test):
        return [
            sum(feature * coef for feature, coef in zip(x, self.weights))
            for x in x_test
        ]

    def evaluate(self, x_test, y_test):
        y_predicted = self.predict(x_test)
        loss_sum = sum(abs(y1 - y2) for y1, y2 in zip(y_predicted, y_test))
        loss_average = loss_sum / self.weights_count
        return loss_average

    def save_weights(self) -> Memento:
       
        return Memento(self.weights)

    def restore_weights(self, memento: Memento):
        
        
        self.weights = memento.state
def main():
     iteration = 0

    weights_count = 1000
    NN = NeuralNetwork(weights_count)

    test_samples = 100
    x_test = [get_random_vector(weights_count) for _ in range(test_samples)]
    y_test = [value * weights_count * 2 for value in get_random_vector(test_samples)]

    epoch_number = 100
    best_loss = weights_count * 2
    history = []
    for epoch in range(epoch_number):
        NN.train()

        loss = NN.evaluate(X_test, y_test)

        if loss < best_loss:
            best_loss = loss
            history.append(NN.save_weights())
            iteration += 1
        else:
            NN.restore_weights(history[-1])

    print(f'Result: {best_loss=}')
    
if __name__ == "__main__":
   main()

