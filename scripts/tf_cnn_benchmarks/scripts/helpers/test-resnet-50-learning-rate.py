import tensorflow as tf
import sys

class Model(object):

    def __init__(self):
        #
        # The ResNet paper uses a starting learning rate of of 0.1 
        # at batch size 256.
        #
        self.images = 1281167
        self.r = 0.128
        self.g = 8
        self.b = 256

    def piecewise_constant(self, step, boundaries, values):
       if len(boundaries) != 4:
           raise ValueError("Fatal error")

       if step <= boundaries[0]:
           return values[0]
       elif step > boundaries[0] and step <= boundaries[1]:
           return values[1]
       elif step > boundaries[1] and step <= boundaries[2]:
           return values[2]
       elif step > boundaries[2] and step <= boundaries[3]:
           return values[3]
       elif step > boundaries[3]:
           return value[4]
    
    def get_learning_rate(self, step, b):

        scaled = self.get_scaled_learning_rate(b)
        # Compute total number of batches
        n = (self.images / b)
        boundaries = [int(n * x) for x in [30, 60, 80, 90]]
        values = [1, 0.1, 0.01, 0.001, 0.0001]
        values = [scaled * v for v in values]
        
        rate = self.piecewise_constant(step, boundaries, values)

        # Warm up the first 5 epochs
        warmup_steps = int(n * 5)
        warmup_rate  = (scaled * float(step) / float(warmup_steps))

        if (step >= warmup_steps):
            # print("%d: %.5f" % (step, rate))
            return rate
        else:
            # print("%d: %.5f" % (step, warmup_rate))
            return warmup_rate
    
    def get_scaled_learning_rate(self, b):
        """
        Calculates base learning rate.

        In replicated mode, gradients are summed rather than averaged which, with
        the SGD and momentum optimizers, increases the effective learning rate by
        xG, where G is the number of GPU devices. Dividing the base learning rate 
        by G negates the increase.

        Args:
            b: Total batch size.
        
        Returns:
            Base learning rate to use to create learning rate schedule.
        """
        base = self.r / self.g
        scaled = base * (b / self.b)
        return scaled

    def simulate(self, epochs, b):
        n = self.images // b
        steps = epochs * n
        print("Simulate %d steps of batch size %d" % (steps, b))
        for i in range(steps):
            rate = self.get_learning_rate(i + 1, b) * self.g
            print("%7d %8.5f: %8.5f" % (i + 1, float(i + 1) / float(n), rate))


if __name__ == "__main__":
    # My code
    model = Model()
    b = 512
    epochs = 90
    print("Simulate")
    model.simulate(epochs, b)
    sys.exit(0)
