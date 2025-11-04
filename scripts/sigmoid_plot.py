# plot sigmoid function

import numpy as np
import matplotlib.pyplot as plt

# Define the function
x = np.linspace(-10, 10, 400)
y = 1 / (1 + np.exp(-x))

# Plot
plt.figure(figsize=(8, 5))
plt.plot(x, y, linewidth=2)
plt.title(r'Plot of $f(x) = \frac{1}{1 + e^{-x}}$ (Sigmoid Function)')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)
plt.show()