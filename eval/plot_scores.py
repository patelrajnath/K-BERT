import json
import ast
from matplotlib import pyplot as plt

with open('../outputs/logs/dev.scores') as dev, open('../outputs/logs/test.scores') as test:
    dev_f1_list = []
    test_f1_list = []
    for l1, l2 in zip(dev, test):
        dev_f1 = ast.literal_eval(l1)
        test_f1 = ast.literal_eval(l2)
        dev_f1_list.append(dev_f1['f1'])
        test_f1_list.append(test_f1['f1'])

plt.plot(range(len(dev_f1_list)), dev_f1_list, label="Dev f1")
plt.plot(range(len(test_f1_list)), test_f1_list, label="Test f1")

plt.xlabel('x - axis')
# Set the y axis label of the current axis.
plt.ylabel('y - axis')
# Set a title of the current axes.
plt.title('Two or more lines on same plot with suitable legends ')
# show a legend on the plot
plt.legend()
# Display a figure.
plt.show()