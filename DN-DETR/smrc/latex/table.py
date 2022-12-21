import scipy
# pip install tabulate
from tabulate import tabulate

a = scipy.random.rand(3, 3)

print(tabulate(a, tablefmt="latex", floatfmt=".2f"))

# other solutions
# https://pypi.org/project/PyLaTeX/0.4.1/