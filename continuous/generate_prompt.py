import numpy as np

np.random.seed(0)
samples = []
for i in range(10):
    samples.append("{:.3f}".format(np.random.exponential(0.5)))
prompt = [f"<output>{sample}</output>\n" for sample in samples]
prompt = "".join(prompt)
prompt = prompt + "<output>"
print(prompt)