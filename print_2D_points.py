import matplotlib.pyplot as plt

with open("input.in") as f:
    data = f.read()

data = data.split('\n')[1:-2]

x = [row.split(' ')[0] for row in data]
y = [row.split(' ')[1] for row in data]

fig = plt.figure()
ax = plt.subplot()

ax.set_title("2D Data Points")
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.grid(linewidth=2)
ax.plot(x, y, 'ro')

leg = ax.legend()

plt.show()
