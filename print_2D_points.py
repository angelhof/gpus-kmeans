import matplotlib.pyplot as plt

with open("data/dataset_elki_500.in") as f:
    data = f.read()

with open("centers.out") as f:
    centers_data = f.read()

centers_lines = centers_data[:-1].split('\n')

centers = map(lambda x: map(float, x.split()), centers_lines)

data = data.split('\n')[1:-2]

x = [float(row.split(' ')[0]) for row in data]
y = [float(row.split(' ')[1]) for row in data]

x_c = [row[0] for row in centers]
y_c = [row[1] for row in centers]

fig = plt.figure()
ax = plt.subplot()

ax.set_title("2D Data Points")
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.grid(linewidth=2)
ax.plot(x, y, 'ro')
ax.plot(x_c, y_c, 'bo')

leg = ax.legend()

plt.show()
