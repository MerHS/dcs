import matplotlib.pyplot as plt

# Define data for each object
object1_x = [1, 2, 3, 4, 5]
object1_y = [2, 3, 4, 3, 2]

object2_x = [2, 3, 4, 5, 6]
object2_y = [3, 4, 5, 6, 7]

# Create a new figure and axis
fig, ax = plt.subplots()

# Plot the trajectories of each object
ax.plot(object1_x, object1_y, label="Object 1")
ax.plot(object2_x, object2_y, label="Object 2")

# Mark the starting and end points for each object
ax.plot(object1_x[0], object1_y[0], marker="o", color="green")
ax.plot(object1_x[-1], object1_y[-1], marker="x", color="red")

ax.plot(object2_x[0], object2_y[0], marker="o", color="green")
ax.plot(object2_x[-1], object2_y[-1], marker="x", color="red")


# Add labels and a legend
ax.set(xlabel="X-axis", ylabel="Y-axis", title="Trajectories")

# Show the plot
plt.show()
