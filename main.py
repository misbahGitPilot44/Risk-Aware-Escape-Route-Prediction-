
# # main.py

## Test 1

# # The symbols mean:

# ## S — starting position
# ## E — emergency exit
# ## — wall
# ## . — traversable floor

# from environment import Environment
# from config import WALL, EXIT

# environment = Environment()

# print("Environment initialized successfully!")
# print("Grid dimensions:", environment.grid.shape)
# print("Starting position:", environment.start)
# print("Emergency exits:", environment.exits)

# wall_count = (environment.grid == WALL).sum()
# exit_count = (environment.grid == EXIT).sum()

# print("Number of wall cells:", wall_count)
# print("Number of exits:", exit_count)

# print("\nBuilding layout:")
# for row in range(environment.rows):
#     line = ""

#     for col in range(environment.cols):
#         position = (row, col)

#         if position == environment.start:
#             line += "S "
#         elif environment.grid[row, col] == WALL:
#             line += "# "
#         elif environment.grid[row, col] == EXIT:
#             line += "E "
#         else:
#             line += ". "

#     print(line)


# main.py 

## Test 2

from environment import Environment
from route_planner import find_best_route

environment = Environment()

print("Environment initialized successfully!")
print("Grid dimensions:", environment.grid.shape)
print("Starting position:", environment.start)
print("Emergency exits:", environment.exits)

result = find_best_route(environment)

print("\n--- BEST ESCAPE ROUTE ---")

if result is None:
    print("No reachable exit found.")
else:
    print("Selected exit:", result["exit"])
    print("Path length:", len(result["path"]))
    print("Movement cost:", result["cost"])
    print("Path coordinates:")

    print(result["path"])
