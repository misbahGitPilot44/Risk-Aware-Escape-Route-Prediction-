
# route_planner.py

from astar import astar


def path_cost(path):
    """Calculate ordinary movement cost."""
    if not path:
        return float("inf")

    return len(path) - 1


def find_best_route(environment):
    """
    Evaluate all exits and return the best reachable route.

    Returns a dictionary containing:
    - path
    - exit
    - cost
    """

    best_result = None

    for exit_position in environment.exits:
        path = astar(
            environment,
            environment.start,
            exit_position
        )

        if path is None:
            print(f"Exit {exit_position}: unreachable")
            continue

        cost = path_cost(path)

        print(
            f"Exit {exit_position}: "
            f"distance={cost}, cells={len(path)}"
        )

        if best_result is None or cost < best_result["cost"]:
            best_result = {
                "path": path,
                "exit": exit_position,
                "cost": cost,
            }

    return best_result
