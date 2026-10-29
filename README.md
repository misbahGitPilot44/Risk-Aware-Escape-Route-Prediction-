# Risk-Aware Emergency Escape Route Prediction Using A* Search in Dynamic Environments

A Python-based emergency evacuation simulation that uses **Risk-Aware A* Search** to predict safe escape routes in a dynamic building environment. The system simulates fire propagation, evaluates multiple emergency exits, calculates environmental risk, and dynamically replans evacuation routes whenever the current path becomes unsafe.

The project combines Artificial Intelligence search algorithms, risk modelling, dynamic environment simulation, and real-time visualization to demonstrate intelligent emergency evacuation planning.

---

## 1. Project Overview

During emergencies such as building fires, finding the shortest route to an exit is not necessarily the safest option. A route that is initially accessible may become dangerous as fire spreads, obstacles appear, or environmental risk changes.

Traditional pathfinding algorithms generally focus on finding a path based on distance. However, an emergency evacuation system must also consider environmental hazards and continuously adapt to changing conditions.

This project addresses that problem by implementing a risk-aware pathfinding system using the A* search algorithm.

The simulation represents a building as a two-dimensional grid containing walls, corridors, walkable areas, a starting position, and multiple emergency exits.

As the simulation runs, fire can be introduced and spread across the environment. The system updates the risk values of affected cells, detects unsafe routes, evaluates alternative exits, and calculates a new escape route when necessary.

The entire process is visualized through an interactive Pygame application.

## 2. Project Objectives

The primary objectives of this project are:

- Implement a risk-aware version of the A* search algorithm.
- Represent a building environment using a two-dimensional grid.
- Support multiple emergency exits and evaluate their accessibility.
- Simulate fire propagation in different directions.
- Calculate environmental risk based on the proximity of fire.
- Dynamically detect blocked or unsafe evacuation routes.
- Recalculate escape routes when environmental conditions change.
- Provide an interactive visualization of the evacuation process.
- Evaluate the system through automated tests and simulation scenarios.

## 3. Key Features

### Risk-Aware A* Pathfinding
- Uses A* search to calculate evacuation routes.
- Considers both movement distance and environmental risk.
- Uses a weighted path-cost function to discourage dangerous routes.
- Maintains compatibility with the original pathfinding interface.

### Multiple Emergency Exit Evaluation
- Supports multiple exits within the building.
- Evaluates the reachability and cost of each exit.
- Selects the most suitable reachable exit based on weighted route cost.
- Uses additional tie-breaking criteria when route costs are equal.

### Dynamic Route Replanning
- Monitors the current evacuation route.
- Detects blocked or invalid paths.
- Recalculates routes when environmental conditions change.
- Evaluates alternative emergency exits.
- Avoids unnecessary replanning when the existing route remains suitable.

### Dynamic Fire Simulation
- Allows fire to be introduced into walkable grid cells.
- Simulates fire propagation over time.
- Supports North, South, East, West, and All-direction spread settings.
- Allows the fire direction to be changed during simulation.
- Marks burning cells as blocked for pathfinding.

### Environmental Risk Modelling
- Assigns risk values to cells based on their proximity to fire.
- Updates the environment's risk map as fire spreads.
- Assigns maximum risk to burning cells.
- Recalculates risk when environmental conditions change.

### Real-Time Simulation
- Connects the environment, fire simulation, risk model, and route planner.
- Simulates player movement and fire propagation using separate update intervals.
- Tracks evacuation status and simulation metrics.
- Supports pause, resume, single-step execution, and reset.

### Interactive Pygame Visualization
- Displays the building grid, walls, corridors, exits, and evacuation route.
- Shows the current player position.
- Visualizes fire and environmental risk.
- Displays evacuation status, selected exit, route cost, route length, simulation time, fire steps, and replanning count.
- Supports keyboard and mouse interaction.

## 4. Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| A* Search | Intelligent pathfinding |
| Heapq | Priority queue for A* |
| Pygame-CE | Interactive simulation and visualization |
| NumPy | Numerical operations and supporting calculations |
| Matplotlib | Supporting visualization dependency |
| Pytest | Automated testing |
| Unittest | Unit testing |

## 5. System Architecture

The application is divided into independent modules that work together to perform emergency evacuation planning.

```text
+-----------------------------+
|        main.py              |
|   Application Entry Point   |
+--------------+--------------+
               |
               v
+-----------------------------+
|       simulation.py         |
|    Evacuation Simulation    |
+--------------+--------------+
               |
       +-------+-------+
       |               |
       v               v
+--------------+  +----------------+
| environment  |  | fire_simulation|
|     .py      |  |      .py       |
+------+-------+  +--------+-------+
       |                   |
       |                   v
       |           +---------------+
       |           | risk_model.py |
       |           +-------+-------+
       |                   |
       +---------+---------+
                 |
                 v
       +-------------------+
       |    route_planner  |
       |        .py        |
       +---------+---------+
                 |
                 v
       +-------------------+
       |      astar.py     |
       |  Risk-Aware A*    |
       +---------+---------+
                 |
                 v
       +-------------------+
       |   replanning.py   |
       | Dynamic Replanning|
       +---------+---------+
                 |
                 v
       +-------------------+
       |  visualization.py |
       | Pygame Interface  |
       +-------------------+
```

### Main Modules

| Module | Responsibility |
|---|---|
| `main.py` | Application entry point |
| `config.py` | Simulation and algorithm configuration |
| `environment.py` | Building grid, walls, exits, obstacles, and risk data |
| `astar.py` | Risk-aware A* pathfinding and path-cost calculations |
| `route_planner.py` | Evaluation and selection of multiple emergency exits |
| `replanning.py` | Dynamic route validation and replanning |
| `fire_simulation.py` | Fire ignition, directional spread, and propagation |
| `risk_model.py` | Environmental risk calculation and risk-map updates |
| `simulation.py` | Main evacuation simulation controller |
| `visualization.py` | Pygame rendering and user interaction |
| `metrics.py` | Simulation statistics and performance metrics |

## 6. Algorithms and Mathematical Model

### 6.1 A* Search Algorithm

A* is an informed search algorithm that finds a path between a starting position and a goal by evaluating the cost of reaching a node and the estimated cost of reaching the goal.

The evaluation function is:

\[
f(n)=g(n)+h(n)
\]

Where:

- \(g(n)\) is the accumulated cost from the starting position to node \(n\).
- \(h(n)\) is the estimated cost from node \(n\) to the goal.
- \(f(n)\) is the total estimated path cost.

The implementation uses a priority queue to process the node with the lowest estimated total cost.

### 6.2 Risk-Aware Path Cost

Unlike conventional shortest-path algorithms, this implementation considers environmental risk when calculating movement costs.

The movement cost is:

\[
c(u,v)=w_d+w_r\frac{R(v)}{100}
\]

Where:

- \(c(u,v)\) = cost of moving from cell \(u\) to cell \(v\).
- \(w_d\) = movement distance weight.
- \(w_r\) = environmental risk weight.
- \(R(v)\) = risk value of the destination cell, ranging from 0 to 100.

The total path cost is the sum of the movement costs of all transitions along the path.

This allows the algorithm to prefer a slightly longer route when that route has a lower overall risk-adjusted cost.

### 6.3 Heuristic Function

The implementation uses Manhattan distance as the heuristic:

\[
h(n)=w_d\left(|x_n-x_g|+|y_n-y_g|\right)
\]

Where:

- \((x_n,y_n)\) is the current grid position.
- \((x_g,y_g)\) is the goal position.
- \(w_d\) is the distance weight.

The heuristic estimates the remaining distance to the target while avoiding overestimation under the implemented non-negative risk-cost model.

### 6.4 Multiple Exit Selection

The system evaluates the available emergency exits individually.

For every exit, it calculates:

- Whether the exit is reachable.
- The resulting evacuation path.
- The number of movement steps.
- The accumulated weighted route cost.

The route planner selects the reachable exit with the lowest weighted path cost.

Additional tie-breaking criteria include fewer movement steps and a deterministic exit-position comparison.

### 6.5 Dynamic Replanning

The environment can change while an evacuation is in progress.

For example:
- Fire can spread onto the current route.
- A previously accessible cell can become blocked.
- Risk values can increase.
- An alternative exit can become more suitable.

The replanning controller checks the remaining route and determines whether it is still valid.

When necessary, it calculates new routes from the player's current position and evaluates alternative exits.

The controller also uses a cost-improvement threshold to avoid unnecessary route changes when a new route offers only a small improvement.

Blocked or invalid routes trigger immediate replanning.

### 6.6 Fire Propagation Model

Fire propagation is simulated probabilistically.

The base spread probability is calculated using:

\[
P=P_{base}\times\frac{I}{100}
\]

Where:

- \(P\) = base fire spread probability.
- \(P_{base}\) = configured base probability.
- \(I\) = fire intensity, ranging from 0 to 100.

Directional bias modifies the probability of fire spreading in the selected direction.

The simulation supports five directional settings:

- North
- South
- East
- West
- All directions

A configurable random seed can be used to make simulation results reproducible.

### 6.7 Environmental Risk Model

The risk model calculates the risk of a cell based on its distance from active fire sources.

For a fire source with intensity \(I\), the contribution to another cell is:

\[
R_i=\frac{I}{d+1}
\]

Where \(d\) is the Manhattan distance between the fire source and the evaluated cell.

When multiple fire sources exist, the model uses the maximum contribution:

\[
R(v)=\max_i\left(\frac{I_i}{d_i+1}\right)
\]

Burning cells are assigned a risk value of 100.

The calculated risk values are used by the risk-aware A* algorithm during route planning.

## 7. Installation and Setup

### Prerequisites

- Windows, Linux, or macOS.
- Python installation compatible with the dependencies in `requirements.txt`.
- Git (optional, for cloning the repository).

The project was tested using Python 3.14.2 with Pygame-CE.

### Step 1: Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

Navigate to the project directory:

```bash
cd Risk-Aware-Escape-Route-Prediction-
```

### Step 2: Create a Virtual Environment

```bash
python -m venv .venv
```

### Step 3: Activate the Virtual Environment

For Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

For Windows Command Prompt:

```cmd
.venv\Scripts\activate.bat
```

For Linux/macOS:

```bash
source .venv/bin/activate
```

### Step 4: Install Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Step 5: Run the Application

```bash
python main.py
```

The Pygame window should open and display the building environment with the initial evacuation route.

## 8. Simulation Controls

The application supports interactive keyboard and mouse controls.

| Key / Input | Action |
|---|---|
| SPACE | Start or pause the simulation |
| N | Advance one simulation step while paused |
| R | Reset the simulation |
| I | Enable or disable click-to-ignite mode |
| 1 | Set fire direction to North |
| 2 | Set fire direction to South |
| 3 | Set fire direction to East |
| 4 | Set fire direction to West |
| 5 | Set fire direction to All |
| ESC | Exit the application |
| Mouse click | Ignite a valid walkable cell when ignition mode is enabled |

## 9. Testing and Validation

The project includes automated tests for the main algorithmic and simulation components.

The following test suites are included:

- A* pathfinding tests.
- Route planner tests.
- Dynamic replanning tests.
- Building environment tests.
- Fire simulation tests.
- Risk model tests.
- Evacuation simulation tests.

### Running the Tests

Run the complete test suite:

```bash
python -m pytest -v
```

The implementation's reported verification results include:

- **104 pytest tests passed.**
- **21 passing subtests.**
- **67 passing tests across the existing unittest suites.**
- Successful repository whitespace validation using `git diff --check`.

### Tested Simulation Scenarios

| Scenario | Expected Behaviour |
|---|---|
| Normal evacuation | Player follows the calculated route and reaches an exit |
| Fire blocks current route | The system detects the unsafe route and replans |
| Directional fire spread | Fire propagation responds to the selected direction |
| Alternative exit | The planner selects another reachable emergency exit |
| No reachable exit | The system reports that no safe route is available |
| Reset | Environment, fire, risk, player, route, and metrics return to their initial states |

## 10. Current Limitations

This project is an academic simulation and proof of concept. It does not represent a certified real-world emergency evacuation system.

Current limitations include:

1. **Simplified building representation:** The building is represented as a two-dimensional grid rather than a complete architectural floor plan.

2. **Simplified fire propagation:** Fire spread is probabilistic and does not model real combustion physics, temperature, ventilation, or material properties.

3. **Simplified risk estimation:** Environmental risk is based on Manhattan distance and fire intensity. It does not simulate smoke dynamics, toxic gases, heat transfer, or structural damage.

4. **Grid-based movement:** Movement and pathfinding are restricted by the implemented grid representation.

5. **Single-agent simulation:** The current evacuation simulation focuses on one moving player rather than a large population of evacuees.

6. **No real-world sensor integration:** Fire and environmental changes are generated within the simulation rather than received from physical sensors.

These limitations define the current scope of the project and provide opportunities for future development.

## 11. Future Enhancements

Potential future improvements include:

- Integration with real building floor plans.
- More realistic fire and smoke propagation models.
- Dynamic heat and smoke-based risk estimation.
- Multi-agent evacuation simulation.
- Crowd density and congestion modelling.
- Integration with real-time sensor data.
- Comparison of A* with Dijkstra's algorithm and other pathfinding approaches.
- More advanced visualization and simulation analytics.

## 12. Academic Context

**Project Title:** Risk-Aware Emergency Escape Route Prediction Using A* Search in Dynamic Environments

**Course:** AI & Soft Computing

**Programming Language:** Python

**Core AI Technique:** Risk-Aware A* Search

**Project Type:** Artificial Intelligence and Dynamic Simulation

---

## 13. Disclaimer

This project is developed for educational and academic demonstration purposes. The simulation is not intended to replace professional fire safety systems, certified evacuation planning software, or guidance from emergency response authorities.

---

**Built with Python, A* Search, Pygame-CE, and Artificial Intelligence.**
