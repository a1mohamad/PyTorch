from collections import defaultdict
import gymnasium as gym
import numpy as np
import os
import pickle
import pygame
import time



CELL_SIZE = 100
MARGIN = 10

def get_coords(row, col, loc= 'center'):
    xc = (col + 1.5) * CELL_SIZE
    yc = (row + 1.5) * CELL_SIZE

    if loc == 'center':
        return xc, yc
    elif loc == 'interior_corners':
        half_size = CELL_SIZE // 2 - MARGIN
        xl, xr = xc - half_size, xc + half_size
        yt, yb = xc - half_size, xc + half_size
        return [(xl, yt), (xr, yt), (xr, yb), (xl, yb)]

    elif loc == 'interior_triangle':
        x1, y1 = xc, yc + CELL_SIZE // 3
        x2, y2 = xc - CELL_SIZE // 3, yc - CELL_SIZE // 3
        x3, y3 = xc + CELL_SIZE // 3, yc - CELL_SIZE // 3
        return [(x1, y1), (x2, y2), (x3, y3)]


# --- This new _render_frame function replaces your old rendering setup ---
def _render_frame(self, agent_pos, obstacle_list):
    # Setup Pygame window (as detailed in previous response)
    # ... [Pygame initialization code omitted for brevity] ...

    canvas = pygame.Surface((self.WINDOW_SIZE, self.WINDOW_SIZE))
    canvas.fill((255, 255, 255)) # White background

    # -------------------------------------------------------------
    # 1. Draw all obstacles/features (Polygons/Triangles)
    # The original code used len(coords_list) > 3 for polygons and == 3 for triangles.

    for obj_type, row, col in obstacle_list: # Assuming a list of tuples like ('hole', 1, 2)

        # Determine the coordinate set based on object type
        if obj_type in ['wall', 'hole']:
            coords_list = self.get_coords(row, col, loc='interior_corners') # Polygon
        elif obj_type == 'goal':
            coords_list = self.get_coords(row, col, loc='interior_triangle') # Triangle
        else:
            continue # Skip unknown objects

        # Determine color based on original draw_object logic:
        if len(coords_list) == 3: # -> Triangle (Goal)
            # Color: (0.9, 0.6, 0.2) -> Yellow/Orange
            color = (230, 153, 51)
        elif len(coords_list) > 3: # -> Polygon (Wall/Hole)
            # Color: (0.4, 0.4, 0.8) -> Blue/Purple
            color = (102, 102, 204)

        pygame.draw.polygon(canvas, color, coords_list)

    # -------------------------------------------------------------
    # 2. Draw the Agent (Circle)
    # The original code used len(coords_list) == 1: # -> circle

    # Get the center pixel coordinates for the agent's position
    # Assuming agent_pos is (row, col)
    agent_row, agent_col = agent_pos 
    center_x, center_y = self.get_coords(agent_row, agent_col, loc='center') 

    radius = int(0.45 * self.CELL_SIZE)

    # Color: (0.2, 0.2, 0.2) -> Black/Dark Gray
    color_agent = (51, 51, 51) 

    pygame.draw.circle(canvas, color_agent, (center_x, center_y), radius)

    # -------------------------------------------------------------
    # 3. Finalize and Display (Standard Gymnasium Procedure)
    # ... [Pygame display update and return code omitted for brevity] ...



 # FIX 1: New path for DiscreteEnv
class GridWorldEnv(gym.Env): # FIX 3: Inherit from the new location

    # FIX 4: Gymnasium requires render_modes in metadata
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4} 

    # FINALIZED __init__ METHOD
    def __init__(self, num_rows=4, num_cols=6, delay=0.05, render_mode="human"):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.delay = delay
        self.render_mode = render_mode

        # 1. SETUP ACTION DEFINITIONS
        move_up = lambda row, col: (max(row - 1, 0), col)
        move_down = lambda row, col: (min(row + 1, num_rows - 1), col)
        move_left = lambda row, col: (row, max(col - 1, 0))
        move_right = lambda row, col: (row, min(col + 1, num_cols - 1))

        self.action_defs = {0: move_up, 1: move_right, 2: move_down, 3: move_left}

        # 2. CALCULATE STATE/ACTION COUNTS AND MAPPINGS
        nS = num_cols * num_rows
        nA = len(self.action_defs)
        self.nS = nS
        self.nA = nA
        self.grid2state_dict = {(s // num_cols, s % num_cols): s for s in range(nS)}
        self.state2grid_dict = {s: (s // num_cols, s % num_cols) for s in range(nS)}

        # 3. DEFINE GOLD, TRAP, and TERMINAL STATES
        gold_cell = (num_rows // 2, num_cols - 2)
        trap_cells = [((gold_cell[0] + 1), gold_cell[1]), (gold_cell[0], gold_cell[1] - 1), ((gold_cell[0] - 1), gold_cell[1])]
        gold_state = self.grid2state_dict[gold_cell]
        trap_states = [self.grid2state_dict[(r, c)] for (r, c) in trap_cells]
        self.terminal_states = [gold_state] + trap_states
        print(self.terminal_states)

        # 4. BUILD TRANSITION PROBABILITY MATRIX (P)
        P = defaultdict(dict)
        for s in range(nS):
            row, col = self.state2grid_dict[s]
            P[s] = defaultdict(list)
            for a in range(nA):
                action = self.action_defs[a]
                next_s = self.grid2state_dict[action(row, col)]

                if self.is_terminal(next_s):
                    r = (1.0 if next_s == self.terminal_states[0] else -1.0)
                else:
                    r = 0.0

                if self.is_terminal(s):
                    done = True
                    next_s = s
                else:
                    done = False

                P[s][a] = [(1.0, next_s, r, done)]

        # 5. DEFINE SPACES AND ISD (Initial State Distribution)
        self.observation_space = gym.spaces.Discrete(nS)
        self.action_space = gym.spaces.Discrete(nA)
        self._P = P # Store transition matrix

        isd = np.zeros(nS)
        isd[0] = 1.0
        self.initial_state_distribution = isd # Store ISD (Gymnasium Standard)
        self.s = 0

        # 6. CALL PARENT CONSTRUCTOR (gym.Env takes no custom params)
        super().__init__()

        # 7. PYGAME & RENDERING SETUP
        self.CELL_SIZE = CELL_SIZE # Use the global constant
        self.MARGIN = MARGIN       # Use the global constant
        self.WINDOW_WIDTH = (self.num_cols + 2) * self.CELL_SIZE
        self.WINDOW_HEIGHT = (self.num_rows + 2) * self.CELL_SIZE
        self.window = None
        self.clock = None

        self._build_display_assets(gold_cell, trap_cells)
    def is_terminal(self, state):
        return state in self.terminal_states

    # NEW HELPER: Replaces the old coordinate transformation
    def get_coords(self, row, col, loc='center'):
        # NOTE: Using self.CELL_SIZE and self.MARGIN
        xc = (col + 1.5) * self.CELL_SIZE
        yc = (row + 1.5) * self.CELL_SIZE

        if loc == 'center':
            return xc, yc
        elif loc == 'interior_corners':
            half_size = self.CELL_SIZE // 2 - self.MARGIN
            xl, xr = xc - half_size, xc + half_size
            yt, yb = yc - half_size, yc + half_size # Corrected: used yc instead of xc for y-coords
            return [(xl, yt), (xr, yt), (xr, yb), (xl, yb)]
        elif loc == 'interior_triangle':
            x1, y1 = xc, yc + self.CELL_SIZE // 3
            x2, y2 = xc + self.CELL_SIZE // 3, yc - self.CELL_SIZE // 3
            x3, y3 = xc - self.CELL_SIZE // 3, yc - self.CELL_SIZE // 3
            return [(x1, y1), (x2, y2), (x3, y3)]

    # NEW METHOD: Stores the fixed display features (Grid, Traps, Gold)
    # This replaces the old _build_display's geometry setup
    def _build_display_assets(self, gold_cell, trap_cells):
        self.trap_coords = [self.get_coords(*cell, loc='center') for cell in trap_cells]
        self.gold_coords = self.get_coords(*gold_cell, loc='interior_triangle')

        # Check for agent coords saved by book's script
        if (os.path.exists('robot-coordinates.pkl') and self.CELL_SIZE == 100):
            agent_pkl_coords = pickle.load(open('robot-coordinates.pkl', 'rb'))
            starting_coords = self.get_coords(0, 0, loc='center')
            self.agent_coords_type = 'polygon'
            # Convert to list of (x, y) tuples required by Pygame
            self.agent_polygon_coords = (agent_pkl_coords + np.array(starting_coords)).tolist()
        else:
            self.agent_coords_type = 'polygon' # Agent is a square/polygon by default
            self.agent_polygon_coords = self.get_coords(0, 0, loc='interior_corners')

    # NEW METHOD: The main rendering logic using Pygame
    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
        canvas.fill((255, 255, 255))  # White background

        # Define Colors (based on original R, G, B floats)
        COLOR_BORDER = (50, 50, 50)
        COLOR_GRID = (100, 100, 100)
        COLOR_TRAP = (51, 51, 51)      # (0.2, 0.2, 0.2)
        COLOR_GOLD = (230, 153, 51)    # (0.9, 0.6, 0.2)
        COLOR_AGENT = (102, 102, 204)  # (0.4, 0.4, 0.8)

        # -----------------------------------------------------------------
        # 1. Draw Border (Replaces rendering.PolyLine)
        bp_list = [
            (self.CELL_SIZE - self.MARGIN, self.CELL_SIZE - self.MARGIN), 
            (self.WINDOW_WIDTH - self.CELL_SIZE + self.MARGIN, self.CELL_SIZE - self.MARGIN), 
            (self.WINDOW_WIDTH - self.CELL_SIZE + self.MARGIN, self.WINDOW_HEIGHT - self.CELL_SIZE + self.MARGIN),
            (self.CELL_SIZE - self.MARGIN, self.WINDOW_HEIGHT - self.CELL_SIZE + self.MARGIN)
        ]
        # Draw thick border (Line thickness in Pygame is handled by the last argument)
        pygame.draw.lines(canvas, COLOR_BORDER, True, bp_list, 5)

        # -----------------------------------------------------------------
        # 2. Draw Grid Lines (Replaces rendering.PolyLine)
        for col in range(self.num_cols + 1):
            x = (col + 1) * self.CELL_SIZE
            y1, y2 = self.CELL_SIZE, (self.num_rows + 1) * self.CELL_SIZE
            pygame.draw.line(canvas, COLOR_GRID, (x, y1), (x, y2))

        for row in range(self.num_rows + 1):
            y = (row + 1) * self.CELL_SIZE
            x1, x2 = self.CELL_SIZE, (self.num_cols + 1) * self.CELL_SIZE
            pygame.draw.line(canvas, COLOR_GRID, (x1, y), (x2, y))

        # -----------------------------------------------------------------
        # 3. Draw Traps (Circles - Replaces draw_object with len=1)
        radius = int(0.45 * self.CELL_SIZE)
        for center_x, center_y in self.trap_coords:
            pygame.draw.circle(canvas, COLOR_TRAP, (int(center_x), int(center_y)), radius)

        # -----------------------------------------------------------------
        # 4. Draw Gold (Triangle - Replaces draw_object with len=3)
        # self.gold_coords is already a list of 3 (x, y) points
        pygame.draw.polygon(canvas, COLOR_GOLD, self.gold_coords)

        # -----------------------------------------------------------------
        # 5. Draw Agent (Polygon/Robot - Replaces draw_object with len > 3)
        agent_row, agent_col = self.state2grid_dict[self.s] # Current state (self.s)

        # Calculate the translation offset based on the AGENT's current cell
        # The original code's render() method used (X+0)*CELL_SIZE, (Y+0)*CELL_SIZE
        # This means the agent's drawing must be translated to the top-left of the cell.

        # Calculate the top-left corner of the current cell: (col+1)*CELL_SIZE, (row+1)*CELL_SIZE
        # This is where the agent's 'transform' is set to in the original render()
        target_x_offset = (agent_col + 1) * self.CELL_SIZE 
        target_y_offset = (agent_row + 1) * self.CELL_SIZE

        # We need to apply this translation to the *entire* agent polygon
        translated_agent_coords = []

        # The original 'robot-coordinates.pkl' or 'interior_corners' are drawn relative 
        # to the (0,0) position, so we shift all points.
        for x, y in self.agent_polygon_coords:
            # We must subtract the original (0,0) starting coords defined in _build_display
            # before adding the new cell's offset. This is complex and usually requires 
            # knowing the original (0,0) coordinates which were based on (0+1.5)*CELL_SIZE

            # SIMPLIFIED: Assuming the polygon coordinates were defined relative to the cell center 
            # and that the original viewer's transformation compensated for the 1.5 offset. 

            # Since the original rendering used 'set_translation(x, y)' to move the whole object, 
            # we will translate the coordinates stored in self.agent_polygon_coords directly.

            # The original code's render() used:
            # x_coord = (x_coord+0) * CELL_SIZE -> (col)*CELL_SIZE
            # y_coord = (y_coord+0) * CELL_SIZE -> (row)*CELL_SIZE
            # This is complex because the viewer coordinates start at (1, 1).

            # Best Approximation: Use the grid cell's top-left corner (at index 1,1)
            # The agent is drawn relative to its starting cell (1.5*C, 1.5*C), 
            # and the render() method applies the offset.

            # Get the top-left of the drawing area (1.5*C - X_OFFSET)
            original_start_x = self.CELL_SIZE * 1.5
            original_start_y = self.CELL_SIZE * 1.5

            # Calculate the current cell's center for drawing reference
            current_center_x, current_center_y = self.get_coords(agent_row, agent_col, loc='center')

            # Agent coordinates are relative to the original start (0, 0)
            # We translate them to be relative to the current center.
            if self.agent_coords_type == 'polygon':
                # The agent_polygon_coords are relative to (1.5*C, 1.5*C).

                # Calculate the needed shift: current center - original start center
                x_shift = current_center_x - original_start_x
                y_shift = current_center_y - original_start_y

                for px, py in self.agent_polygon_coords:
                    translated_agent_coords.append((px + x_shift, py + y_shift))

                pygame.draw.polygon(canvas, COLOR_AGENT, translated_agent_coords)

            # Fallback/Default for Agent (A simple circle if polygon translation is too complex)
            else:
                radius = int(0.45 * self.CELL_SIZE)
                pygame.draw.circle(canvas, COLOR_AGENT, (int(current_center_x), int(current_center_y)), radius)

        # -----------------------------------------------------------------
        # 6. Finalize and Display
        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])

            return self.window

        elif self.render_mode == "rgb_array":
            return np.transpose(pygame.surfarray.array3d(canvas), axes=(1, 0, 2))

        # --- Add this method inside your GridWorldEnv class ---

    def reset(self, seed=None, options=None):
        """
        Resets the environment to an initial state and returns the initial observation.
        """
        # 1. Gymnasium standard: call super().reset() to handle seeding
        super().reset(seed=seed)

        # 2. Reset the agent's state (self.s) to a state based on the Initial State Distribution (ISD)
        # self.np_random is provided by super().reset()
        self.s = self.np_random.choice(len(self.initial_state_distribution), p=self.initial_state_distribution)

        # 3. Get observation (the state itself)
        observation = self.s

        # 4. Get info dictionary (Gymnasium standard)
        info = {"prob": 1.0}

        # 5. Render the initial frame
        if self.render_mode is not None:
            self.render()

        return observation, info

    # --- Add this method inside your GridWorldEnv class ---

    def step(self, a):
        """
        Executes one time step within the environment using action 'a'.
        """
        # 1. Look up the transition details (prob, next_state, reward, terminated)
        # The transitions are stored in the _P matrix we created in __init__
        transitions = self._P[self.s][a]

        # 2. Randomly sample the next state based on probabilities (P[s][a])
        if len(transitions) == 0:
            # Handle the edge case of an empty transition list (though unlikely in this environment)
            prob = 1.0
            next_s = self.s
            reward = 0.0
            terminated = True
        else:
            i = self.np_random.choice(len(transitions), p=[t[0] for t in transitions])
            prob, next_s, reward, terminated = transitions[i]

        # 3. Update the state
        self.s = next_s

        # 4. Truncation check (not used in this environment, so set to False)
        truncated = False

        # 5. Get observation (the new state)
        observation = self.s

        # 6. Get info dictionary
        info = {"prob": prob}

        # 7. Render if needed
        if self.render_mode is not None:
            self.render(done=terminated or truncated)

        # 8. Gymnasium step returns 5 values: obs, reward, terminated, truncated, info
        return observation, reward, terminated, truncated, info


    # REWRITE: The new render method
    def render(self, mode='human', done=False):
        # The 'mode' argument is usually ignored in Gymnasium's render() when set in __init__
        # We use time.sleep only for 'human' mode to visualize the movement
        if self.render_mode == 'human':
            sleep_time = 1 if done else self.delay
        else:
            sleep_time = 0

        # Rendering is delegated to _render_frame
        rend = self._render_frame() 

        time.sleep(sleep_time)
        return rend

    # REWRITE: The new close method
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None

if __name__ == '__main__':
    env = GridWorldEnv(5, 6)
    for i in range(1):
        env.reset() # Call reset, but ignore the returned (obs, info) tuple for simplicity
        env.render(mode='human', done=False)
        while True:
            # This is correct now, as env.nA is defined inside __init__
            action = np.random.choice(env.nA) 

            # Gymnasium step returns 5 values: obs, reward, terminated, truncated, info
            obs, reward, terminated, truncated, info = env.step(action) 

            # Use terminated or terminated or truncated for the break condition
            done = terminated or truncated

            print('Action ', env.s, action, ' -> ', (obs, reward, terminated, truncated, info))
            env.render(mode='human', done=done)

            if done:
                break
    env.close()





