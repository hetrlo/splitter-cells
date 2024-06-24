import numpy as np
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.ticker import MultipleLocator


def line_intersect(p1, p2, P3, P4):
    """ Calculates intersection point of two segments. segment 1: [p1,p2], segment 2: [P3,P4].
    Inputs:
    p1, p2: arrays containing the coordinates of the points of first segments.
    P3, P4: arrays containing the coordinates of the points of second segments
    Outputs: list of X,Y coordinates of the intersection points. Set to np.inf if no intersection.
    """
    p1 = np.atleast_2d(p1)
    p2 = np.atleast_2d(p2)
    P3 = np.atleast_2d(P3)
    P4 = np.atleast_2d(P4)

    x1, y1 = p1[:, 0], p1[:, 1]
    x2, y2 = p2[:, 0], p2[:, 1]
    X3, Y3 = P3[:, 0], P3[:, 1]
    X4, Y4 = P4[:, 0], P4[:, 1]

    D = (Y4 - Y3) * (x2 - x1) - (X4 - X3) * (y2 - y1)

    # Colinearity test
    C = (D != 0)

    # Calculate the distance to the intersection point
    UA = ((X4 - X3) * (y1 - Y3) - (Y4 - Y3) * (x1 - X3))
    UA = np.divide(UA, D, where=C)
    UB = ((x2 - x1) * (y1 - Y3) - (y2 - y1) * (x1 - X3))
    UB = np.divide(UB, D, where=C)

    # Test if intersections are inside each segment
    C = C * (UA > 0) * (UA < 1) * (UB > 0) * (UB < 1)

    # intersection of the point of the two lines
    X = np.where(C, x1 + UA * (x2 - x1), np.inf)
    Y = np.where(C, y1 + UA * (y2 - y1), np.inf)
    return np.stack([X, Y], axis=1)


class Maze:
    """
    A simple 8-maze made of straight walls (line segments)
    """

    def __init__(self, simulation_mode = "esn"):
        self.name = 'maze'
        self.walls = np.array([
            # Surrounding walls
            [(0, 0), (0, 500)],
            [(0, 500), (300, 500)],
            [(300, 500), (300, 0)],
            [(300, 0), (0, 0)],
            # Bottom hole
            [(100, 100), (200, 100)],
            [(200, 100), (200, 200)],
            [(200, 200), (100, 200)],
            [(100, 200), (100, 100)],
            # Top hole
            [(100, 300), (200, 300)],
            [(200, 300), (200, 400)],
            [(200, 400), (100, 400)],
            [(100, 400), (100, 300)],
            # Moving walls (invisibles) to constraining bot path
            [(0, 250), (100, 200)],
            [(200, 300), (300, 250)]
        ])

        if simulation_mode == "walls":
            self.invisible_walls = True
        else:
            self.invisible_walls = False
            self.walls[12:] = [[(0, 0), (0, 0)],
                               [(0, 0), (0, 0)]]

        self.alternate = None
        self.iter = 0
        self.in_corridor = False

    def draw(self, ax, grid=True, margin=5):
        """
        Render the maze
        """

        # Building a filled patch from walls
        V, C, S = [], [], self.walls
        V.extend(S[0 + i, 0] for i in [0, 1, 2, 3, 0])
        V.extend(S[4 + i, 0] for i in [0, 1, 2, 3, 0])
        V.extend(S[8 + i, 0] for i in [0, 1, 2, 3, 0])
        C = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY] * 3
        path = Path(V, C)
        patch = PathPatch(path, clip_on=False, linewidth=1.5,
                          edgecolor="black", facecolor="white")

        # Set figure limits, grid and ticks
        ax.set_axisbelow(True)
        ax.add_artist(patch)
        ax.set_xlim(0 - margin, 300 + margin)
        ax.set_ylim(0 - margin, 500 + margin)
        if grid:
            ax.xaxis.set_major_locator(MultipleLocator(100))
            ax.xaxis.set_minor_locator(MultipleLocator(10))
            ax.yaxis.set_major_locator(MultipleLocator(100))
            ax.yaxis.set_minor_locator(MultipleLocator(10))
            ax.grid(True, "major", color="0.75", linewidth=1.00, clip_on=False)
            ax.grid(True, "minor", color="0.75", linewidth=0.50, clip_on=False)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(axis='both', which='major', size=0)
        ax.tick_params(axis='both', which='minor', size=0)

    def update_walls(self, bot_position):
        """ Add the invisible walls to force the bot alternating right and left direction."""
        if bot_position[1] < 100:
            self.walls[12:] = [[(0, 250), (100, 300)],
                               [(200, 200), (300, 250)]]
        elif bot_position[1] > 400:
            self.walls[12:] = [[(0, 250), (100, 200)],
                               [(200, 300), (300, 250)]]
        else:
            pass

    def update_walls_RR_LL(self, bot_position):
        """ Add the invisible walls to force the bot alternating right and left direction every other time."""
        if 200 < bot_position[1] < 300:
            if not self.in_corridor:
                if self.iter == 1:
                    self.iter = 0
                else:
                    self.iter += 1
            self.in_corridor = True
        else:
            self.in_corridor = False

        if bot_position[1] < 100 and self.iter < 1:
            self.walls[12:] = [[(0, 250), (100, 300)],
                               [(200, 300), (300, 250)]]

        elif bot_position[1] < 100 and self.iter == 1:
                self.walls[12:] = [[(0, 250), (100, 300)],
                                   [(200, 100), (300, 250)]]

        elif bot_position[1] > 400 and self.iter < 1:
            self.walls[12:] = [[(0, 250), (100, 200)],
                               [(200, 200), (300, 250)]]

        elif bot_position[1] > 400 and self.iter == 1:
            self.walls[12:] = [[(0, 250), (100, 200)],
                               [(200, 300), (300, 250)]]
        else:
            pass

class MazeFour:
    """ A maze with 4 walls along the y axis
    """
    def __init__(self, simulation_mode = "esn"):
        self.name = 'maze_four'
        self.walls = np.array([
            # Surrounding walls
            [(0, 0), (0, 500)],
            [(0, 500), (300, 500)],
            [(300, 500), (300, 0)],
            [(300, 0), (0, 0)],
            # 1st hole
            [(100, 100), (200, 100)],
            [(200, 100), (200, 150)],
            [(200, 150), (100, 150)],
            [(100, 150), (100, 100)],
            # 2nd hole
            [(100, 200), (200, 200)],
            [(200, 200), (200, 250)],
            [(200, 250), (100, 250)],
            [(100, 250), (100, 200)],
            # 3rd hole
            [(100, 300), (200, 300)],
            [(200, 300), (200, 350)],
            [(200, 350), (100, 350)],
            [(100, 350), (100, 300)],
            # 4th hole
            [(100, 400), (200, 400)],
            [(200, 400), (200, 450)],
            [(200, 450), (100, 450)],
            [(100, 450), (100, 400)],

        # Moving walls (invisible) to constraining bot path : from index 20 to end
            [(0, 250), (100, 200)],
            [(200, 300), (300, 250)]
        ])
        
        if simulation_mode == "walls":
            self.invisible_walls = True
        else:
            self.invisible_walls = False
            for k in range(20, len(self.walls)):
                self.walls[k] = [(0, 0), (0, 0)]

        self.alternate = None
        self.iter = 0 # Sert pour le RR-LL
        self.in_corridor = False

    def draw(self, ax, grid=True, margin=5):
        """
        Render the maze
        """

        # Building a filled patch from walls
        V, C, S = [], [], self.walls[:20] # Visible walls
        n_walls = len(S)//4 # For now is ok, but careful
        for k in range(n_walls):
            V.extend(S[4*k + i, 0] for i in [0, 1, 2, 3, 0])

        C = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY] * n_walls
        path = Path(V, C)
        patch = PathPatch(path, clip_on=False, linewidth=1.5,
                          edgecolor="black", facecolor="white")
        

        # Set figure limits, grid and ticks
        ax.set_axisbelow(True)
        ax.add_artist(patch)
        ax.set_xlim(0 - margin, 300 + margin)
        ax.set_ylim(0 - margin, 500 + margin)
        if grid:
            ax.xaxis.set_major_locator(MultipleLocator(100))
            ax.xaxis.set_minor_locator(MultipleLocator(10))
            ax.yaxis.set_major_locator(MultipleLocator(100))
            ax.yaxis.set_minor_locator(MultipleLocator(10))
            ax.grid(True, "major", color="0.75", linewidth=1.00, clip_on=False)
            ax.grid(True, "minor", color="0.75", linewidth=0.50, clip_on=False)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(axis='both', which='major', size=0)
        ax.tick_params(axis='both', which='minor', size=0)


    def update_walls(self, bot_position): # Inherited from Maze : task R-L
        """ Add the invisible walls to force the bot alternating right and left direction."""
        if bot_position[1] < 250:
            self.walls[20:] = [[(0, 275), (100, 250)],
                               [(200, 300), (300, 275)]]
        elif bot_position[1] > 300:
            self.walls[20:] = [[(0, 275), (100, 300)],
                               [(200, 250), (300, 275)]]
        else:
            pass

    def update_walls_4_loop(self, bot_position):# TODO: finish this function (ou pas, peut être peu)
        """ Add the invisible walls to alternate right and left between the four holes."""
        if 250 < bot_position[1] < 300: # Center corridor
            if bot_position[0] > 150: # Right side
                self.walls[20:] = [[(300, 275), (200, 325)],
                               [(300, 175), (200, 125)]]
            
                

        elif bot_position[1] > 450:
            self.walls[20:] = [[(0, 275), (100, 225)],
                               [(200, 325), (300, 275)]]

    def update_walls_RR_LL(self, bot_position):
        """ Add the invisible walls to force the bot alternating right and left direction every other time."""
        if 250 < bot_position[1] < 300:
            if 125 < bot_position[0] < 175:
                if not self.in_corridor:
                    if self.iter == 1:
                        self.iter = 0
                    else:
                        self.iter += 1
                self.in_corridor = True
        else:
            self.in_corridor = False

        if bot_position[1] < 250 and self.iter < 1:
            self.walls[20:] = [[(0, 275), (100, 325)],
                               [(200, 325), (300, 275)]]

        elif bot_position[1] < 259 and self.iter == 1:
                self.walls[20:] = [[(0, 275), (100, 325)],
                                   [(200, 225), (300, 275)]]

        elif bot_position[1] > 300 and self.iter < 1:
            self.walls[20:] = [[(0, 275), (100, 225)],
                               [(200, 225), (300, 275)]]

        elif bot_position[1] > 300 and self.iter == 1:
            self.walls[20:] = [[(0, 275), (100, 225)],
                               [(200, 325), (300, 275)]]
        else:
            pass


# Function to create a wall, given the corners
def create_wall(bl, br, tr, tl): # bottom-left, bottom-right...
    return [
        [bl,br],
        [br,tr],
        [tr,tl],
        [tl,bl]
    ]
class RandomWalls:
    """ A maze with random objects
    """

    def __init__(self, simulation_mode = "esn"):
        self.name = 'random_walls'

        # Surrounding walls
        self.walls = create_wall((0,0), (300,0), (300,500), (0,500))
        # 1st hole
        self.walls += create_wall((0,30), (50,30), (50,160), (0,160))
        # 2nd hole
        self.walls += create_wall((200,300), (250,300), (250,350), (200,350))
        # 3rd hole
        self.walls += create_wall((0, 275), (100, 275), (100, 325), (0, 325))
        # 4th hole
        self.walls += create_wall((200, 450), (300, 450), (300, 500), (200, 500))
        # Corner 1st piece
        self.walls += create_wall((150, 50), (250, 50), (250, 100), (150, 100))
        # Corner 2nd piece
        self.walls += create_wall((200, 100), (250, 100), (250, 150), (200, 150))
        self.walls = np.array(self.walls)
        
        if simulation_mode == "walls":
            self.invisible_walls = True
        else:
            self.invisible_walls = False
            #for k in range(28, len(self.walls)):
            #    self.walls[k] = [(0, 0), (0, 0)]

        self.alternate = None
        self.iter = 0 # Sert pour le RR-LL
        self.in_corridor = False

    def draw(self, ax, grid=True, margin=5):
        """
        Render the maze
        """

        # Building a filled patch from walls
        V, C, S = [], [], self.walls # Visible walls
        n_walls = len(S)//4 # For now is ok, but careful
        for k in range(n_walls):
            V.extend(S[4*k + i, 0] for i in [0, 1, 2, 3, 0])

        C = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY] * n_walls
        path = Path(V, C)
        patch = PathPatch(path, clip_on=False, linewidth=1.5,
                          edgecolor="black", facecolor="white")
        

        # Set figure limits, grid and ticks
        ax.set_axisbelow(True)
        ax.add_artist(patch)
        ax.set_xlim(0 - margin, 300 + margin)
        ax.set_ylim(0 - margin, 500 + margin)
        if grid:
            ax.xaxis.set_major_locator(MultipleLocator(100))
            ax.xaxis.set_minor_locator(MultipleLocator(10))
            ax.yaxis.set_major_locator(MultipleLocator(100))
            ax.yaxis.set_minor_locator(MultipleLocator(10))
            ax.grid(True, "major", color="0.75", linewidth=1.00, clip_on=False)
            ax.grid(True, "minor", color="0.75", linewidth=0.50, clip_on=False)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(axis='both', which='major', size=0)
        ax.tick_params(axis='both', which='minor', size=0)

class MazeOther:
    """ A maze with random objects to generate a pseudo random trajectory
    """

    def __init__(self, simulation_mode = "esn"):
        self.name = 'maze_other'

        # Surrounding walls
        self.walls = create_wall((0,0), (300,0), (300,500), (0,500))
        # 1st hole
        self.walls += create_wall((60,70), (90,70), (90,130), (60,130))
        # 2nd hole
        self.walls += create_wall((160,150), (250,150), (250,175), (160,175))
        # 2nd hole bottom piece
        self.walls += create_wall((215,75), (250,75), (250,150), (215,150))
        # 2nd hole top piece
        self.walls += create_wall((120,175), (185,175), (185,200), (120,200))
        # 2nd hole botbot piece
        self.walls += create_wall((145,50), (250,50), (250,75), (145,75))
        # 3rd hole
        self.walls += create_wall((220,285), (245,285), (245,360), (220,360))
        # 3rd hole bottom piece
        self.walls += create_wall((170,250), (300,250), (300,285), (170,285))
        # 3rd hole top piece
        self.walls += create_wall((245,340), (300,340), (300,360), (245,360))
        # 4th hole
        self.walls += create_wall((70,340), (100,340), (100,440), (70,440))
        # 4th hole middle piece
        self.walls += create_wall((100,320), (130,320), (130,370), (100,370))
        # 5th hole
        self.walls += create_wall((40,220), (65,220), (65,265), (40,265))
        # 6th hole
        self.walls += create_wall((130,480), (170,480), (170,500), (130,500))
        
        self.walls = np.array(self.walls)
        
        if simulation_mode == "walls":
            self.invisible_walls = True
        else:
            self.invisible_walls = False
            #for k in range(28, len(self.walls)):
            #    self.walls[k] = [(0, 0), (0, 0)]

        self.alternate = None
        self.iter = 0 # Sert pour le RR-LL
        self.in_corridor = False

    def draw(self, ax, grid=True, margin=5):
        """
        Render the maze
        """

        # Building a filled patch from walls
        V, C, S = [], [], self.walls # Visible walls
        n_walls = len(S)//4 # For now is ok, but careful
        for k in range(n_walls):
            V.extend(S[4*k + i, 0] for i in [0, 1, 2, 3, 0])

        C = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY] * n_walls
        path = Path(V, C)
        patch = PathPatch(path, clip_on=False, linewidth=1.5,
                          edgecolor="black", facecolor="white")
        

        # Set figure limits, grid and ticks
        ax.set_axisbelow(True)
        ax.add_artist(patch)
        ax.set_xlim(0 - margin, 300 + margin)
        ax.set_ylim(0 - margin, 500 + margin)
        if grid:
            ax.xaxis.set_major_locator(MultipleLocator(100))
            ax.xaxis.set_minor_locator(MultipleLocator(10))
            ax.yaxis.set_major_locator(MultipleLocator(100))
            ax.yaxis.set_minor_locator(MultipleLocator(10))
            ax.grid(True, "major", color="0.75", linewidth=1.00, clip_on=False)
            ax.grid(True, "minor", color="0.75", linewidth=0.50, clip_on=False)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(axis='both', which='major', size=0)
        ax.tick_params(axis='both', which='minor', size=0)