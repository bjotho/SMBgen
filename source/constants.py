__author__ = 'marble_xu'

DEBUG = False
DEBUG_START_X = 110
DEBUG_START_Y = 534

SCREEN_HEIGHT = 600
SCREEN_WIDTH = 800
SCREEN_SIZE = (SCREEN_WIDTH, SCREEN_HEIGHT)

ORIGINAL_CAPTION = 'Super Mario Bros.'

# COLORS
#                R    G    B
GRAY         = (100, 100, 100)
NAVYBLUE     = ( 60,  60, 100)
WHITE        = (255, 255, 255)
RED          = (255,   0,   0)
GREEN        = (  0, 255,   0)
FOREST_GREEN = ( 31, 162,  35)
BLUE         = (  0,   0, 255)
SKY_BLUE     = ( 39, 145, 251)
YELLOW       = (255, 255,   0)
ORANGE       = (255, 128,   0)
PURPLE       = (255,   0, 255)
CYAN         = (  0, 255, 255)
BLACK        = (  0,   0,   0)
NEAR_BLACK   = ( 19,  15,  48)
COMBLUE      = (233, 232, 255)
GOLD         = (255, 215,   0)

BGCOLOR = WHITE


SIZE_MULTIPLIER = 2.5
BRICK_SIZE_MULTIPLIER = 2.69
SOLID_SIZE_MULTIPLIER = 2.69
BACKGROUND_MULTIPLER = 2.679
GROUND_HEIGHT = SCREEN_HEIGHT - 62

GAME_TIME_OUT = 301

# STATES FOR ENTIRE GAME
MAIN_MENU = 'main menu'
LOAD_SCREEN = 'load screen'
TIME_OUT = 'time out'
GAME_OVER = 'game over'
LEVEL = 'level'

# MAIN MENU CURSOR STATES
PLAYER1 = '1 PLAYER GAME'
PLAYER2 = '2 PLAYER GAME'

# GAME INFO DICTIONARY KEYS
BASE_FPS = 'base fps'
COIN_TOTAL = 'coin total'
SCORE = 'score'
TOP_SCORE = 'top score'
LIVES = 'lives'
CURRENT_TIME = 'current time'
LEVEL_NUM = 'level num'
PLAYER_NAME = 'player name'
PLAYER_MARIO = 'mario'
PLAYER_LUIGI = 'luigi'

# MAP GENERATION
GENERATE_MAP = True
RANDOM_GEN = False
TRAIN_GEN = True
LOAD_GEN_MODEL = True
INSERT_GROUND = True
ONLY_GROUND = False
READ = True
WRITE = False
SAVE_LEVEL = False
PRINT_GEN_REWARD = False
PRINT_Q_VALUES = False
SNAKING = True
CHUNK_BASED_GREEDY = False  # If False -> Tile based greedy (greedy determined for each individual tile)
GREEDY = False  # Only perform greedy actions
UPDATE_RADIUS = 1 * SCREEN_WIDTH
GEN_DISTANCE = 2 * SCREEN_WIDTH
GEN_HEIGHT = 580
GEN_LENGTH = 5
PLATFORM_LENGTH = 5
Y_OFFSET = 64
COL_HEIGHT = 13
TILE_SIZE = 43
GEN_PX_LEN = GEN_LENGTH * TILE_SIZE
GEN_MODEL_SAVE_INTERVAL = 100  # Number of training sessions
REP_MEM = 'rep_mem'
GEN_BORDER = 'gen_border'
GEN_LINE = 'gen_line'
TIMESTEP = 'timestep'
PLAYER_X = 'player_x'
REWARD = 'reward'
OPTIMAL_V = 'optimal_v'
DONE = 'done'

# GENERATION IDENTIFIERS
AIR_ID = '_'
GROUND_ID = 'g'
BRICK_ID = 'b'
BOX_ID = 'q'
STEP_ID = 't'
SOLID_ID = 's'
PIPE_ID = 'p'
FLAG_ID = 'f'
COIN_ID = 'c'
MUSHROOM_ID = 'm'
FIREFLOWER_ID = 'i'
FIREBALL_ID = 'o'
STAR_ID = 'x'
LIFE_ID = 'u'
GOOMBA_ID = '0'
KOOPA_ID = '1'
FLY_KOOPA_ID = '2'
PIRANHA_ID = '3'
FIRESTICK_ID = '4'
FIRE_KOOPA_ID = '5'
FIRE_ID = 'w'
ENEMY_IDS = [GOOMBA_ID, KOOPA_ID, FLY_KOOPA_ID, PIRANHA_ID, FIRESTICK_ID, FIRE_KOOPA_ID, FIRE_ID]
SOLID_IDS = [GROUND_ID, BRICK_ID, BOX_ID, STEP_ID, SOLID_ID, PIPE_ID]

TILES = [
    STAR_ID,
    LIFE_ID,
    MUSHROOM_ID,
    FIREFLOWER_ID,
    FIREBALL_ID,
    FLAG_ID,
    AIR_ID,
    BOX_ID,
    BRICK_ID,
    SOLID_ID,
    GOOMBA_ID,
    KOOPA_ID,
    FLY_KOOPA_ID,
    FIRE_KOOPA_ID,
    FIRE_ID
]

GENERATOR_TILES = [
    AIR_ID,
    BOX_ID,
    BRICK_ID,
    SOLID_ID,
    GOOMBA_ID,
    KOOPA_ID
]

# FOR TESTING PURPOSES (reduce lag)
# while len(GENERATOR_TILES) < 30:
#     GENERATOR_TILES.append(AIR_ID)

# GENERATION NETWORK PARAMETERS
LEARNING_RATE = 0.001
MEMORY_LENGTH = 64
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01
REPLAY_MEMORY_SIZE = 100
MIN_REPLAY_MEMORY_SIZE = 50
MINIBATCH_SIZE = 10
DISCOUNT = 0.99

# GYM COMPONENTS
ENV_NAME = 'MarioEnv'
ACTION_KEYS = 323
EVALUATE = False
SKIP_MENU = True
HUMAN_PLAYER = False
LOAD_CHECKPOINT = True
PRINT_OBSERVATION = False
PRINT_LEVEL = False
OBS_RADIUS = 10
OBS_SIZE = 2 * OBS_RADIUS + 1
OBS_FRAMES = OBS_SIZE

# MAP COMPONENTS
MAP_IMAGE = 'image_name'
MAP_MAPS = 'maps'
SUB_MAP = 'sub_map'
MAP_GROUND = 'ground'
MAP_PIPE = 'pipe'
PIPE_TYPE_NONE = 0
PIPE_TYPE_IN = 1                # can go down in the pipe
PIPE_TYPE_HORIZONTAL = 2        # can go right in the pipe
MAP_STEP = 'step'
MAP_BRICK = 'brick'
STEP_NUM = 'step_num'
BRICK_NUM = 'brick_num'
TYPE_NONE = 0
TYPE_COIN = 1
TYPE_STAR = 2
MAP_BOX = 'box'
TYPE_MUSHROOM = 3
TYPE_FIREFLOWER = 4
TYPE_FIREBALL = 5
TYPE_LIFEMUSHROOM = 6
MAP_ENEMY = 'enemy'
ENEMY_TYPE_GOOMBA = 0
ENEMY_TYPE_KOOPA = 1
ENEMY_TYPE_FLY_KOOPA = 2
ENEMY_TYPE_PIRANHA = 3
ENEMY_TYPE_FIRESTICK = 4
ENEMY_TYPE_FIRE_KOOPA = 5
ENEMY_RANGE = 'range'
MAP_CHECKPOINT = 'checkpoint'
ENEMY_GROUPID = 'enemy_groupid'
MAP_INDEX = 'map_index'
CHECKPOINT_TYPE_ENEMY = 0
CHECKPOINT_TYPE_FLAG = 1
CHECKPOINT_TYPE_CASTLE = 2
CHECKPOINT_TYPE_MUSHROOM = 3
CHECKPOINT_TYPE_PIPE = 4        # trigger player to go right in a pipe
CHECKPOINT_TYPE_PIPE_UP = 5     # trigger player to another map and go up out of a pipe
CHECKPOINT_TYPE_MAP = 6         # trigger player to go to another map
CHECKPOINT_TYPE_BOSS = 7        # defeat the boss
MAP_FLAGPOLE = 'flagpole'
FLAGPOLE_TYPE_FLAG = 0
FLAGPOLE_TYPE_POLE = 1
FLAGPOLE_TYPE_TOP = 2
MAP_SLIDER = 'slider'
HORIZONTAL = 0
VERTICAL = 1
VELOCITY = 'velocity'
MAP_COIN = 'coin'

# COMPONENT COLOR
COLOR = 'color'
COLOR_TYPE_ORANGE = 0
COLOR_TYPE_GREEN = 1
COLOR_TYPE_RED = 2

# BRICK STATES
RESTING = 'resting'
BUMPED = 'bumped'
OPENED = 'opened'

# MUSHROOM STATES
REVEAL = 'reveal'
SLIDE = 'slide'

# Player FRAMES
PLAYER_FRAMES = 'image_frames'
RIGHT_SMALL_NORMAL = 'right_small_normal'
RIGHT_BIG_NORMAL = 'right_big_normal'
RIGHT_BIG_FIRE = 'right_big_fire'

# PLAYER States
STAND = 'standing'
WALK = 'walk'
JUMP = 'jump'
FALL = 'fall'
FLY = 'fly'
SMALL_TO_BIG = 'small to big'
BIG_TO_FIRE = 'big to fire'
BIG_TO_SMALL = 'big to small'
FLAGPOLE = 'flag pole'
WALK_AUTO = 'walk auto'     # ignoring key input in this state
END_OF_LEVEL_FALL = 'end of level fall'
IN_CASTLE = 'in castle'
DOWN_TO_PIPE = 'down to pipe'
UP_OUT_PIPE = 'up out of pipe'

# PLAYER FORCES
PLAYER_SPEED = 'speed'
WALK_ACCEL = 'walk_accel'
RUN_ACCEL = 'run_accel'
JUMP_VEL = 'jump_velocity'
MAX_Y_VEL = 'max_y_velocity'
MAX_RUN_SPEED = 'max_run_speed'
MAX_WALK_SPEED = 'max_walk_speed'
SMALL_TURNAROUND = .35
JUMP_GRAVITY = .25
GRAVITY = 1.01

# LIST of ENEMIES
GOOMBA = 'goomba'
KOOPA = 'koopa'
FLY_KOOPA = 'fly koopa'
FIRE_KOOPA = 'fire koopa'
FIRE = 'fire'
PIRANHA = 'piranha'
FIRESTICK = 'firestick'

# GOOMBA Stuff
LEFT = 'left'
RIGHT = 'right'
JUMPED_ON = 'jumped on'
DEATH_JUMP = 'death jump'

# KOOPA STUFF
SHELL_SLIDE = 'shell slide'

# FLAG STATE
TOP_OF_POLE = 'top of pole'
SLIDE_DOWN = 'slide down'
BOTTOM_OF_POLE = 'bottom of pole'

# FIREBALL STATE
FLYING = 'flying'
BOUNCING = 'bouncing'
EXPLODING = 'exploding'

# IMAGE SHEET
ENEMY_SHEET = 'smb_enemies_sheet'
ITEM_SHEET = 'item_objects'
