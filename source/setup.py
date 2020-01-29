__author__ = 'marble_xu'

import os
import pygame as pg
import numpy as np
from . import constants as c
from . import tools

# Fix random seed for reproducibility
seed = np.random.randint(0,10000)
np.random.seed(seed)
print("random seed:", seed)

pg.init()
pg.event.set_allowed([pg.KEYDOWN, pg.KEYUP, pg.QUIT])
pg.display.set_caption(c.ORIGINAL_CAPTION)
SCREEN = pg.display.set_mode(c.SCREEN_SIZE)
SCREEN_RECT = SCREEN.get_rect()

# Clear level_gen.txt file if SAVE_LEVEL is set to False
if not c.SAVE_LEVEL:
    map_gen_file = 'level_gen.txt'
    file_path = os.path.join('source', 'data', 'maps', map_gen_file)
    open(file_path, 'w').close()

GFX = tools.load_all_gfx(os.path.join("resources","graphics"))