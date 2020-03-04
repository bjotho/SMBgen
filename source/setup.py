__author__ = 'marble_xu'

import pygame as pg
import numpy as np
from source import constants as c
from source import tools
import os

dir_path = os.path.dirname(os.path.realpath(__file__))

# Fix random seed for reproducibility
seed = np.random.randint(0, 10_000)
np.random.seed(seed)
print("random seed:", seed)

pg.init()
pg.event.set_allowed([pg.KEYDOWN, pg.KEYUP, pg.QUIT])
pg.display.set_caption(c.ORIGINAL_CAPTION)
SCREEN = pg.display.set_mode(c.SCREEN_SIZE)
SCREEN_RECT = SCREEN.get_rect()

# Reset level_gen.txt file if SAVE_LEVEL is set to False
if not c.SAVE_LEVEL:
    file_path = os.path.join(dir_path, "data", "maps", "level_gen.txt")
    with open(file_path, 'w') as file:
        file.write(str((str(c.SOLID_ID * 2) + str(c.AIR_ID * 11) + "\n") * c.PLATFORM_LENGTH))

GFX = tools.load_all_gfx(os.path.join("resources", "graphics"))
