from .. import setup
from .. import constants as c
from . import stuff


def create_solid_tile(step_group, item, level):
    sprite_x, sprite_y, x, y, type = item['sprite_x'], item['sprite_y'], item['x'], item['y'], item['type']
    step_group.add(Solid_tile(sprite_x, sprite_y, x, y, type))

class Solid_tile(stuff.Stuff):
    def __init__(self, sprite_x, sprite_y, x, y, type, group=None, name=c.MAP_STEP):
        frame_rect = [(sprite_x, sprite_y, 16, 16)]
        stuff.Stuff.__init__(self, x, y, setup.GFX['tile_set'], frame_rect, c.SOLID_TILE_SIZE_MULTIPLIER)

        self.rest_height = y
        self.state = c.RESTING
        self.y_vel = 0
        self.gravity = 1.2
        self.type = type
        self.group = group
        self.name = name

    def update(self):
        pass
