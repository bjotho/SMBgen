from .. import setup
from .. import constants as c
from . import stuff
from ..states import level_state


def create_solid_tile(group, item, level):
    sprite_x, sprite_y, x, y, type = item['sprite_x'], item['sprite_y'], item['x'], item['y'], item['type']
    group.add(SolidTile(sprite_x, sprite_y, x, y, type))


class SolidTile(stuff.Stuff):
    def __init__(self, sprite_x, sprite_y, x, y, type, group=None, name=c.MAP_STEP):
        frame_rect = [(sprite_x, sprite_y, 16, 16)]
        stuff.Stuff.__init__(self, x, y, setup.GFX['tile_set'], frame_rect, c.SOLID_SIZE_MULTIPLIER)

        self.rest_height = y
        self.state = c.RESTING
        self.y_vel = 0
        self.gravity = 1.2
        self.type = type
        self.group = group
        self.name = name

        level_state.insert_observation(x, y, c.SOLID_ID)

    def update(self):
        pass
