from source import setup
from source import constants as c
from source.components import stuff
from source.states import level_state


def create_solid_tile(group, item, level):
    sprite_x, sprite_y, x, y, type = item['sprite_x'], item['sprite_y'], item['x'], item['y'], item['type']
    q = None if 'q' not in item else item['q']
    group.add(SolidTile(sprite_x, sprite_y, x, y, q, type, level=level))


class SolidTile(stuff.Stuff):
    def __init__(self, sprite_x, sprite_y, x, y, q, type, group=None, name=c.MAP_STEP, level=None):
        frame_rect = [(sprite_x, sprite_y, 16, 16)]
        textsurface = None
        if q is not None and level is not None:
            if level.gen_line > 5:
                textsurface = level.q_font.render(q, True, (255, 255, 255))
        stuff.Stuff.__init__(self, x, y, setup.GFX['tile_set'], frame_rect, c.SOLID_SIZE_MULTIPLIER, textsurface)

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
