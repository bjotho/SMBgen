from .. import constants as c

state = [[]]


def insert_tile(tile):
    if len(state[-1]) >= c.COL_HEIGHT:
        state.append([])

    state[-1].append(tile)
