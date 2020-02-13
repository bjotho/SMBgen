from .. import constants as c

state = [[c.AIR_ID for _ in range(c.COL_HEIGHT)]]


def insert_observation(x_px, y_px, id):
    x = int(x_px // c.TILE_SIZE)
    y = int(c.COL_HEIGHT - ((y_px - c.Y_OFFSET) // c.TILE_SIZE) - 1)
    if y >= c.COL_HEIGHT or y < 0:
        return

    while x >= len(state):
        state.append([c.AIR_ID for _ in range(c.COL_HEIGHT)])

    state[x][y] = id


def update_observation(old_x, old_y, new_x, new_y, id, replacement=c.AIR_ID):
    if old_x == new_x and old_y == new_y:
        return replacement

    if old_y >= c.COL_HEIGHT:
        old_y = c.COL_HEIGHT - 1
    if new_y >= c.COL_HEIGHT:
        new_y = c.COL_HEIGHT - 1

    while old_x >= len(state) or new_x >= len(state):
        state.append([c.AIR_ID for _ in range(c.COL_HEIGHT)])

    placeholder = state[new_x][new_y]
    state[old_x][old_y] = replacement
    state[new_x][new_y] = id

    # print_2d(state)

    return placeholder


def delete_observation(x, y, replacement=c.AIR_ID):
    state[x][y] = replacement


def print_2d(_list):
    output = "[\n"
    for col in _list:
        output += "  [ "
        for tile in col:
            output += str(tile) + ", "

        output = output[:-2] + " ]\n"

    output += "]"
    print(output)
