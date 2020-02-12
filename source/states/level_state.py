from .. import constants as c

state = [[c.AIR_ID for _ in range(c.COL_HEIGHT)]]


def insert_observation(x_px, y_px, id):
    x = int(x_px // c.TILE_SIZE)
    y = int(c.COL_HEIGHT - ((y_px - c.Y_OFFSET) // c.TILE_SIZE) - 1)
    if y >= c.COL_HEIGHT or y < 0:
        return

    while x >= len(state):
        state.append([c.AIR_ID for _ in range(c.COL_HEIGHT)])

    # print(id)
    # print("x, y: (", x, y, ")")
    state[x][y] = id


def update_observation(old_x, old_y, new_x, new_y, id, placeholder=None):
    if old_x == new_x and old_y == new_y:
        return

    if old_y >= c.COL_HEIGHT:
        old_y = c.COL_HEIGHT - 1
    if new_y >= c.COL_HEIGHT:
        new_y = c.COL_HEIGHT - 1

    while old_x >= len(state) or new_x >= len(state):
        state.append([c.AIR_ID for _ in range(c.COL_HEIGHT)])

    # if placeholder not in [None, c.AIR_ID, c.GOOMBA_ID, c.KOOPA_ID, c.FLY_KOOPA_ID]:
    # replacement = state[new_x][new_y]
    state[old_x][old_y] = c.AIR_ID
    state[new_x][new_y] = id

    # print_state()

    # return replacement


def delete_observation(x, y):
    # print("deleting (", x, y, ")")
    # print("state[" + str(x) + "][" + str(y) + "]:", state[x][y])
    # print_state()
    state[x][y] = c.AIR_ID


def print_state():
    output = "[\n"
    for col in state:
        output += "  [ "
        for tile in col:
            output += str(tile) + ", "

        output = output[:-2] + " ]\n"

    output += "]"
    print(output)
