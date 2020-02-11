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


def update_observation(old_x_px, old_y_px, new_x_px, new_y_px, id):
    old_x = int(old_x_px // c.TILE_SIZE)
    old_y = int(c.COL_HEIGHT - ((old_y_px - c.Y_OFFSET) // c.TILE_SIZE) - 1)
    new_x = int(new_x_px // c.TILE_SIZE)
    new_y = int(c.COL_HEIGHT - ((new_y_px - c.Y_OFFSET) // c.TILE_SIZE) - 1)

    if old_x == new_x and old_y == new_y:
        return

    if old_y >= c.COL_HEIGHT:
        old_y = c.COL_HEIGHT - 1
    if new_y >= c.COL_HEIGHT:
        new_y = c.COL_HEIGHT - 1

    while old_x >= len(state) or new_x >= len(state):
        state.append([c.AIR_ID for _ in range(c.COL_HEIGHT)])

    state[old_x][old_y] = c.AIR_ID
    state[new_x][new_y] = id

    print_state()


def delete_observation(x_px, y_px):
    x = int(x_px // c.TILE_SIZE)
    y = int(c.COL_HEIGHT - ((y_px - c.Y_OFFSET) // c.TILE_SIZE) - 1)
    print("deleting (", x, y, ")")
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
