import numpy as np
import os
from source import constants as c


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
    if replacement in c.ENEMY_IDS:
        replacement = c.AIR_ID

    if old_x == new_x and old_y == new_y and not id == c.SOLID_ID:
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


def get_coordinates(x_px, y_px):
    x = int(x_px // c.TILE_SIZE)
    y = int(c.COL_HEIGHT - ((y_px - c.Y_OFFSET) // c.TILE_SIZE) - 1)
    if x < 0:
        x = 0
    if y < 0:
        y = 0
    elif y >= c.COL_HEIGHT:
        y = c.COL_HEIGHT - 1

    return x, y


def get_observation(player):
    x, y = player.get_coordinates(player)
    start_x = x - c.OBS_RADIUS
    stop_x = x + c.OBS_RADIUS + 1
    start_y = y - c.OBS_RADIUS
    stop_y = y + c.OBS_RADIUS + 1
    offset_x = 0
    prepend_y = 0
    append_y = 0
    observation = []
    for _ in range(start_x, stop_x):
        observation.append([c.AIR_ID for _ in range(c.COL_HEIGHT)])

    if start_x < 0:
        offset_x = np.abs(start_x)
        start_x = 0

    for n, col in enumerate(state[start_x:stop_x]):
        observation[n + offset_x] = col

    if start_y < 0:
        prepend_y = np.abs(start_y)
        start_y = 0
    if stop_y >= c.COL_HEIGHT:
        append_y = stop_y - c.COL_HEIGHT

    for n, _ in enumerate(observation):
        observation[n] = [c.AIR_ID for _ in range(prepend_y)] + \
                         observation[n][start_y:stop_y] + \
                         [c.AIR_ID for _ in range(append_y)]

    return observation


def print_2d(_list, chop=None):
    # Print a 1d or 2d list row by row in console, chopping 1d lists after each element specified by chop
    if chop:
        tmp = []
        for i in range(len(_list) // chop):
            tmp.append(_list[i * chop:(i + 1) * chop])
        remaining = len(_list) - (len(_list) // chop) * chop
        if remaining > 0:
            tmp.append(_list[len(_list) - (remaining + 1):-1])
        _list = tmp

    output = "[\n"
    for col in _list:
        output += "  [ "
        for tile in col:
            output += str(tile) + ", "

        output = output[:-2] + " ]\n"

    output += "]"
    print(output)


def find_latest_checkpoint(_dir):
    largest = -1
    for chkpath in os.listdir(_dir):
        try:
            checkpoint_id = int(chkpath.split('_')[1])

            if checkpoint_id > largest:
                largest = checkpoint_id
        except ValueError:
            pass

    return largest
