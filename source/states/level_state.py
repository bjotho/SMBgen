from .. import constants as c

state = [[]]


def insert_tile(tile):
    if len(state[-1]) >= c.COL_HEIGHT:
        state.append([])

    state[-1].append(tile)


def print_state():
    output = "[\n  "
    for col in state:
        output += "[ "
        for tile in col:
            output += str(tile) + ", "

        output = output[:-2] + " ]\n"

    output += "]"
    print(output)