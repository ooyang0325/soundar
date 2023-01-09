import json
import pandas as pd
from aimyon import *
from math import atan2, pi, floor, sqrt, ceil

file_path = '../data/dataset/'

def build_coordinate_mapping_table(count=(4, 4)):
    """
    mapping coordinate to block_id, coordinates have to be multiple of 50

    count refers to the coordinate table size
    now only support 8 x 8 and 4 x 4

    table and dic seem to be useless, but I'd like to pretend nothing happened :/
    """

    nx, ny = 0, 0
    table = [[0 for _ in range(8)] for _ in range(8)]
    dx = [1, -1, -1, 1]
    dy = [1, 1, -1, -1]
    dic = {}
    invdic = {}
    for i in range(4):
        for j in range(i + 1):
            tx = i * i + j
            ty = i * i + 2 * i - j
            for d in range(4):
                if d % 2 == 0:
                    dic[tx + 16 * d] = (50 * dx[d] * (i + 1), 50 * dy[d] * (j + 1))
                    dic[ty + 16 * d] = (50 * dx[d] * (j + 1), 50 * dy[d] * (i + 1))
                    invdic[(50 * dx[d] * (i + 1), 50 * dy[d] * (j + 1))] = tx + 16 * d
                    invdic[(50 * dx[d] * (j + 1), 50 * dy[d] * (i + 1))] = ty + 16 * d
                if d % 2 == 1:
                    dic[ty + 16 * d] = (50 * dx[d] * (i + 1), 50 * dy[d] * (j + 1))
                    dic[tx + 16 * d] = (50 * dx[d] * (j + 1), 50 * dy[d] * (i + 1))
                    invdic[(50 * dx[d] * (i + 1), 50 * dy[d] * (j + 1))] = ty + 16 * d
                    invdic[(50 * dx[d] * (j + 1), 50 * dy[d] * (i + 1))] = tx + 16 * d

            # first quadrant
            table[3 - j][4 + i] = tx
            table[3 - i][4 + j] = ty
            # second quadrant
            table[3 - i][3 - j] = tx + 16
            table[3 - j][3 - i] = ty + 16
            # third quadrant
            table[4 + j][3 - i] = tx + 32
            table[4 + i][3 - j] = ty + 32
            # fourth quadrant
            table[4 + i][4 + j] = tx + 48
            table[4 + j][4 + i] = ty + 48

    

    # print(*table, sep='\n')

    dic = dict(sorted(dic.items()))
    invdic = dict(sorted(invdic.items(), key=lambda x: x[1]))

    if count == (4, 4):
        tmp = []
        for tup, v in invdic.items():
            if v % 16 == 2:
                invdic[tup] = v // 16 * 4
            elif v % 16 == 10:
                invdic[tup] = v // 16 * 4 + 1
            elif v % 16 == 12:
                invdic[tup] = v // 16 * 4 + 2
            elif v % 16 == 14:
                invdic[tup] = v // 16 * 4 + 3
            else:
                tmp.append(tup)
        for v in tmp:
            del invdic[v]

    if count == (2, 2):
        tmp = []
        for tup, v in invdic.items():
            if v % 16 == 12:
                invdic[tup] = v // 16
            else:
                tmp.append(tup)
        for v in tmp:
            del invdic[v]
                
    print(invdic)

    return invdic


def add_coordinate(count, freq):
    """
    Add coordinate label to the json file.
    Coordinate is labeled like this: https://i.imgur.com/upXuv8t.png

    count refers to the coordinate table size
    now only support 8 x 8 and 4 x 4
    """

    print(f'{aimyon=}')

    freq = str(freq)
    coord_table = build_coordinate_mapping_table(count)
    if count == (4, 4):
        block_size = 100
    elif count == (2, 2):
        block_size = 200
    else:
        block_size = 50 

    df = pd.read_csv(file_path + freq + '/loc.csv')
    blockarr = []

    for x, y, filename in zip(df['x'], df['y'], df['filename']):
        nx = (abs(x) // block_size + (1 if abs(x) % block_size != 0 else 0)) * (1 if x > 0 else -1) * block_size
        ny = (abs(y) // block_size + (1 if abs(y) % block_size != 0 else 0)) * (1 if y > 0 else -1) * block_size
        if nx == 0: nx = block_size
        if ny == 0: ny = block_size
        # print(x, y, nx, ny)
        block = coord_table[(nx, ny)]
        blockarr.append(block)

    df.insert(3, column='block', value=blockarr)
    
    df.to_csv(file_path + freq + '/' + freq + '.csv')

if __name__ == '__main__':
    add_coordinate((2, 2), 1046.5)
    # json_to_csv()
