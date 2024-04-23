import argparse
import json
import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('game', type=open)
    parser.add_argument('--output', '-o', type=argparse.FileType('w'), default=sys.stdout)

    args = parser.parse_args()
    game = json.load(args.game)

    local_game = {
        "name": [game[0][0], game[1][0], game[2][0]],
        "score": game[-1][2],
        "game": [[i%3, g[1], g[2]] for i, g in enumerate(game)],
    }
    obj = json.dumps(local_game)
    args.output.write(obj)