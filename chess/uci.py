#!/usr/bin/env pypy -u
# -*- coding: utf-8 -*-

# Credit to SunFish, https://github.com/thomasahle/sunfish/blob/master/uci.py
import importlib
import re
import sys
import time
import chess
from agent import Agent
WHITE,BLACK = 0,1

class Unbuffered(object):
    def __init__(self, stream):
        self.stream = stream
    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
    def __getattr__(self, attr):
        return getattr(self.stream, attr)
sys.stdout = Unbuffered(sys.stdout)

def main():
    board = chess.Board()
    agent = 
    forced = False
    color = WHITE
    our_time, opp_time = 1000, 1000 # time in centi-seconds
    show_thinking = True

    # print name of chess engine
    print('OMGchess')

    stack = []
    while True:
        if stack:
            smove = stack.pop()
        else: smove = input()

        if smove == 'quit':
            break

        elif smove == 'uci':
            print('uciok')

        elif smove == 'isready':
            print('readyok')

        elif smove == 'ucinewgame':
            stack.append('position fen ' + chess.Board().fen())

        elif smove.startswith('position'):
            params = smove.split(' ', 2)
            if params[1] == 'fen':
                fen = params[2]
                board = chess.Board(fen)
                color = WHITE if fen.split()[1] == 'w' else BLACK

        elif smove.startswith('go'):
            #  default options
            depth = 1000
            movetime = -1

            # parse parameters
            params = smove.split(' ')
            if len(params) == 1: continue

            i = 0
            while i < len(params):
                param = params[i]
                if param == 'depth':
                    i += 1
                    depth = int(params[i])
                if param == 'movetime':
                    i += 1
                    movetime = int(params[i])
                i += 1

            move = agent.compute_action(movetime)

            if move == None:
                print('resign')
            else:
                print('bestmove ' + move)

        elif smove.startswith('time'):
            our_time = int(smove.split()[1])

        elif smove.startswith('otim'):
            opp_time = int(smove.split()[1])

        else:
            pass

if __name__ == '__main__':
    main()