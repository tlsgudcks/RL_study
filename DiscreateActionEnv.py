

# Discreate Action base 환경입니다
# 해당 환경은 RL 월드입니다
# 좌로 이동시 -1의 보상 우로 이동시 +1의 보상을 얻습니다
# 좌우좌좌우에서 좌를 선택시 +1000점을 받습니다
# 목표는 999점을 받는것입니다

import random
import numpy as np
import pandas as pd


class GridWorld():
    def __init__(self):
        self.final_state = 6
        self.c_step = 0
        self.s = [0,0,1] * self.final_state

    def step(self, a):
        if a == 0:
            if self.s == [1,0,0 ,0,1,0 ,1,0,0 ,0,1,0 ,1,0,0, 0,0,1]: # 좌우좌우좌에서 좌 선택시 +1000점
                reward = + 1000
            else:
                reward = -1
            self.move_left()
        else:
            reward = +1
            self.move_right()
        done = self.is_done()
        self.c_step += 3

        df = pd.Series(self.s)
        s = df.to_numpy()

        return s, reward, done

    def move_left(self):
        self.s[self.c_step:self.c_step+3] = [1,0,0]

    def move_right(self):
        self.s[self.c_step:self.c_step+3] = [0,1,0]

    def is_done(self):
        if self.s[-1] == 0:
            return True
        else:
            return False

    def reset(self):
        self.c_step = 0
        self.s = [0, 0, 1] * self.final_state

        df = pd.Series(self.s)
        s = df.to_numpy()

        return s