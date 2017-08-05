
from utils import *

class Agent:
    def __str__(self):
        return "<Agent>"

    def step(self, obs, reward, done):
        raise NotImplementedError("Agent requires a step() method")
