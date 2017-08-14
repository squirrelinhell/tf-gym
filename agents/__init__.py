
class Agent:
    def __init__(o_space, a_space):
        pass

    def next_action(self, obs):
        # returns action from a_space
        raise NotImplementedError("Not implemented: next_action()")

    def take_reward(self, reward, episode_end):
        pass

    def __str__(self):
        return "<Agent>"
