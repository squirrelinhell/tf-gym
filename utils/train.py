
import time

class Agent:
    def next_action(self, obs):
        raise NotImplementedError("Not implemented: next_action()")

    def take_reward(self, reward, episode_end):
        pass

def thread(env, agent, steps=50000, print_stats=True):
    history = []
    obs = env.reset()
    train_time = time.time()

    for t in range(steps):
        action = agent.next_action(obs)
        obs, reward, done, _ = env.step(action)
        agent.take_reward(reward, episode_end=done)
        if done:
            obs = env.reset()

    train_time = time.time() - train_time
    env.close()

    if print_stats:
        print(
            "Finished %d training steps in %.2fs: %.2f steps/s" %
                (steps, train_time, steps/train_time),
            flush=True
        )

    return history

def parse_args(*args):
    result = dict()
    for arg in (",".join(args)).split(","):
        if not ":" in arg:
            continue
        name, value = arg.split(":", 1)
        name, value = name.strip(), value.strip()
        if value[0] in "0123456789-":
            if "." in value:
                result[name] = float(value)
            elif len(value) >= 1:
                result[name] = int(value)
        else:
            result[name] = value
    return result
