"""PPO agent
"""

class BaseAgent:
    def __init__(self):
        pass

    def step(self):
        pass

    def save(self, fpath):
        pass

    def restore(self, fpath):
        pass


class PPOAgent(BaseAgent):
    def __init__(self):
        """PPO Agent
        Parameters
        ----------
        environment :
            対象環境オブジェクト
        network :
            予測モデルそのもの
        """
        pass

    def step(self):
        """1ステップ進める
        """
        pass
