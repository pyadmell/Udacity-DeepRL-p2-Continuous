"""PPO agent
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PPOAgent:
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
        # rollout step
        # environment を何ステップか進める
        # states, actions, rewards, next_states の組を得る
        # 同時に log_prob と value も取得

        # reward の計算 .. 割引ありの reward を算出
        # 割引ありの reward から value を引いて advantage としておく

        # ミニバッチ更新する
        # 綺麗にやるのであれば、shuffle すべきところだが
        # on-policy の順序に準じる
