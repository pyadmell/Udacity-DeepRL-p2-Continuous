# 実装手順
----
# 基本方針
1. 基本ロジックの実装
  - agent
    - REINFORCEの実装
    - PPOの実装
  - 実行
    - trajectory の収集(rollout)
    - rollout結果の整理/replay-bufferへの格納
    - 各時点で trajectory の末端までの累積報酬を用いる
    - baseline としては rollout 内で平均・標準偏差で正規化する
2. 基本ロジックの改良
  - GAEの実装
  - Value Network の利用
    - 要確認 : Value Network と Policy Network でネットワークを
      どのように共通化すべきか
3. パラメータチューニング

# 実装の自由度
- 基本の考え方 : DeepRLに準じて on-policy とする
  - 一定の rollout 回数だけ trajectory を収集
  - rollout 後に結果を整理する。
    - この際に GAE の advantage 計算をやってしまう
  - 整理された trajectory を用いてミニバッチ更新
    - 直前の rollout 結果バッファからランダムサンプリング
    - 一応 on-policy である。
  - pi_old とは、ミニバッチループに入る前の確率値のことを言う。
    - agent で行動を選択した際にその時点での確率を持っておけば十分。
    - 特段古いネットワークを保持しておく必要はない。
- Agentアルゴリズム
  - PPO + GAE とする
  - GAE にしたがって TD-error から Generalized Advantage を算出
  - 1本の trajectory で再度正規化をする（これは必要なのか？）
- policy network
  - continuous 出力するものでなくてはいけない。
    - gaussian とする。
    - draw はそこからのノイズ出力とする。
    - action の logprob の求め方は DeepRL の実装に準じる
      - action を与えた場合はその action のlogprob（事後の評価）
      - action を抽出した場合は、**抽出した action の** logprob
        - mean の logprob ではない。
      - std は 1 にしているらしい
      - mean は tanh で出力しているらしい（これでいいのかな？）
  - torch distribution を用いて効率よく実装できる
    - dist = torch.distribution.Normal(mu, sigma)
      - 分布オブジェクトの生成
    - dist.sample() : 1サンプルの生成。多分サイズも指定できる
    - dist.logprob() : 対数確率(尤度)の算出
    - dist.entropy() : エントロピーの算出
- Baseline
  - Value Network とする。
  - Baseline の update は、累積報酬 または TD-error を用いる
    - Deep RL では trajectory で最大の累積報酬を利用
- network の共通化
  - state から中間層までの出力は共通化する。

# 設計
- Network (GaussianActorCriticNetwork)
  - input として、states, actions=None
  - output として actions, logprob, entropy, v(=state value) を返す
- Agent (PPO)
  - environment と network を initializer でもらう
  - self.step() が実装されている
    - rolloutする
    - reward, advantage 計算する
    - minibatch してネットワーク更新する
- Agent (DDPG)
  - experience replay + softupdate の合わせ技
  - critic が Q-function になるので、書き替えないとまずい
  - critic 更新後, actor を更新する必要がある。
    - critic, actor 別々に optimizer を設定する
    - それぞれ zero_grad, optimizer, optimize する必要あり
- そとからはひたすら step で回す
  - ここは notebook でやるのでよかろう
- いずれも BaseNetwork, BaseAgent を用意しておく
  - モデル保存部分のみ共通化しておく
