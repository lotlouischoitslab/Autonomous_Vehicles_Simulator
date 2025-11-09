import os
import numpy as np
import pygame
import random, math
import Environment

from Double_Deep_Q import DDQNAgent, set_seed   

TOTAL_GAMETIME = 1000
N_EPISODES = 10000

# training knobs
WARMUP_STEPS = 2000         # collect transitions before learning
TRAIN_EVERY = 1             # learn each step once buffer is warm
SAVE_EVERY_EP = 10
RENDER_EVERY_EP = 10

# ---- init ----
set_seed(42)
game = Environment.RacingEnv()
game.fps = 60
os.makedirs("models", exist_ok=True)

ddqn_agent = DDQNAgent(
    state_dim=19,       # <- your observation vector length
    n_actions=5,        # <- your discrete action count
    gamma=0.99,
    lr=1e-4,
    batch_size=512,
    tau=0.005,                # soft target update (done inside learn)
    eps_start=1.0,
    eps_end=0.10,
    eps_decay_steps=500_000,
    buffer_capacity=100_000,
    grad_clip_norm=10.0,
    seed=42,
    device=None,
)

# Uncomment to resume:
# ddqn_agent.load("models/ddqn_latest.pt")

ddqn_scores = []
eps_history = []

def run():
    global ddqn_scores, eps_history
    total_steps = 0

    for e in range(N_EPISODES):
        game.reset()
        done = False
        score = 0.0
        counter = 0

        # your env returns (obs, reward, done) when stepping; get first obs by a no-op
        observation_, reward, done = game.step(0) 
        observation = np.array(observation_, dtype=np.float32)

        gtime = 0
        renderFlag = (e % RENDER_EVERY_EP == 0 and e > 0)

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    # optional: save before exit
                    ddqn_agent.save("models/ddqn_latest.pt")
                    return

            # epsilon-greedy action from online net
            action = ddqn_agent.act(observation) 
            observation_, reward, done = game.step(action)
            observation_ = np.array(observation_, dtype=np.float32)

            # simple “stuck” timeout: end if no reward for 100 ticks
            if reward == 0:
                counter += 1
                if counter > 100:
                    done = True
            else:
                counter = 0

            score += float(reward)

            # push transition (done as float: 1.0 if terminal else 0.0)
            ddqn_agent.push(observation, int(action), float(reward), observation_, float(done))
            observation = observation_

            # learning
            total_steps += 1
            if total_steps > WARMUP_STEPS and (total_steps % TRAIN_EVERY == 0):
                _ = ddqn_agent.learn()  # returns dict with loss/td_error if you want to log

            gtime += 1
            if gtime >= TOTAL_GAMETIME:
                done = True

            if renderFlag:
                game.render(action)

        # logging
        eps_history.append(ddqn_agent.eps.value())
        ddqn_scores.append(score)
        avg_score = np.mean(ddqn_scores[max(0, e-100):(e+1)])

        # periodic save (soft target updates are automatic; no hard copy needed)
        if e % SAVE_EVERY_EP == 0 and e > 0:
            ddqn_agent.save("models/ddqn_latest.pt")
            print("Saved: models/ddqn_latest.pt")

        print(
            f"episode {e:5d} | score {score:8.2f} | avg100 {avg_score:8.2f} "
            f"| eps {ddqn_agent.eps.value():.3f} | steps {total_steps}"
        )

    # final save
    ddqn_agent.save("models/ddqn_final.pt")
    print("Training complete. Saved models/ddqn_final.pt")

if __name__ == "__main__":
    run()
