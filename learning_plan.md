# Reinforcement Learning Path (CS285 Inspired)

This plan guides you from Basic RL to Advanced topics like DiscoRL.

## Phase 1: Foundations & Basic RL
**Reference**: CS285 Lectures 1-5
- **Concepts**: MDPs, Imitation Learning, Policy Gradients (REINFORCE).
- **Implementation Goal**: Policy Gradient (PG) on `CartPole-v1`.
- **Resources**:
    - Lecture 4: Intro to RL
    - Lecture 5: Policy Gradients

## Phase 2: Q-Learning & DQN
**Reference**: CS285 Lectures 7-8
- **Concepts**: Value Functions, Q-Learning, DQN, Double DQN, Prioritized Replay.
- **Implementation Goal**: DQN on `CartPole-v1` / `LunarLander-v2`.
- **Resources**:
    - Lecture 7: Value Function Methods
    - Lecture 8: Deep RL with Q-Functions

## Phase 3: Advanced Policy Gradients (PPO/TRPO)
**Reference**: CS285 Lecture 9
- **Concepts**: Natural Gradient, TRPO, PPO (Proximal Policy Optimization).
- **Implementation Goal**: PPO on `LunarLander-v2` / `Mujoco` (if available).
- **Resources**:
    - Lecture 9: Advanced Policy Gradients

## Phase 4: Advanced / DiscoRL
**Reference**: Advanced Reading / DiscoRL Papers
- **Concepts**:
    - **DisCo RL** (Distribution-Conditioned RL): Goal-conditioned RL, efficient exploration.
    - **DiscoRL** (Meta-RL): Discovering RL algorithms.
- **Implementation Goal**: Replicate a simplified DiscoRL algorithm (distribution-conditioned) or use an offline RL dataset.
- **Resources**:
    - [DisCo RL Paper (NeurIPS 2021)](https://sites.google.com/view/discorl) or Search for "Discovering RL" DeepMind paper.

## Comparison Framework
use `src/main.py` to run experiments.
- Metrics: Return vs. Episodes, Wall-clock time, Sample efficiency.
