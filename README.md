# PPO_options_hedge


This is an illuatration of the Proximal Policy Optimization (PPO) implementation for optimizing option trading size.


1. Reinforcement Learning (RL) Basics:
In RL, an agent interacts with an environment over a series of time steps. The agent:
Observes the current state s_t at time step t.
Takes an action a_t, which impacts the environment.
Receives a reward r_t based on the outcome of the action.
Transitions to a new state s_{t+1}.
The goal of the agent is to learn a policy pi(a|s) (a probability distribution over actions given a state) that maximizes the expected cumulative reward (also known as the return), typically discounted over time.

The discounted cumulative reward at time t is:
R_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ... = sum_{k=0}^{infinity} gamma^k * r_{t+k}
Where:
gamma is the discount factor (0 <= gamma < 1), which prioritizes immediate rewards over distant ones.


2. Proximal Policy Optimization (PPO) Overview:
PPO is a policy-gradient method, which means it directly optimizes the policy pi(a|s) rather than learning a value function explicitly. PPO improves stability by limiting the size of policy updates.

The PPO Objective:
The primary objective in PPO is to maximize the expected advantage A_t while keeping the policy update conservative. The PPO objective function is:
L^CLIP(theta) = E_t [ min(r_t(theta) * A_t, clip(r_t(theta), 1 - epsilon, 1 + epsilon) * A_t) ]
Where:
r_t(theta) = pi_theta(a_t|s_t) / pi_{old}(a_t|s_t) is the ratio of the probability of action a_t under the new policy pi_theta to that under the old policy pi_{old}.
A_t is the advantage function, which measures how much better an action is compared to the average action.
epsilon is a small hyperparameter that controls how much deviation from the old policy is allowed (epsilon is typically 0.1 or 0.2).
The objective ensures that the ratio r_t(theta) stays within the range [1 - epsilon, 1 + epsilon], preventing large updates that could destabilize the training process.


3. Environment Setup:
The environment is modeled as a Markov Decision Process (MDP) where:
The state s_t includes the option price, option greeks (delta, gamma, etc.), market volatility, and other relevant financial metrics.
The action a_t is the trading size of options (scaled between -1 and 1), where:
Positive values indicate buying.
Negative values indicate selling.
The reward r_t is the profit or loss based on the option price change multiplied by the trading size, adjusted with a penalty for high-risk actions.

The step function calculates the reward:
reward = trading_size * (option_price_{t+1} - option_price_t) - risk_penalty
Where the risk_penalty is based on the absolute size of the trade, encouraging smaller trades during high volatility.


4. Advantage Function (A_t):
The advantage function A_t helps the agent decide how good a specific action was compared to the expected reward. It is given by:
A_t = Q(s_t, a_t) - V(s_t)
Where:
Q(s_t, a_t) is the action-value function, representing the expected return if the agent takes action a_t in state s_t and follows the policy thereafter.
V(s_t) is the value function, which estimates the expected return if the agent starts in state s_t and follows the policy.

In PPO, the advantage is often computed using Generalized Advantage Estimation (GAE) to reduce variance:
A_t = delta_t + (gamma * lambda) * delta_{t+1} + ... + (gamma * lambda)^{T-t+1} * delta_{T-1}
Where:
delta_t = r_t + gamma * V(s_{t+1}) - V(s_t) is the temporal difference (TD) error.
lambda is a hyperparameter (usually between 0.9 and 1) that controls bias-variance trade-off.


5. Policy Update:
To update the policy, PPO uses stochastic gradient ascent to maximize the PPO objective. At each iteration, the policy is updated as:
theta_{new} = theta_{old} + alpha * grad(L^CLIP(theta_{old}))
Where:
alpha is the learning rate.
grad(L^CLIP(theta_{old})) is the gradient of the PPO objective with respect to the policy parameters.


6. Clipping in PPO:
PPO uses clipping to avoid making large updates that could lead to poor performance. The clipping operation ensures that the probability ratio r_t(theta) does not deviate too much from 1:
r_t(theta) = pi_theta(a_t|s_t) / pi_{old}(a_t|s_t)
clip(r_t(theta), 1 - epsilon, 1 + epsilon)
This restricts how much the new policy pi_theta can differ from the old policy pi_{old}, ensuring more stable and consistent learning.


7. Training Process:
Initialize the policy pi_theta and value function V(s_t) with random parameters.
For each iteration:
Collect trajectories (state, action, reward, next state) by interacting with the environment using the current policy.
Compute the advantages A_t and the reward-to-go R_t.
Update the policy by maximizing the PPO objective L^CLIP(theta).
Update the value function to minimize the squared error between the predicted value and the actual reward-to-go:
Loss_V = E_t [(V(s_t) - R_t)^2]
Repeat until convergence.
