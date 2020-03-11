import numpy as np
from gym.envs.toy_text import blackjack
import progressbar

GAMMA = 1
returns = {}
Q = {}


def add_to_returns(new_state, action, G):
    global returns

    if (new_state, action) in returns:
        returns[(new_state, action)].append(G)
    else:
        returns[(new_state, action)] = [G]


# initial policy is hit unless sum is 20 or 21
def initial_policy():
    policy = {}

    for usable_ace in [True, False]:
        for players_card in range(2, 22):
            for dealers_card in range(1, 11):
                if players_card < 20:
                    policy[(players_card, dealers_card, usable_ace)] = True
                else:
                    policy[(players_card, dealers_card, usable_ace)] = False

    return policy


def find_best_Q(state):
    values = []
    for key in Q:
        if key[0] == state:
            if (key[0], True) in Q:
                values.append(Q[(key[0], True)])
            if (key[0], False) in Q:
                values.append(Q[(key[0], False)])

    max_value = np.argmax(values)

    return True if max_value == 0 else False


def randomly_select_action():
    action = np.random.choice([True, False])
    return True if action else False


def print_policy(policy):
    print("State  -  Action")
    for key in policy:
        print(key, "Hit" if policy[key] else "Stick")


def main():
    global returns, Q

    env = blackjack.BlackjackEnv()
    policy = initial_policy()

    for i in progressbar.progressbar(range(100000)):

        action = randomly_select_action()
        G = 0

        passed_states = [(blackjack.sum_hand(env.player), env.dealer[0], blackjack.usable_ace(env.player))]
        passed_actions = [action]
        passed_rewards = []

        while True:
            new_state, rew, done, _ = env.step(action)
            if done:
                passed_rewards.append(rew)
                env.reset()
                break

            passed_states.append(new_state)
            passed_rewards.append(rew)

            action = policy[new_state]
            passed_actions.append(action)

        for j, _ in reversed(list(enumerate(passed_states))):
            G = GAMMA * G + passed_rewards[j]
            add_to_returns(passed_states[j], passed_actions[j], G)

            Q[(passed_states[j], passed_actions[j])] = np.average(returns[(passed_states[j], passed_actions[j])])

            policy[passed_states[j]] = find_best_Q(passed_states[j])

    print_policy(policy)


if __name__ == '__main__':
    main()
