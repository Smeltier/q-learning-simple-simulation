import argparse

import pygame

from src.agents.approximate_q_learning_agent import ApproximateQLearningAgent
from src.agents.base_agent import BaseAgent
from src.agents.q_learning_agent import QLearningAgent
from src.agents.sarsa_agent import SarsaAgent
from src.maze import Maze
import src.utils as utils


AGENT_NAMES = ("q_learning", "sarsa", "approximate_q_learning")


def _build_agent(
    agent_name: str,
    maze: Maze,
    number_of_actions: int,
    alpha: float,
    gamma: float,
    epsilon: float,
    number_of_features: int,
) -> BaseAgent:
    if agent_name == "q_learning":
        return QLearningAgent(
            rows=maze.rows,
            cols=maze.cols,
            number_of_actions=number_of_actions,
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
        )

    if agent_name == "sarsa":
        return SarsaAgent(
            rows=maze.rows,
            cols=maze.cols,
            number_of_actions=number_of_actions,
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
        )

    if agent_name == "approximate_q_learning":
        return ApproximateQLearningAgent(
            number_of_features=number_of_features,
            number_of_actions=number_of_actions,
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
        )

    raise ValueError(f"Agente desconhecido: {agent_name}")


def _build_caption(agent_name: str) -> str:
    titles = {
        "q_learning": "Q-learning Simulation",
        "sarsa": "SARSA Simulation",
        "approximate_q_learning": "Approximate Q-learning Simulation",
    }
    return titles.get(agent_name, "Reinforcement Learning Simulation")


def run(
    agent_name: str = "q_learning",
    max_episodes: int = 0,
    save_progress: bool = False,
    path_saved: str | None = None,
    save_path: str | None = None,
    show_results: bool = False,
    render: bool = True,
    maze_path: str = "data/playground.txt",
    number_of_actions: int = 4,
    alpha: float = 0.1,
    gamma: float = 0.99,
    epsilon: float = 0.1,
    number_of_features: int = 3,
    fps: int = 20,
    show_weights: bool = False,
) -> None:
    pygame.init()
    pygame.display.set_caption(_build_caption(agent_name))

    width, height = 1000, 800
    screen = None
    clock = None

    if render:
        screen = pygame.display.set_mode((width, height))
        clock = pygame.time.Clock()

    maze = Maze(maze_path)
    agent = _build_agent(
        agent_name=agent_name,
        maze=maze,
        number_of_actions=number_of_actions,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
        number_of_features=number_of_features,
    )

    if path_saved is not None:
        agent.load_model(path_saved)

        if render:
            agent.epsilon = 0.0

    state = maze.reset()
    episode_number = 0
    total_reward = 0.0
    history: list[float] = []

    running = True
    while running:
        if render:
            assert screen is not None
            assert clock is not None

            screen.fill("black")
            clock.tick(fps)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

        if episode_number != 0 and episode_number == max_episodes:
            running = False

        action = agent.choose_action(state, maze)
        next_state, reward, done = agent.update(state, action, maze)

        total_reward += reward
        state = next_state
        maze.agent_position = next_state

        if show_weights and agent_name == "approximate_q_learning":
            weights = getattr(agent, "_weights", None)
            if weights is not None:
                print(
                    "Pesos: "
                    f"Bias: {weights[0]:.2f} | Dist: {weights[1]:.2f} | Wall: {weights[2]:.2f}"
                )

        if done:
            print(f"Episode {episode_number} | Total Reward: {total_reward}")
            state = maze.reset()

            if show_results:
                history.append(total_reward)

            total_reward = 0.0
            episode_number += 1

        if render:
            assert screen is not None

            maze.draw(screen)

            # Apenas métodos tabulares devem desenhar o mapa de cores.
            q_table = getattr(agent, "q_table", None)
            if q_table is not None:
                utils.draw_color_map(screen, maze, next_state, q_table)

            pygame.display.flip()

    if save_progress:
        output_path = save_path or f"treino_{agent_name}_{episode_number}.npy"
        agent.save_model(output_path)

    if show_results:
        utils.show_rewards_graph(history)

    pygame.quit()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulacao parametrizavel de Reinforcement Learning")

    parser.add_argument("--agent", type=str, choices=AGENT_NAMES)
    parser.add_argument("--episodes", type=int, default=0)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--load", type=str, default=None)
    parser.add_argument("--results", action="store_true")
    parser.add_argument("--no-render", action="store_false", dest="render")
    parser.add_argument("--maze", type=str, default="data/playground.txt")
    parser.add_argument("--actions", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--features", type=int, default=3)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--show-weights", action="store_true")
    parser.set_defaults(render=True)

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    run(
        agent_name=args.agent,
        max_episodes=args.episodes,
        save_progress=args.save,
        path_saved=args.load,
        save_path=args.save_path,
        show_results=args.results,
        render=args.render,
        maze_path=args.maze,
        number_of_actions=args.actions,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        number_of_features=args.features,
        fps=args.fps,
        show_weights=args.show_weights,
    )
