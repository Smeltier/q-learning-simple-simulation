import random as rd
from typing import Any

import numpy as np

from src.agents.base_agent import BaseAgent
from src.maze import Maze


class SarsaAgent(BaseAgent):

    _q_table: np.ndarray[tuple[int, int, int], np.dtype[np.floating[Any]]]

    def __init__(self, rows:int, cols:int, number_of_actions:int, alpha:float, gamma:float, epsilon:float) -> None:
        super().__init__(number_of_actions, alpha, gamma, epsilon)
        self._q_table = np.zeros((rows, cols, number_of_actions))

    def update(self, state:tuple[int,int], action:int, maze:Maze) -> tuple[tuple[int,int], float, bool]:
        next_state = maze.predict_next_state(state, action)
        reward = maze.get_reward(next_state)
        done = maze.is_target(*next_state)

        self.learn(state, action, reward, next_state, done, maze)

        return next_state, reward, done

    def learn(self, state:tuple[int,int], action:int, reward:float, next_state:tuple[int,int], done:bool, maze:Maze) -> None:
        row, col = state
        nrow, ncol = next_state

        old_value = self._q_table[row, col, action]

        if done:
            target = reward
        else:
            next_action = self.choose_action(next_state, maze)
            next_value = self._q_table[nrow, ncol, next_action]
            target = reward + self._gamma * next_value

        self._q_table[row, col, action] += self._alpha * (target - old_value)

    def choose_action(self, state: tuple[int, int], maze:Maze) -> int:
        row, col = state

        # epsilon-greedy
        if rd.uniform(0, 1) < self._epsilon:
            return rd.randrange(0, self._number_of_actions)
        return int(np.argmax(self._q_table[row, col]))

    def save_model(self, file_name:str) -> None:
        np.save(file_name, self._q_table)

    def load_model(self, file_name:str) -> None:
        try:
            self._q_table = np.load(file_name)
        except FileNotFoundError:
            print("Arquivo não encontrado.")

    @property
    def q_table(self) -> np.ndarray[tuple[int, int, int], np.dtype[np.floating[Any]]]:
        """Retorna a tabela Q do agente."""
        return self._q_table
