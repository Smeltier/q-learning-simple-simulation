from abc import ABC, abstractmethod

from src.maze import Maze


class BaseAgent(ABC):
    """ Classe base para os agentes de aprendizado por reforço. """

    _number_of_actions: int
    _alpha: float
    _gamma: float
    _epsilon: float

    def __init__(self, number_of_actions:int, alpha:float, gamma:float, epsilon:float) -> None:
        """Inicializa o agente com os parâmetros fornecidos.

        Args:
            number_of_actions (int): O número de ações possíveis para o agente.
            alpha (float): A taxa de aprendizado, que determina o quanto o agente atualiza seu conhecimento com base em novas experiências.
            gamma (float): O fator de desconto, que determina a importância das recompensas futuras em comparação com as recompensas imediatas.
            epsilon (float): A taxa de exploração, que determina a probabilidade de o agente escolher uma ação aleatória em vez de seguir a política atual.
        """
        self._number_of_actions = number_of_actions
        self._alpha = alpha
        self._gamma = gamma
        self._epsilon = epsilon

    @abstractmethod
    def update(self, state:tuple[int,int], action:int, maze:Maze) -> tuple[tuple[int,int], float, bool]:
        """Atualiza o estado do agente com base na ação tomada e no ambiente.

        Deve ser implementado por classes derivadas para definir como o agente interage com o ambiente e aprende com suas experiências.

        Args:
            state (tuple[int, int]): O estado atual do agente.
            action (int): A ação tomada pelo agente.
            maze (Maze): O labirinto que representa o ambiente.

        Returns:
            tuple[tuple[int, int], float, bool]: O próximo estado, a recompensa e um indicador de conclusão.
        """
        pass

    @abstractmethod
    def choose_action(self, state:tuple[int,int], maze:Maze) -> int:
        """Escolhe a próxima ação com base no estado atual e no ambiente.

        Args:
            state (tuple[int, int]): O estado atual do agente.
            maze (Maze): O labirinto que representa o ambiente.

        Returns:
            int: A ação escolhida.

        """
        pass

    @abstractmethod
    def learn(self, state:tuple[int,int], action:int, reward:float, next_state:tuple[int,int], done:bool, maze:Maze) -> None:
        """Aprende com a experiência do agente.

        Deve ser implementado por classes derivadas para definir como o agente atualiza seu conhecimento com base nas experiências vivenciadas.

        Args:
            state (tuple[int, int]): O estado atual do agente.
            action (int): A ação tomada pelo agente.
            reward (float): A recompensa recebida após tomar a ação.
            next_state (tuple[int, int]): O próximo estado do agente após tomar a ação.
            done (bool): Um indicador de conclusão, que indica se o episódio terminou.
            maze (Maze): O labirinto que representa o ambiente.
        """
        pass

    @abstractmethod
    def save_model(self, file_name:str) -> None:
        """Salva o modelo do agente em um arquivo."""
        pass

    @abstractmethod
    def load_model(self, file_name:str) -> None:
        """Carrega o modelo do agente a partir de um arquivo."""
        pass

    @property
    def epsilon(self) -> float:
        """Retorna a taxa de exploração do agente."""
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value:float) -> None:
        """Define a taxa de exploração do agente."""
        self._epsilon = value

    @property
    def alpha(self) -> float:
        """Retorna a taxa de aprendizado do agente."""
        return self._alpha

    @alpha.setter
    def alpha(self, value:float) -> None:
        """Define a taxa de aprendizado do agente."""
        self._alpha = value

    @property
    def gamma(self) -> float:
        """Retorna o fator de desconto do agente."""
        return self._gamma

    @gamma.setter
    def gamma(self, value:float) -> None:
        """Define o fator de desconto do agente."""
        self._gamma = value

    @property
    def number_of_actions(self) -> int:
        """Retorna o número de ações possíveis para o agente."""
        return self._number_of_actions
