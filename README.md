## Simulando Reinforcement Learning

O objetivo do projeto é aprender por meio de testes como alguns modelos de Aprendizado por Reforço realmente funcionam.

Para isso, um ambiente simulado foi criado.

As simulações podem ser parametrizadas, podendo modificar a quantidade de episódios, a presença de imagem, salvamento, entre outras opções.

#### Parametrização da simulação

Agora existe um entrypoint unico para executar os experimentos e escolher o agente via argumento:

```bash
python -m src.simulation ...
```

Argumentos disponiveis:

- `--agent`: define o agente utilizado na simulação.
- `--episodes`: quantidade máxima de episodios executados. O valor padrão `0` roda indefinidamente até a janela ser fechada.
- `--save`: salva o modelo treinado ao final da execução.
- `--save-path`: define o caminho/arquivo de saida para salvar o modelo. Se omitido, usa o padrão `treino_<agent>_<episodes>.npy`.
- `--load`: carrega um modelo salvo anteriormente antes de iniciar a execução.
- `--results`: exibe o grafico com o histórico de recompensas por episódio ao final da simulação.
- `--no-render`: executa a simulação sem abrir a interface gráfica do Pygame.
- `--maze`: define o arquivo do labirinto utilizado na simulação. O padrão é `data/playground.txt`.
- `--alpha`: taxa de aprendizado do agente. O padrão e `0.1`.
- `--gamma`: fator de desconto das recompensas futuras. O padrão e `0.99`.
- `--epsilon`: taxa de exploração da politica epsilon-greedy. O padrão e `0.1`.
- `--fps`: controla a velocidade de renderização da simulação. O padrão é `20`.

Agentes disponiveis no parametro `--agent`:

- `q_learning`
- `approximate_q_learning`
- `sarsa`

Exemplos:

```bash
# Treinar SARSA sem render
python -m src.simulation --agent sarsa --episodes 3000 --no-render --save

# Rodar Approximate Q-learning mostrando pesos
python -m src.simulation --agent approximate_q_learning --show-weights

# Carregar um modelo salvo e exibir grafico de recompensas
python -m src.simulation --agent q_learning --load episodes/treino_10000.npy --results
```
