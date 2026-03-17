O objetivo desse projeto é aprender por meio de testes como alguns modelos de Aprendizado por Reforço realmente funcionam.

Para isso, um ambiente simulado foi criado.

As simulações podem ser parametrizadas, podendo modificar a quantidade de episódios, a presença de imagem, salvamento, entre outras opções.

## Simulacao unica parametrizavel

Agora existe um entrypoint unico para executar os experimentos e escolher o agente via argumento:

```bash
python -m src.simulations.simulation --agent q_learning --episodes 1000
```

Agentes disponiveis no parametro `--agent`:

- `q_learning`
- `approximate_q_learning`
- `sarsa`

Exemplos:

```bash
# Treinar SARSA sem render
python -m src.simulations.simulation --agent sarsa --episodes 3000 --no-render --save

# Rodar Approximate Q-learning mostrando pesos
python -m src.simulations.simulation --agent approximate_q_learning --show-weights
```
