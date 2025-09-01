# Projeto 1: Robô de Reciclagem

**Autores:**

* Juliano Genari de Araújo
* Isaque Vieira Machado Pim
* Eduardo Vianna

**Data:** 02/09/2025

## Introdução

Este projeto implementa o problema do Robô de Reciclagem (Exemplo 3.3 de Sutton & Barto) utilizando o algoritmo de Temporal Difference (TD) learning. O robô aprende uma política ótima para maximizar a coleta de latas considerando o nível da bateria, desenvolvendo estratégias que balanceiam busca ativa por latas e conservação de energia.

## 1. Descrição do Problema

### 1.1 Modelo de Processo de Decisão de Markov (MDP)

O Robô de Reciclagem é modelado como um MDP finito com:

- **Estados (S)**: `{high, low}` - níveis de carga da bateria
- **Ações disponíveis**:
  - Estado `high`: `{search, wait}`
  - Estado `low`: `{search, wait, recharge}`
- **Recompensas**: baseadas na coleta de latas e penalidades por esgotamento da bateria

### 1.2 Dinâmica do Ambiente

As transições seguem probabilidades estocásticas definidas pelos parâmetros α e β:

| Estado | Ação | Próximo Estado | Probabilidade | Recompensa |
|--------|------|----------------|---------------|------------|
| high | search | high | α = 0.7 | r_search = 3.0 |
| high | search | low | 1-α = 0.3 | r_search = 3.0 |
| high | wait | high | 1.0 | r_wait = 1.0 |
| low | search | low | β = 0.6 | r_search = 3.0 |
| low | search | high | 1-β = 0.4 | -3.0 (resgate) |
| low | wait | low | 1.0 | r_wait = 1.0 |
| low | recharge | high | 1.0 | 0.0 |

## 2. Implementação

### 2.1 Arquitetura do Sistema

O projeto foi estruturado em dois módulos principais:

1. **`classes.py`**: Todas as classes do sistema (ambiente, agente TD, experimento)
2. **`main.py`**: Script unificado para treinamento e visualização

### 2.2 Classe `RecyclingRobotEnvironment`

```python
class RecyclingRobotEnvironment:
    def __init__(self, alpha=0.7, beta=0.6, r_search=3.0, r_wait=1.0):
        # Implementa o MDP com transições estocásticas

    def step(self, action: Action) -> tuple[State, float]:
        # Executa ação e retorna próximo estado e recompensa

    def get_valid_actions(self, state: State) -> list[Action]:
        # Retorna ações válidas para um estado
```

**Características principais:**
- Validação de ações por estado
- Transições estocásticas conforme probabilidades α e β
- Sistema de recompensas balanceado (r_search > r_wait)
- Método `step()` para execução de ações

### 2.3 Classe `TemporalDifferenceAgent`

```python
class TemporalDifferenceAgent:
    def __init__(self, environment, learning_rate=0.1,
                 discount_factor=0.9, epsilon=0.1):
        # Implementa TD(0) com política ε-greedy

    def select_action(self, state: State) -> Action:
        # Seleciona ação usando política ε-greedy

    def update_q_value(self, state, action, reward, next_state):
        # Atualiza Q-values usando regra TD(0)

    def get_policy(self) -> dict[State, Action]:
        # Extrai política greedy dos Q-values
```

**Algoritmo TD(0) implementado:**
```
Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
```

**Características:**
- Função valor-ação Q(s,a) inicializada em zero
- Política ε-greedy para exploração
- Decaimento de ε ao longo do treinamento
- Extração de política ótima greedy

### 2.4 Classe `RecyclingRobotExperiment`

```python
class RecyclingRobotExperiment:
    def __init__(self, steps_per_epoch=1000, learning_rate=0.1, ...):
        # Pipeline completo de experimento

    def train_agent(self, num_epochs: int) -> TrainingResults:
        # Treina agente por múltiplas épocas

    def run_complete_experiment(self, num_runs=5, num_epochs=100):
        # Executa experimento completo com visualizações

    def plot_training_curve(self, all_rewards, ...):
        # Gera gráfico de curva de aprendizado

    def create_policy_heatmap(self, agent, ...):
        # Cria mapa de calor da política
```

### 2.5 Sistema de Treinamento

- **Épocas**: 100 épocas por execução
- **Passos por época**: 1.000 passos
- **Múltiplas execuções**: 5 execuções independentes para média
- **Parâmetros de aprendizado**:
  - α (learning rate): 0.1
  - γ (discount factor): 0.9
  - ε inicial: 0.3, decaimento: 0.995, mínimo: 0.01

## 3. Parâmetros Escolhidos

### 3.1 Justificativa dos Parâmetros do Ambiente

- **α = 0.7**: Busca em estado `high` mantém bateria alta 70% do tempo, simulando eficiência energética moderada
- **β = 0.6**: Busca em estado `low` mantém bateria baixa 60% do tempo, refletindo alto risco de esgotamento
- **r_search = 3.0**: Recompensa alta por busca ativa (maior eficiência na coleta)
- **r_wait = 1.0**: Recompensa baixa por espera (conservação de energia)
- **Restrição respeitada**: r_search (3.0) > r_wait (1.0)

### 3.2 Justificativa dos Parâmetros de Aprendizado

- **Learning rate (α = 0.1)**: Valor moderado para convergência estável
- **Discount factor (γ = 0.9)**: Valoriza recompensas futuras mantendo foco no presente
- **Exploração (ε inicial = 0.3)**: Alta exploração inicial com decaimento gradual

## 4. Resultados e Análise

### 4.1 Performance de Aprendizado

**Métricas finais (média de 5 execuções):**
- **Recompensa final**: 2.186,40 ± 55,11
- **Recompensa máxima**: ~2.220
- **Convergência**: Estabilização após ~30 épocas

![Curva de Treinamento](training_curve.png)

A curva de aprendizado revela:
- **Convergência rápida**: Política estável em ~30 épocas
- **Estabilidade**: Baixa variância após convergência
- **Tendência positiva**: Melhoria consistente da performance
- **Eficiência do TD(0)**: Algoritmo convergiu rapidamente para política ótima

### 4.2 Política Ótima e Função Valor-Ação

A política ótima convergiu consistentemente para:

| Estado | Ação Ótima | Justificativa |
|--------|------------|---------------|
| **high** | **search** | Máxima eficiência quando bateria permite |
| **low** | **recharge** | Evita penalidade de -3 por esgotamento |

![Mapa de Calor da Política](policy_heatmap.png)

![Mapa de Calor dos Q-Values](q_values_heatmap.png)

**Análise da Função Valor-Ação:**
- **Q(high, search)**: Valor mais alto, confirmando ação ótima
- **Q(low, recharge)**: Valor superior a search em estado low
- **Q(*, wait)**: Valores baixos, confirmando ineficiência da espera

### 4.3 Comportamento do Algoritmo e Insights

**Características do TD(0):**
- **Eficiência**: Convergência rápida para política ótima
- **Robustez**: Resultados consistentes entre execuções (baixo desvio padrão)
- **Adaptabilidade**: Balanceamento eficaz entre exploração e exploitação

**Insights da Política Aprendida:**
- **Estado high**: Maximiza coleta através de busca ativa
- **Estado low**: Prioriza conservação através de recarga preventiva
- **Estratégia conservadora**: Evita riscos de esgotamento da bateria
- **Comportamento intuitivo**: A estratégia aprendida reflete decisões racionais

**Impacto dos Parâmetros:**
- **α=0.7 e β=0.6**: Definem trade-off adequado entre eficiência e risco
- **r_search=3.0 > r_wait=1.0**: Incentivam busca ativa mantendo viabilidade energética
- **Learning rate=0.1**: Permite convergência estável sem oscilações

## 5. Conclusões

O algoritmo de Temporal Difference (TD) demonstrou eficácia excepcional na solução do problema do Robô de Reciclagem, convergindo rapidamente para uma política ótima intuitiva que maximiza coleta quando a bateria está alta (search) e prioriza conservação quando baixa (recharge). A robustez da implementação ficou evidenciada pela consistência dos resultados entre múltiplas execuções independentes, com baixo desvio padrão na performance final. O balanceamento adequado dos parâmetros (α=0.7, β=0.6, r_search=3.0, r_wait=1.0) permitiu um trade-off eficiente entre eficiência na coleta e conservação energética, resultando em uma estratégia conservadora que evita riscos de esgotamento da bateria enquanto maximiza recompensas a longo prazo.

## 6. Referências

1. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press. Example 3.3: Recycling Robot.

2. Algoritmo de Temporal Difference baseado no Capítulo 6 do livro-texto.

## Agradecimentos

Este projeto foi desenvolvido como exercício para a disciplina de Aprendizado por Reforço (Reinforcement Learning) ministrada pelo Prof. Flávio Coelho.

## Anexos

### A. Lista de Arquivos

**Código fonte:**
- `classes.py`: Todas as classes do sistema (ambiente, agente TD, experimento)
- `main.py`: Script unificado para treinamento e visualização

**Resultados gerados:**
- `rewards.txt`: Dados de recompensa acumulada por época
- `training_results.txt`: Resultados detalhados do treinamento
- `training_curve.png`: Curva de aprendizado com média móvel
- `policy_heatmap.png`: Mapa de calor da política ótima
- `q_values_heatmap.png`: Mapa de calor dos valores Q(s,a)

**Configuração e documentação:**
- `pyproject.toml`: Configuração do projeto e dependências
- `requirements.txt`: Lista de dependências em formato legacy
- `uv.lock`: Arquivo de lock das dependências
- `README.md`: Este arquivo

### B. Preparação e Execução

**Pré-requisitos:**
- Python 3.13+ (recomenda-se usar pyenv)
- uv (gerenciador de dependências)

**Configuração do ambiente:**
```bash
# Com pyenv instalado, sincronize as dependências
uv sync
```

**Execução do projeto:**
```bash
# Executar treinamento completo com visualizações
uv run python main.py
```