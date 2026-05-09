# Coverage Path Planning com Recurrent PPO

## 1. Objetivo

O objetivo do projeto é treinar um agente com reinforcement learning para Coverage Path Planning (CPP), em um grid-world com obstáculos em células aleatórias. O agente deve visitar todas as células livres enquanto evita as paredes e obstáculos contidos no ambiente. Um episódio termina com sucesso quando a cobertura completa é atingida dentro do limite de passos, caso contrário o episódio é truncado.

Esse relatório é baseado nas implementações e logs nesse repositório, especialmente:

- `gymnasium_env/grid_world_cpp.py`
- `train_grid_world_cpp.py`
- `CNN.py`
- `goal_conditioned_policy.py`
- `continue_latest_stage_training.py`
- `log/recurrent_ppo_cpp_curriculum_20260505_124325/progress.csv`
- `log/recurrent_ppo_cpp_curriculum_20260505_124325_continued_20260506_124343/progress.csv`
- `data/recurrent_ppo_cpp_curriculum_20260505_124325_interrupt_latest_metadata.json`
- `data/recurrent_ppo_cpp_curriculum_20260507_085117_interrupt_latest_metadata.json`

## 2. Estrutura do Ambiente

O ambiente é um 2D grid-world compativel com a biblioteca Gymnasium. Em cada step o agente escolhe uma de quatro ações: mover para cima, baixo, esquerda ou direita. Obstáculos são colocados em células aleatórias, porém cada layout gerado é verificado com um algoritmo de BFS flood-fill para garantir que a partir do estado inicial o ambiente é completamente explorável, ou seja, o agente não começa preso por obstáculos e não existem celulas completamente cercadas por eles. Isso evita cenários onde a cobertura completa é impossível, a viabilizando como um alvo de aprendizado.

A observação consiste em um dicionário com duas partes:

| Chave | Descrição | Papel no RL |
| --- | --- | --- |
| `agent` | Posição X normalizada, posição Y normalizada e porcentagem coberta atualmente | Provê informações globais de progresso compactas |
| `neighbors` | Um mapa local 3x3 com 3 canais centralizado no agente: parede/obstáculo, célula visitada, célula não visitada | Entrega informação local do espaço ao redor do agente|

Essa observação categoriza o problema como parcialmente observável, ou seja, o agente não recebe o mapa inteiro, apenas uma visão local e seu progresso atual. Por isso uma política baseada em memória é útil nesse cenário, pois o agente deve ser capaz de inferir aonde ele já esteve e como navegar por corredores já visitados para alcançar as células não exploradas restantes.

A função de reward é uma combinação de sinais de curto e longo prazo:

The reward function combines sparse and dense signals:

| Componente da função | Propósito |
| --- | --- |
| Recompensa por explorar células novas, normalizada pelo número de células livres | Encoraja a cobertura e mantém a proporção de recompensas em tamanhos diferentes de grid |
| Penalização por colisão ou revisitar células | Desencoraja movimentos inválidos e ciclos de ações |
| Penalidade de tempo e ineficiência | Favorece caminhos de cobertura menores |
| Reward de distancia para fronteira  | Recompensa movimento por células já exploradas quando se aproxima de células não visitadas |
| Bônus de progresso em 80%, 90%, 95% e 98% de cobertura | Reduz recompensas esparsas em estados de cobertura quase completa |
| Grande recompensa por cobertura completa  | Torna a exploração de todas as células livres o objetivo principal do agente. |
| Penalidade de truncamento com crédito por cobertura parcial | Penaliza o agente enquanto distingue coberturas totais de quase completas |

## 3. Estratégia Escolhida

A estratégia escolhida é o PPO Recorrente treinado por currículo, com extração de características personalizada, normalização de recompensas, exploração adaptativa e um mecanismo de retropropagação contínua.

### 3.1 Algoritmo: PPO Recorrente

A implementação utiliza `RecurrentPPO` da `sb3-contrib`. PPO é apropriado para o problema por conta da sua estabilidade para problemas de gradiente de política e por limitar atualizações destrutivas da política com clipping. O fato da variante Recorrente ser utilizada é especialmente importante pois a observação do ambiente é local. A LSTM carrega informações de estados anteriores, ajudando o agente a lembrar de regiões já exploradas e evitar com que ele se perca ou revisite essas regiões constantemente.

Configurações do PPO:

| Hyperparâmetro | Valor |
| --- | ---: |
| Gamma | 0.995 |
| Rollout por worker `n_steps` | 512 |
| Workers | 10 |
| Rollout efetivo | 5.120 |
| Batch size | 128 |
| Epochs por atualização | 8 |
| GAE lambda | 0.97 |
| Clip range | 0.08 |
| Target KL | 0.02 |
| Learning rate | 7e-5 |
| Learning rate de adaptação | 3e-5 |
| Coeficiente de entropia inicial | 0.015 |

O valor alto de gamma foi escolhido devido ao fato de que as recompensas principais ocorrem tardiamente no episódio, especialmente em grids maiores, onde mais passos são necessários para sua exploração total.

### 3.2 Feature Extractor

O `CPPFeatureExtractor` processa o mapa local 3x3 com uma CNN pequena e processa o estado atual do agente com um MLP. As duas representações são fundidas em um vetor de características de 256 dimensões. Essa escolha de design corresponde à estrutura da observação:

- O mapa local é espacial, logo convoluções são apropriadas.
- A posição e a taxa de cobertura são features escalares de baixa dimensão, logo um MLP é suficiente.
- A política recorrente lida com informação temporal além da observação atual.

### 3.3 Cabeçalho de Política Condicionada ao Objetivo

`GoalConditionedMultiInputLstmPolicy` adiciona um objetivo oculto para o extrator MLP do ator/crítico . O objetivo é concatenado com as características da LSTM antes das redes de política e valor. Na prática, isso proporciona à política uma representação de alvo interno adicional, que é útil em tarefas como um objetivo final complexo e não trivial como cobrir uma área completamente.

### 3.4 Aprendizado por Currículo

O currículo aumenta, gradualmente, o tamanho da área em conjunto com o número de obstáculos e duração máxima de cada episódio:

| Estágio | Área | Obstáculos | Max steps | Desempenho mínimo para promoção |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 5x5 | 3 | 200 | 95% de cobertura completa |
| 2 | 7x7 | 6 | 280 | 95% de cobertura completa |
| 3 | 9x9 | 10 | 360 | 94% de cobertura completa |
| 4 | 11x11 | 15 | 440 | 93% de cobertura completa |
| 5 | 13x13 | 20 | 520 | 92% de cobertura completa |
| 6 | 15x15 | 27 | 600 | 91% de cobertura completa |
| 7 | 17x17 | 35 | 680 | 90% de cobertura completa |
| 8 | 19x19 | 43 | 760 | 90% de cobertura completa |
| 9 | 20x20 | 48 | 800 | Duração Fixa |

Para promoção o agente deve atingir o desempenho mínimo em três avaliações consecutivas após a duração mínima de cada estágio. Cada avaliação consiste em 100 episódios e compara o desempenho da política estocástica com a determinística. Esse requisito serve para verificar que o desempenho do agente é consistente antes que a progressão aconteça.

O currículo se justifica pois explorar grids maiores é muito mais difícil que menores. Começar com grandes áreas criaria um problema de sinais de sucesso muito atrasados e esparsos. Em grids menores o agente aprende comportamentos aplicáveis a grids de qualquer tamanho, como evitar obstáculos e descobrir fronteiras não exploradas, que se mantém em estágios posteriores. 

### 3.5 Retropropagação contínua e aferição de plasticidade

O código utiliza um mecanismo de retropropagação. Neurônios maduros de baixa utilidade são repostos aos poucos. Essa medida é tomada para reduzir a perda de plasticidade ao longo do treinamento. onde a política após os estágios iniciais perde a capacidade de se adaptar nos estágios finais mais difíceis.

Durante o treinamento também são medidas e gravadas estatísticas de diagnóstico como weight norms, low-weight-unit ratio, and effective rank. Essas medidas são úteis pois o agente pode falhar não só por falta de dados, mas também por se tornar incapaz de adaptar a política aprendida.

### 3.6 Parallel Rollout Collection and CPU/GPU Delegation

Boa parte do esforço desse projeto se deu entorno de agilizar o processo de treinamentos para tornar o currículo viável dentro do prazo estabelecido. Fez-se uso de vários workers para o ambiente, de forma que todas as ações e lógicas do ambiente são delegadas a CPU, enquanto a computação da rede neural é feita com o PyTorch na GPU por meio de Cuda.

Durante o treinamento, `make_parallel_cpp_env` cria várias instâncias independentes do ambiente. Por padrão usa-se `CppSubprocVecEnv`, que cria um subprocesso python por ambiente. Tais subprocessos possuem o ambiente, layout do grid, processam as funções de `reset` e `step`, calculam recompensas, e retornam a observação, recompensa, flags e informações do episódio para o processo principal de treinamento. Isso efetivamente mantém a simulação do ambiente, verificação de estados impossíveis, construção do estado de observação e recompensa em processos nos workers da CPU.

O processo principal é responsável pelo modelo `RecurrentPPO`. Uma vez que a batch de observações processadas na CPU são entregues, o modelo executa as passagens recorrentes da política e do valor no dispositivo PyTorch configurado. Nos arquivos de treino `DEVICE` é por padrão configurado como `cuda`,  assim a LSTM, CNN feature extractor, redes de política e valor, computação da loss function, retropropagação e atualização de pesos do otimizador rodam na GPU quando Cuda está disponível. Caso contrário `CPP_DEVICE` e setado para `cpu` e todas as operações são executadas pelo processador.

Isso cria uma divisão clara de papéis:

| Componente | Onde executa | Papel |
| --- | --- | --- |
| Lógica de Recompensa, reset e step | CPU workers | Geram cenários em paralelo |
| Agrupamento de observações e coordenação com PPO | Processo principal na CPU | Coleta rollouts e gerencia o treinamento |
| Passagens recorrentes CNN/LSTM/policy/value | PyTorch device, geralmente a GPU | Escolher ações e estimar valores |
| PPO loss, gradientes, e atualizações do otimizador| PyTorch device, geralmente a GPU | Atualizar pesos da rede neural |
| Episódios de avaliação | CPU env workers plus model inference device | Medir performance estocástica e determinística do agente |

Esse design é vital pois o treinamento gasta tempo tanto na simulação quanto computação da rede neural. CPU workers em paralelo reduzem o tempo necessário para agrupamento de todos os rollouts, enquanto a GPU reduz o tempo de processo para política recorrente e atualizações do PPO. O arquivo `CppSubprocVecEnv` também evita a importação das bibliotecas Stable-Baselines/PyTorch dentro de todo worker, mantendo seu papel focado apenas em execução do ambiente.

A divisão em múltiplos arquivos também visa resolver um problema recorrente de limite de memória paginável do Windows. No Windows cada processo utiliza `spawn`, como consequência cada worker inicializa um interpretador python separado ao invés de herdar o processo `fork`. Versões iniciais mantinham cada worker ligado diretamente ao script de treino principal, importando bibliotecas pesadas como PyTorch, Stable-Baselines, código de política PPO recorrente, utilitários de logging e auxiliares de modelo. Com muitos workers, os imports duplicados acabavam exaurindo a RAM e o espaço disponível para paginação no Disco C:, resultando em crashes no meio do processo de treino.

Assim os códigos para workers foram divididos em módulos menores e especializados:

| Arquivo | Propósito |
| --- | --- |
| `cpp_subproc_worker.py` | Loop mínimo, responsável apenas por executar o ambiente e comandos IPC |
| `cpp_subproc_vec_env.py` | Controlador do vetor de ambientes no processo principal, que se comunica com os workers |
| `cpp_env_factory.py` | Helper leve para registro e criação de ambientes|

Essa separação mantém os processos abertos por workers leves, importando apenas dependências necessárias para construir e dar step no ambiente, enquanto as bibliotecas mais pesadas são importadas apenas uma vez no processo principal de treinamento. A redução de importações duplicadas foi crucial para prevenir que o Windows esgotasse a memória virtual (arquivo de paginação) em treinamentos longos com muitos subprocessos paralelos.

This organization keeps the CPU worker processes lean: they import only what is needed to build and step the Gymnasium environment, while the expensive RL/model stack remains in the main process. That reduction in duplicated imports was important for preventing Windows from running out of page memory during long runs with several parallel environments.

## 4. Resultados

O melhor agente no repositório é:

- Treino principal: `recurrent_ppo_cpp_curriculum_20260505_124325`
- Continuação do treinamento: `recurrent_ppo_cpp_curriculum_20260505_124325_continued_20260506_124343`

### 4.1 Avaliação dos resultados do melhor agente

| Estágio | Grid | Obstáculos | Alvo | Desempenho estocástico ao ser promovido | Desempenho determinístico ao ser promovido | Quantidade de steps do estágio | Status da promoção |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 5x5 | 3 | 95% | 100% | 93% | 332,800 | Aprovado |
| 2 | 7x7 | 6 | 95% | 99% | 84% | 409,600 | Aprovado |
| 3 | 9x9 | 10 | 94% | 99% | 76% | 2,688,000 | Aprovado |
| 4 | 11x11 | 15 | 93% | 98% | 58% | 11,955,200 | Reprovado |

Por mais que o agente tenha atingido o estágio 4 e obtido uma alta taxa de cobertura estocástica, ele não foi aprovado em todos os requerimentos para promoção, a última avaliação do treinamento continuado foi de:

| Métrica | Valor |
| --- | ---: |
| Estágio | 4 |
| Grid | 11x11 |
| Steps no estágio | 12,774,400 |
| Desempenho estocástico | 90% |
| Desempenho determinístico | 53% |
| Recompensa média estocástico| 5.01 |
| Recompensa média determinístico | 4.29 |
| Tamanho médio de episódio estocástico | 264.8 |
| Tamanho médio de episódio determinístico| 304.0 |
| Tamanho médio máximo para promoção | 217.8 |
| Aprovado | Não |

Os metadados da interrupção de `data/recurrent_ppo_cpp_curriculum_20260505_124325_interrupt_latest_metadata.json` mostram um padrão similar: taxa de cobertura estocástica era 90%, determinística 49%, tamanho médio estocástico 249,32. Isso indica que o agente consistentemente atingia a performance necessária para ser promovido, porém não com a eficiência necessária e com modo determinístico inconsistente demais.

### 4.2 Agentes Salvos

Esse repositório contém o estado dos agentes treinados salvos em cada checkpoint: 

| Checkpoint | Status |
| --- | --- |
| `data/recurrent_ppo_cpp_curriculum_20260505_124325_stage1_5x5.zip` | Passou do estágio 1 |
| `data/recurrent_ppo_cpp_curriculum_20260505_124325_stage2_7x7.zip` | Passou do estágio 2 |
| `data/recurrent_ppo_cpp_curriculum_20260505_124325_stage3_9x9.zip` | Passou do estágio 3 |
| `data/recurrent_ppo_cpp_curriculum_20260505_124325_interrupt_latest.zip` | Interrompido no estágio 4 |

## 5. Análise

Os resultados apontam que a estratégia produz agentes com desempenho forte em CPP para grids pequenos e médios. Com aprovações nos estágios 1 a 3, com cobertura estocástica de 100%, 99% e 99%. Isso sugere que a combinação de memória recorrente, features de convolução local, função de reward e transferência de aprendizado ao longo do currículo é efetiva até grids 9x9.

A dificuldade principal aparece no estágio 4. No grid 11x11 com 15 obstáculos o agente consegue, as vezes, obter uma performance quase perfeita quando age estocásticamente, com melhor a taxa de cobertura avaliada sendo 98%. Porém, a política determinística é muito fraca, com melhor performance de 58% e 53% na ultima avaliação registrada. Essa diferença indica que a política aprendida pelo agente depende fortemente em explorar e ter algum nível de sorte do que uma estratégia de cobertura eficiente. Na prática em problemas de CPP, prefere-se agentes com baixa variância ou política deterministica forte, pois a consistência de desempenho é o que realmente importa.

A barreira de qualidade também expôe uma limitação de eficiência. O estágio 4 tinha uma duração média de 264.8 steps, enquanto o máximo para promoção era 217.8. Logo o agente completa a tarefa com muito backtracking. Esse comportamento é consistente com a observação local, sem um mapa global o agente pode revisitar muitos corredores explorados antes de encontrar as últimas regiões e células não exploradas.

Apesar disso o treinamento por currículo ainda é justificado. O reward médio não colapsou quando saindo de 5x5 para 7x7 e mais tarde 9x9, o comportamento do agente era bom o suficiente para ter alta taxa de sucesso em grids 11x11. O principal problema do agente era a incapacidade de desenvolver uma política eficiente e consistente apenas com observação parcial. 

## 6. Limitações

As maiores limitações da estratégia implementada são:

| Limitação | Efeito |
| --- | --- |
| Observação local | A falta de uma memória global do mapa, impacta a capacidade do agente de encontrar regiões não visitadas perdidas no grid |
| Diferença de desempenho entre modo estocástico de determinístico | A política não se torna consistente apesar de bom comportamento observado |
| Eficiência | Agente atinge taxa de sucesso necessária, porém com muitos steps |
| Currículo longo| Estágios finais necessitam de muitos steps e são muito sensíveis a hyperparâmetros |
| Complexidade da função de reward | Muitos mecanismos ajudam o aprendizado, porém isolar quais causam uma mudança de comportamento específica provou-se difícil |

## 7. Possíveis melhorias

| Melhoria | Benefício esperado |
| --- | --- |
| Adicionar uma observação egocêntrica maior ou um mapa global de visitados | Melhorar a percepção global e reduzir a observabilidade parcial, resultando em um planejamento de cobertura mais eficiente |
| Usar um alvo de fronteira explícito ou uma política hierárquica | Melhorar o planejamento de longo horizonte e a eficiência da navegação |
| Destilar o comportamento estocástico em uma política determinística | Reduzir a variância e melhorar a consistência entre treinamento e avaliação |
| Adicionar perdas auxiliares de predição | Melhorar a qualidade das representações ao incentivar o modelo a codificar a estrutura do mapa e o estado de cobertura |
| Ajustar a recompensa de eficiência do estágio 4 e o limite de comprimento | Incentivar trajetórias completas mais curtas e eficientes |
| Comparar com heurísticas clássicas de CPP | Permitir uma avaliação comparativa mais confiável e consistente |
| Executar ablações | Identificar a contribuição de cada componente para o desempenho geral |

## 8. Conclusão

A estratégia selecionada e implementada é ideal para o ambiente, devido a sua natureza de observação parcial, sequencial e sucessos esparsos entre si. O agente aprende como cobrir todo espaço com sucesso porém não de forma eficiente. O principal problema a se resolver é justamente como converter uma boa política estocástica ineficiente em uma eficiente tanto quando age estocásticamente quanto quando age deterministicamente.

Desempenho final:

| Métrica | 5x5 | 10x10 |
| --- | --- | --- |
| Full Coverage Rate | 100.0% | 93.3% |
| Mean Steps | 45 +/- 15.72 | 210.38 +/- 73.99 |
| Mean Reward | 5.80 +/- 2.12| 5.34 +/- 2.13|
| Mean Coverage | 100.0% | 99.9% |
| Min/Max Coverage| 100.0%/100.0% | 88.6% / 100.0%|