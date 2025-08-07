# Definições Projeto Compras com IA

## 1. Concepção e Planejamento

### 1.1. Definição do Problema de Negócio (Business Problem Definition)

A gestão atual de estoques e compras em nosso negócio apresenta ineficiências significativas, resultando em custos operacionais elevados e perda de oportunidades de receita.  

O problema se manifesta em:

- **Excesso de estoque e desperdício:**  Com produtos parados e itens com validade vencida, o que gera custos de manutenção e capital parado.  

- **Falta de produtos (ruptura):**  Gerando perda de vendas e insatisfação do cliente.  

- **Ineficiência na logística:**  Impactando a velocidade e o giro do estoque.  

O objetivo deste projeto é resolver essa dor de negócio, utilizando um Sistema Inteligente de Otimização de Estoque e Compras com Visão Computacional para monitorar e prever a demanda em tempo real.

O valor potencial que a solução de Data Science pode agregar é a **otimização de toda a cadeia de suprimentos**, resultando em:

- **Maximização dos resultados financeiros:**  Reduzindo custos de armazenamento e desperdício.

- **Aumento da liquidez:**  Com o capital liberado do estoque parado, é possível reinvestir em áreas estratégicas da empresa, como produção, tecnologia e capacitação da equipe.

- **Melhora na satisfação do cliente:**  Garantindo que os produtos desejados estejam sempre disponíveis.

### 1.2. Definição de Objetivos Claros e Mensuráveis (SMART)
O objetivo principal é aprimorar a gestão de estoques e o processo de compras através da automação e análise preditiva. Os objetivos SMART são:

- **Específico:**  Otimizar a gestão de estoque e o processo de compras, utilizando visão computacional para monitorar e prever a demanda, reduzindo custos e maximizando a eficiência operacional.  

- **Mensurável:**  A solução deverá reduzir o **excesso de estoque em 20%** e diminuir o **índice de ruptura em 15%** no primeiro semestre após a implementação. Além disso, busca-se aumentar a **precisão das previsões de compras em 10%.**

- **Atingível:**  Com a aplicação de modelos de aprendizado de máquina e visão computacional, é viável criar um sistema que colete dados em tempo real e forneça recomendações assertivas para compras e reabastecimento.

- **Relevante:**  A otimização do estoque e a melhoria na precisão das compras têm um impacto direto e positivo nas finanças da empresa, aumentando a liquidez e a lucratividade, além de melhorar a satisfação do cliente.

- **Temporizável:**  O projeto tem como meta alcançar os resultados mensuráveis no prazo de **seis meses** após a sua completa implementação.

> **Observação:** Os percentuais acima são exemplos utilizados para demonstrar o entendimento de como definir objetivos mensuráveis. Em um projeto real, esses valores seriam definidos com base em uma análise aprofundada dos dados históricos e em alinhamento com as metas estratégicas da empresa.

### 1.3. Identificação de Métricas de Sucesso
As métricas serão divididas em métricas de negócio e métricas técnicas para garantir que o projeto entregue valor.
#### Métricas de Negócio
- **Redução de Custos:**  Diminuição dos custos com excesso de estoque e logística de armazenamento.

- **Redução do Índice de Ruptura:**  Diminuição da porcentagem de itens em falta.

- **Retorno sobre o Investimento (ROI):**  Avaliar o impacto financeiro da solução, comparando os custos de implementação com os ganhos gerados pela otimização.

- **Aumento na Liquidez:**  Monitorar o capital liberado que antes estava parado em estoque.

#### Métricas Técnicas
- **Acurácia da Previsão de Demanda:**  Medir a precisão com que o modelo prevê a quantidade de produtos a serem vendidos.

- **Precisão e Recall da Visão Computacional:**  Avaliar a capacidade do sistema de identificar e contar corretamente os produtos no estoque.

- **Latência do Sistema:**  Medir o tempo que o sistema leva para processar as imagens e fornecer as informações sobre o estoque.

- **Erro Quadrático Médio (RMSE) ou Erro Médio Absoluto (MAE):**  Avaliar o quão distante as previsões do modelo estão dos valores reais.

### 1.4. Identificação de Stakeholders e Expectativas
A comunicação e o alinhamento com as partes interessadas são cruciais para o sucesso do projeto.
- **Líderes de Negócio (CEO, Diretores Financeiros):** 
    - **Expectativa:**  Redução de custos, aumento da lucratividade e um ROI positivo.
    - **Canais de Comunicação:**  Reuniões de acompanhamento mensais para apresentar o progresso e os resultados financeiros.
- **Gerentes de Estoque e Compras:** 
    - **Expectativa:**  Uma ferramenta que automatize o monitoramento e forneça recomendações inteligentes, facilitando a tomada de decisões e reduzindo o trabalho manual.
    - **Canais de Comunicação:**  Reuniões semanais e acesso a dashboards em tempo real para feedback contínuo.
- **Equipe Operacional e Logística:** 
    - **Expectativa:**  Um sistema que simplifique a contagem de estoque, otimize o espaço de armazenamento e agilize a movimentação de produtos.
    - **Canais de Comunicação:**  Treinamento prático e sessões de feedback para garantir que a ferramenta seja intuitiva e útil no dia a dia.
- **Equipe de TI:** 
    - **Expectativa:**  Uma solução robusta, escalável e de fácil manutenção, que se integre bem com os sistemas existentes da empresa.
    - **Canais de Comunicação:**  Documentação técnica, reuniões de alinhamento e suporte durante a implementação.

## 2. Coleta e Entendimento dos Dados (Data Collection and Understanding)
### 2.1. Fontes de Dados e Repositórios
Para simular um ambiente de produção, serão utilizadas múltiplas fontes de dados, tanto de repositórios públicos quanto, se necessário, de conjuntos de dados sintéticos.

**Datasets de Imagens (Visão Computacional)**
- **Roboflow Universe:** Uma vasta coleção de datasets públicos com anotações (caixas delimitadoras, segmentação) prontas para tarefas de detecção de objetos. É uma excelente fonte para o seu modelo de visão computacional.

- **Kaggle:** A plataforma é um dos maiores repositórios de datasets do mundo, com uma seção específica para visão computacional que inclui datasets de produtos, objetos e muito mais.

- **ImageNet:** Um dos maiores e mais famosos datasets do mundo, com milhões de imagens categorizadas em milhares de classes. É uma referência para modelos de classificação de imagens.

- **MS COCO (Common Objects in Context):** Um dataset robusto e amplamente usado para detecção de objetos, segmentação e legendagem.

**Datasets Tabulares (Vendas, Estoque e Compras)**
- **Kaggle:** Possui diversos datasets de varejo, vendas e e-commerce, como o "Online Retail Dataset" ou o "Brazilian E-commerce Public Dataset by Olist". Eles são ideais para treinar um modelo de previsão de demanda.

- **UCI Machine Learning Repository:** Embora seja uma fonte mais clássica, ainda oferece alguns datasets de vendas e comportamento do consumidor que podem ser adaptados.

- **Data.gov:** O portal de dados abertos do governo dos EUA oferece datasets sobre vendas e varejo, especialmente em nível municipal ou estadual. Outros governos também podem ter portais similares.

**Dados Sintéticos**  

Para complementar as informações e criar um cenário de negócio realista, serão gerados datasets sintéticos de vendas e estoque. Esta abordagem permitirá:

- Preencher lacunas em dados públicos.

- Simular um histórico de vendas com padrões específicos (sazonalidade, picos de vendas) que são essenciais para o modelo de previsão.

- Ter um dataset tabular que se integre perfeitamente com o dataset de imagens, criando uma base de dados completa para o projeto.

### 2.2. Exploração e Análise dos Dados:
- **Estrutura dos Dados:** A análise focará em entender a estrutura dos dados sintéticos e públicos, garantindo a integração entre as imagens (dados não estruturados) e os dados tabulares (vendas e estoque).

- **Qualidade e Tratamento:** Serão realizadas análises para identificar possíveis inconsistências e dados ausentes, especialmente no dataset sintético, para garantir que o modelo seja treinado com informações confiáveis.

- **Insights:** A análise exploratória buscará padrões de sazonalidade e tendências de vendas que possam ser utilizados para informar o modelo de previsão de demanda.


## 3. Preparação e Engenharia de Dados (Data Preparation and Feature Engineering)
Nesta fase, o foco é transformar os dados brutos — sejam eles de repositórios públicos ou sintéticos — em um formato ideal para o treinamento dos modelos.

### 3.1. Preparação dos Dados (Data Preparation)
Aqui, as tarefas se concentram na limpeza e no pré-processamento dos dados para garantir sua qualidade e consistência.
- **Tratamento de Dados Ausentes:** Identificar e gerenciar dados ausentes no dataset de vendas e estoque. As estratégias incluirão a imputação de valores médios ou a remoção de registros, dependendo da criticidade e do volume dos dados faltantes.

- **Tratamento de Dados Ruidosos e Outliers:** Detectar e analisar valores atípicos (outliers) que possam distorcer o treinamento do modelo. Para o dataset de vendas, isso pode envolver a análise de picos incomuns que não se encaixam em padrões sazonais.

- **Normalização/Padronização:** Escalar os dados numéricos para um intervalo padrão, o que é essencial para o bom desempenho de muitos algoritmos de Machine Learning.

- **Pré-processamento de Imagens:** As imagens de produtos serão redimensionadas para um tamanho uniforme, normalizadas (ex:** escala de cinza, ajuste de brilho) e, se necessário, passarão por operações de aumento de dados (data augmentation), como rotação e zoom, para aumentar a robustez do modelo de visão computacional.

### 3.2. Engenharia de Variáveis (Feature Engineering)
A engenharia de variáveis se dedica a criar novas variáveis a partir dos dados existentes, a fim de melhorar a capacidade preditiva dos modelos.

- **Variáveis Temporais:** Criar novas variáveis a partir do histórico de vendas, como o dia da semana, mês, trimestre e feriados, para capturar padrões sazonais e de recorrência.

- **Variáveis de Lag:** Gerar variáveis que representem o histórico de vendas de um produto em períodos anteriores (por exemplo, vendas na semana anterior ou no mês anterior), que são cruciais para modelos de séries temporais.

- **Variáveis de Análise de Imagem:** A partir das imagens, extrair características importantes que possam ser usadas como variáveis no modelo de previsão de demanda. Isso pode envolver a extração de características como cor dominante, textura ou até mesmo a contagem de produtos na prateleira, que serão usadas em conjunto com os dados tabulares.

> **Observação:** A execução exata de cada etapa de preparação e engenharia de variáveis dependerá das características específicas e da qualidade final dos datasets públicos e sintéticos. A documentação acima representa o plano de trabalho ideal, que será ajustado conforme a análise exploratória dos dados.

## 4. Modelagem e Análise (Modeling and Analysis)
Nesta fase, o foco é construir e treinar os modelos de Machine Learning e Visão Computacional que irão compor o Sistema Inteligente.

### 4.1. Seleção e Desenvolvimento de Modelos
O projeto utilizará uma abordagem combinada para extrair o máximo de valor dos dados.

- **Modelo de Visão Computacional:** Para a contagem e identificação de produtos, será desenvolvido um modelo de **detecção de objetos**. A escolha inicial é por uma arquitetura moderna e eficiente, como **YOLO (You Only Look Once)** ou **SSD (Single Shot MultiBox Detector)**, que são ideais para detecção em tempo real. O modelo será treinado para identificar e contar as unidades de cada produto a partir das imagens do estoque.

- **Modelo de Previsão de Demanda:** Para prever as vendas futuras e otimizar as compras, será utilizado um modelo de séries temporais. A escolha pode variar entre abordagens mais tradicionais como **ARIMA** ou **Prophet**, ou modelos mais avançados de aprendizado de máquina como **LightGBM** ou **XGBoost**, que são excelentes para dados tabulares e podem incorporar as variáveis de engenharia criadas na fase anterior.

### 4.2. Treinamento e Validação dos Modelos

- **Treinamento:** Os modelos serão treinados com os datasets preparados. O modelo de visão computacional usará as imagens rotuladas, enquanto o modelo de previsão de demanda usará o histórico de vendas, estoque e as variáveis criadas.

- **Validação Cruzada:** Para evitar o overfitting e garantir a generalização, serão aplicadas técnicas de validação cruzada. Isso garantirá que o desempenho do modelo seja consistente, mesmo com dados que ele nunca viu antes.

- **Ajuste de Hiperparâmetros:** Serão realizados ajustes finos nos hiperparâmetros de cada modelo para otimizar seu desempenho e precisão.

### 4.3. Avaliação dos Resultados
Os resultados serão avaliados usando as métricas definidas no início do projeto.

- **Métricas Técnicas:** A performance do modelo de detecção de objetos será avaliada por meio de métricas como **mAP (mean Average Precision)**. Para o modelo de previsão de demanda, serão usadas métricas como **Erro Quadrático Médio (RMSE)** e **Erro Médio Absoluto (MAE)**.

- **Métricas de Negócio:** Os resultados dos modelos serão traduzidos em métricas de negócio para avaliar o impacto na redução de custos, diminuição da ruptura e aumento da liquidez. Por exemplo, a precisão do modelo de previsão será usada para calcular a economia potencial na otimização de compras.

> **Observação**: A seleção final dos modelos pode ser ajustada com base na análise exploratória dos dados e no desempenho inicial de cada abordagem. Por exemplo, se os dados de vendas apresentarem padrões complexos não lineares, a utilização de modelos de Deep Learning como **LSTMs (Long Short-Term Memory)** pode ser considerada. Esta documentação reflete o plano inicial, que será iterado para encontrar a solução mais eficaz.

## 5. Implantação e Monitoramento (Deployment and Monitoring)
Nesta fase, o modelo e todo o sistema são colocados em um ambiente de produção para que a solução possa ser utilizada pelo negócio.

### 5.1. Arquitetura da Solução e Implantação

- **Conteinerização:** O modelo de visão computacional, o modelo de previsão de demanda e o código de processamento serão empacotados em contêineres Docker. Isso garante que a solução seja consistente e portável, facilitando a implantação em qualquer ambiente.

- **Implantação:** A solução será **arquitetada para um ambiente de nuvem**, como AWS, GCP ou Azure. Isso permite escalabilidade e alta disponibilidade, garantindo que o sistema possa lidar com o volume de dados e solicitações em tempo real.

- **Integração:** A API do sistema será desenvolvida para se integrar com as câmeras ou com o sistema de gestão de estoque da empresa. Isso fará com que as informações geradas (contagem de produtos, previsões de vendas) possam ser consumidas diretamente pelas equipes de logística e compras.

### 5.2. Monitoramento e Manutenção

- **Dashboard de Monitoramento:** Será criado um dashboard para monitorar o desempenho do sistema em tempo real. Ele exibirá métricas-chave como a acurácia das previsões, a latência do sistema e o índice de ruptura de estoque. Isso permitirá que os stakeholders visualizem o valor gerado e identifiquem rapidamente qualquer anomalia.

- **Detecção de Desvio (Drift):** O modelo será monitorado para detectar o "model drift" — a perda de performance ao longo do tempo. Isso pode acontecer, por exemplo, se o comportamento do cliente mudar ou se novos produtos forem adicionados ao estoque.

- **Retreinamento do Modelo:** Com base no monitoramento, será estabelecida uma rotina para retreinar o modelo periodicamente, utilizando novos dados. Isso garante que o sistema continue a ser preciso e relevante, mantendo a sua capacidade de otimização ao longo da vida útil do projeto.


>**Observação**: Para este projeto de portfólio, a fase de implantação em nuvem e a integração com sistemas externos são descritas como **etapas de um plano de produção**, mas a execução real será limitada a um **ambiente local e simulado**. O foco é demonstrar a arquitetura robusta e escalável da solução, sem incorrer em custos de infraestrutura, mantendo o projeto acessível e focado na prova de conceito.