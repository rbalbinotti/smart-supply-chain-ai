# Projeto Sistema Inteligente para Gestão de Compras e Estoque com IA
## Autor: **Roberto Rosário Balbinotti**  
Contato e Licença:  
[![Email](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](https://mailto:rbalbinotti@gmail.com)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/roberto-balbinotti)  
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

## 1. Concepção e Planejamento

### 1.1. Definição do Problema de Negócio (Business Problem Definition)

A gestão atual de estoques e compras em nosso negócio apresenta ineficiências significativas, resultando em custos operacionais elevados e perda de oportunidades de receita. O problema se manifesta em:

- **Excesso de estoque e desperdício:** Produtos parados e itens com validade vencida, o que gera custos de manutenção e capital parado.
- **Falta de produtos (ruptura):** Gerando perda de vendas e insatisfação do cliente.
- **Ineficiência na logística:** Impactando a velocidade e o giro do estoque.

O objetivo deste projeto é resolver essa dor de negócio, utilizando um **Sistema Inteligente de Otimização de Estoque e Compras com Visão Computacional** para monitorar e prever a demanda em tempo real.

O valor potencial que a solução de Data Science pode agregar é a otimização de toda a cadeia de suprimentos, resultando em:

- **Maximização dos resultados financeiros:** Reduzindo custos de armazenamento e desperdício.
- **Aumento da liquidez:** O capital liberado do estoque parado pode ser reinvestido em áreas estratégicas da empresa.
- **Melhora na satisfação do cliente:** Garantindo que os produtos desejados estejam sempre disponíveis.

### 1.2. Definição de Objetivos Claros e Mensuráveis (SMART)

O objetivo principal é aprimorar a gestão de estoques e o processo de compras através da automação e análise preditiva. Os objetivos SMART são:

- **Específico:** Otimizar a gestão de estoque e o processo de compras, utilizando visão computacional para monitorar e prever a demanda, reduzindo custos e maximizando a eficiência operacional.
- **Mensurável:** A solução deverá reduzir o **excesso de estoque em 20%** e diminuir o **índice de ruptura em 15%** no primeiro semestre após a implementação. Além disso, busca-se aumentar a **precisão das previsões de compras em 10%.**
- **Atingível:** Com a aplicação de modelos de aprendizado de máquina e visão computacional, é viável criar um sistema que colete dados em tempo real e forneça recomendações assertivas para compras e reabastecimento.
- **Relevante:** A otimização do estoque e a melhoria na precisão das compras têm um impacto direto e positivo nas finanças da empresa, aumentando a liquidez e a lucratividade, além de melhorar a satisfação do cliente.
- **Temporizável:** O projeto tem como meta alcançar os resultados mensuráveis no prazo de **seis meses** após a sua completa implementação.

> **Observação:** Os percentuais acima são exemplos utilizados para demonstrar o entendimento de como definir objetivos mensuráveis. Em um projeto real, esses valores seriam definidos com base em uma análise aprofundada dos dados históricos e em alinhamento com as metas estratégicas da empresa.

---

## 2. Estrutura da Solução e Módulos

### 2.1. Módulos do Sistema de Otimização

Para alcançar os objetivos do projeto, a solução será desenvolvida em módulos interconectados, cada um com uma função específica na otimização de estoque e compras.

#### 2.1.1. Módulo de Visão Computacional

- **Função:** Este módulo será responsável por analisar as imagens do estoque em tempo real para identificar e contar produtos nas prateleiras. Ele utilizará modelos de **detecção de objetos (YOLO, SSD)** para localizar cada item, extraindo informações cruciais sobre a quantidade disponível.
- **Saída:** A principal saída deste módulo será a informação sobre a contagem de estoque atual, que será usada como entrada para o próximo módulo.

#### 2.1.2. Módulo de Previsão de Demanda

- **Função:** Utilizando dados históricos de vendas e variáveis de engenharia (como sazonalidade e feriados), este módulo irá prever a demanda futura por produto. Ele poderá usar modelos de **séries temporais (ARIMA, Prophet)** ou modelos avançados de aprendizado de máquina como **LightGBM** para gerar previsões precisas.
- **Saída:** A saída será uma projeção da quantidade de cada produto que será vendida nos próximos dias ou semanas.

#### 2.1.3. Módulo de Recomendação de Compras

- **Função:** Este módulo funciona como o "cérebro" do sistema. Ele combinará a contagem de estoque atual (Módulo de Visão Computacional) e a previsão de demanda (Módulo de Previsão de Demanda). Com base nesses dados, ele calculará a quantidade ideal de produtos a serem comprados para evitar a ruptura e o excesso de estoque.
- **Saída:** A saída será uma lista de recomendações inteligentes para a equipe de compras, indicando quais produtos e em qual quantidade devem ser adquiridos.

#### 2.1.4. Módulo de Monitoramento e Dashboard

- **Função:** Este módulo será a interface do sistema, oferecendo um dashboard intuitivo para os usuários. Ele exibirá em tempo real as métricas de sucesso (KPIs) do projeto, como o índice de ruptura, a precisão das previsões e o custo com excesso de estoque.
- **Saída:** A saída é uma visualização clara do desempenho em tempo real, permitindo que os líderes de negócio e gerentes acompanhem os resultados e tomem decisões informadas.

### 2.2. Identificação de Métricas de Sucesso

As métricas serão divididas em métricas de negócio e métricas técnicas para garantir que o projeto entregue valor.

#### Métricas de Negócio

- **Redução de Custos:** Diminuição dos custos com excesso de estoque e logística de armazenamento.
- **Redução do Índice de Ruptura:** Diminuição da porcentagem de itens em falta.
- **Retorno sobre o Investimento (ROI):** Avaliar o impacto financeiro da solução.
- **Aumento na Liquidez:** Monitorar o capital liberado que antes estava parado em estoque.

#### Métricas Técnicas

- **Acurácia da Previsão de Demanda:** Medir a precisão com que o modelo prevê a quantidade de produtos a serem vendidos.
- **Precisão e Recall da Visão Computacional:** Avaliar a capacidade do sistema de identificar e contar corretamente os produtos no estoque.
- **Latência do Sistema:** Medir o tempo que o sistema leva para processar as imagens e fornecer as informações.
- **Erro Quadrático Médio (RMSE) ou Erro Médio Absoluto (MAE):** Avaliar o quão distante as previsões do modelo estão dos valores reais.

### 2.3. Identificação de Stakeholders e Expectativas

A comunicação e o alinhamento com as partes interessadas são cruciais para o sucesso do projeto.

- **Líderes de Negócio (CEO, Diretores Financeiros):**
    - **Expectativa:** Redução de custos, aumento da lucratividade e um ROI positivo.
    - **Canais de Comunicação:** Reuniões de acompanhamento mensais para apresentar o progresso e os resultados financeiros.
- **Gerentes de Estoque e Compras:**
    - **Expectativa:** Uma ferramenta que automatize o monitoramento e forneça recomendações inteligentes, facilitando a tomada de decisões e reduzindo o trabalho manual.
    - **Canais de Comunicação:** Reuniões semanais e acesso a dashboards em tempo real para feedback contínuo.
- **Equipe Operacional e Logística:**
    - **Expectativa:** Um sistema que simplifique a contagem de estoque, otimize o espaço de armazenamento e agilize a movimentação de produtos.
    - **Canais de Comunicação:** Treinamento prático e sessões de feedback para garantir que a ferramenta seja intuitiva e útil no dia a dia.
- **Equipe de TI:**
    - **Expectativa:** Uma solução robusta, escalável e de fácil manutenção, que se integre bem com os sistemas existentes da empresa.
    - **Canais de Comunicação:** Documentação técnica, reuniões de alinhamento e suporte durante a implementação.

---

## 3. Coleta e Entendimento dos Dados (Data Collection and Understanding)

### 3.1. Fontes de Dados e Repositórios

Para simular um ambiente de produção, serão utilizadas múltiplas fontes de dados, tanto de repositórios públicos quanto de conjuntos de dados sintéticos, se necessário.

**Datasets de Imagens (Visão Computacional)**
- **Roboflow Universe:** Vasta coleção de datasets públicos com anotações prontas para tarefas de detecção de objetos.
- **Kaggle:** Um dos maiores repositórios de datasets do mundo, com uma seção específica para visão computacional.
- **ImageNet:** Um dos maiores e mais famosos datasets do mundo, com milhões de imagens categorizadas.
- **MS COCO (Common Objects in Context):** Dataset amplamente usado para detecção de objetos, segmentação e legendagem.

**Datasets Tabulares (Vendas, Estoque e Compras)**
- **Kaggle:** Diversos datasets de varejo, vendas e e-commerce (ex.: "Online Retail Dataset" ou "Brazilian E-commerce Public Dataset by Olist").
- **UCI Machine Learning Repository:** Oferece datasets de vendas e comportamento do consumidor que podem ser adaptados.
- **Data.gov:** Portal de dados abertos do governo dos EUA com datasets sobre vendas e varejo.

**Dados Sintéticos**
Para complementar as informações e criar um cenário de negócio realista, serão gerados datasets sintéticos de vendas e estoque. Esta abordagem permitirá preencher lacunas em dados públicos e simular um histórico de vendas com padrões específicos (sazonalidade, picos).

### 3.2. Exploração e Análise dos Dados

- **Estrutura dos Dados:** A análise focará em entender a estrutura dos dados sintéticos e públicos, garantindo a integração entre imagens (dados não estruturados) e dados tabulares (vendas e estoque).
- **Qualidade e Tratamento:** Serão realizadas análises para identificar possíveis inconsistências e dados ausentes, garantindo que o modelo seja treinado com informações confiáveis.
- **Insights:** A análise exploratória buscará padrões de sazonalidade e tendências de vendas que possam ser utilizados para informar o modelo de previsão de demanda.

---

## 4. Preparação e Engenharia de Dados (Data Preparation and Feature Engineering)

Nesta fase, o foco é transformar os dados brutos em um formato ideal para o treinamento dos modelos.

### 4.1. Preparação dos Dados (Data Preparation)

As tarefas se concentram na limpeza e no pré-processamento dos dados para garantir sua qualidade e consistência.

- **Tratamento de Dados Ausentes:** Identificar e gerenciar dados ausentes no dataset de vendas e estoque, com estratégias como a imputação de valores médios ou a remoção de registros.
- **Tratamento de Dados Ruidosos e Outliers:** Detectar e analisar valores atípicos que possam distorcer o treinamento do modelo.
- **Normalização/Padronização:** Escalar os dados numéricos para um intervalo padrão, o que é essencial para o bom desempenho de muitos algoritmos de Machine Learning.
- **Pré-processamento de Imagens:** As imagens de produtos serão redimensionadas para um tamanho uniforme, normalizadas e, se necessário, passarão por operações de aumento de dados (**data augmentation**), como rotação e zoom.

### 4.2. Engenharia de Variáveis (Feature Engineering)

A engenharia de variáveis se dedica a criar novas variáveis a partir dos dados existentes, a fim de melhorar a capacidade preditiva dos modelos.

- **Variáveis Temporais:** Criar novas variáveis a partir do histórico de vendas (dia da semana, mês, feriados) para capturar padrões sazonais e de recorrência.
- **Variáveis de Lag:** Gerar variáveis que representem o histórico de vendas de um produto em períodos anteriores (vendas na semana ou mês anterior), cruciais para modelos de séries temporais.
- **Variáveis de Análise de Imagem:** A partir das imagens, extrair características importantes que possam ser usadas no modelo de previsão de demanda.

> **Observação:** A execução exata de cada etapa dependerá das características específicas e da qualidade final dos datasets públicos e sintéticos. A documentação acima representa o plano de trabalho ideal, que será ajustado conforme a análise exploratória dos dados.

---

## 5. Modelagem e Análise (Modeling and Analysis)

Nesta fase, o foco é construir e treinar os modelos de Machine Learning e Visão Computacional que irão compor o Sistema Inteligente.

### 5.1. Seleção e Desenvolvimento de Modelos

O projeto utilizará uma abordagem combinada para extrair o máximo de valor dos dados.

- **Modelo de Visão Computacional:** Para a contagem e identificação de produtos, será desenvolvido um modelo de **detecção de objetos**. A escolha inicial é por uma arquitetura moderna e eficiente, como **YOLO** ou **SSD**, ideais para detecção em tempo real.
- **Modelo de Previsão de Demanda:** Para prever as vendas futuras, será utilizado um modelo de séries temporais. A escolha pode variar entre abordagens mais tradicionais como **ARIMA** ou **Prophet**, ou modelos mais avançados de aprendizado de máquina como **LightGBM** ou **XGBoost**.

### 5.2. Treinamento e Validação dos Modelos

- **Treinamento:** Os modelos serão treinados com os datasets preparados (imagens rotuladas para visão computacional; histórico de vendas e variáveis para previsão de demanda).
- **Validação Cruzada:** Serão aplicadas técnicas de validação cruzada para evitar o overfitting e garantir que o desempenho do modelo seja consistente.
- **Ajuste de Hiperparâmetros:** Serão realizados ajustes finos nos hiperparâmetros de cada modelo para otimizar seu desempenho e precisão.

### 5.3. Avaliação dos Resultados

Os resultados serão avaliados usando as métricas definidas no início do projeto.

- **Métricas Técnicas:** A performance do modelo de detecção de objetos será avaliada por meio de métricas como **mAP (mean Average Precision)**. Para o modelo de previsão de demanda, serão usadas métricas como **RMSE** e **MAE**.
- **Métricas de Negócio:** Os resultados dos modelos serão traduzidos em métricas de negócio para avaliar o impacto na redução de custos, diminuição da ruptura e aumento da liquidez.

> **Observação**: A seleção final dos modelos pode ser ajustada com base na análise exploratória dos dados e no desempenho inicial de cada abordagem. Por exemplo, se os dados de vendas apresentarem padrões complexos, a utilização de modelos de Deep Learning como **LSTMs** pode ser considerada.

---

## 6. Implantação e Monitoramento (Deployment and Monitoring)

Nesta fase, o modelo e todo o sistema são colocados em um ambiente de produção para que a solução possa ser utilizada.

### 6.1. Arquitetura da Solução e Implantação

- **Conteinerização:** O modelo de visão computacional, o de previsão de demanda e o código de processamento serão empacotados em contêineres **Docker**. Isso garante consistência e portabilidade da solução.
- **Implantação:** A solução será **arquitetada para um ambiente de nuvem** (AWS, GCP ou Azure), permitindo escalabilidade e alta disponibilidade.
- **Integração:** A API do sistema será desenvolvida para se integrar com câmeras ou com o sistema de gestão de estoque da empresa.

### 6.2. Monitoramento e Manutenção

- **Dashboard de Monitoramento:** Será criado um dashboard para monitorar o desempenho do sistema em tempo real, exibindo métricas-chave como a acurácia das previsões, a latência do sistema e o índice de ruptura de estoque.
- **Detecção de Desvio (Drift):** O modelo será monitorado para detectar a perda de performance ao longo do tempo ("model drift").
- **Retreinamento do Modelo:** Será estabelecida uma rotina para retreinar o modelo periodicamente, utilizando novos dados, garantindo que o sistema continue preciso e relevante.

> **Observação**: Para este projeto de portfólio, a fase de implantação em nuvem e a integração com sistemas externos são descritas como **etapas de um plano de produção**, mas a execução real será limitada a um **ambiente local e simulado**. O foco é demonstrar a arquitetura robusta e escalável da solução.