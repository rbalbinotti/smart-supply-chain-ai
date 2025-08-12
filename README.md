# Sistema Inteligente de Otimização de Estoque e Compras com Visão Computacional

## Autor: Roberto Rosário Balbinotti
Contato e Licença:  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/roberto-balbinotti)
[![Email](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](https://mailto:rbalbinotti@gmail.com)  
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)


## Descrição

Este projeto demonstra a aplicação de **Machine Learning (ML)** e **Visão Computacional (CV)** para transformar a gestão de estoque. A solução utiliza técnicas avançadas para otimizar o giro de mercadorias, minimizar desperdícios e evitar compras mal dimensionadas. O resultado é uma cadeia de suprimentos mais ágil, previsível e lucrativa.

## Objetivo
Maximizar a eficiência e os resultados financeiros de empresas através da otimização inteligente de estoque e do processo de compras. O foco é garantir um estoque enxuto e seguro, melhorando a liquidez e a saúde financeira da organização.

Aqui você pode encontrar a [proposta detalhada do projeto](docs/propose_project.md).



## Módulos da Solução
A solução foi arquitetada em módulos interconectados, cada um com uma função específica para a otimização de estoque e compras:

* **Módulo de Visão Computacional:** Responsável por analisar imagens do estoque em tempo real para identificar e contar produtos nas prateleiras.
* **Módulo de Previsão de Demanda:** Utiliza dados históricos de vendas para prever a demanda futura por produto.
* **Módulo de Recomendação de Compras:** Combina a contagem de estoque atual e a previsão de demanda para calcular a quantidade ideal de produtos a serem comprados.
* **Módulo de Monitoramento e Dashboard:** Exibe um dashboard intuitivo com métricas-chave (KPIs) do projeto em tempo real.



## Tecnologias Utilizadas

A solução é construída com as seguintes tecnologias:

* **Linguagem de Programação:** Python
* **Visão Computacional:** YOLO, OpenCV
* **Análise de Dados:** Pandas, NumPy
* **Machine Learning:** Scikit-learn, LightGBM, Prophet
* **Visualização e UI:** Streamlit
* **Arquitetura:** Docker (para conteinerização)

## Estrutura do Projeto
Para uma visão detalhada da organização dos arquivos e diretórios, consulte a [estrutura detalhada do projeto](docs/structure_project.md).


## Como Rodar/Instalar

Siga os passos abaixo para configurar e executar o projeto localmente:

1.  Instale o PDM, Certifique-se de ter o PDM instalado:
    ```bash
    pip install pdm
    ```
    Para mais informações sobre a instalação, consulte a [documentação oficial do PDM.](https://pdm-project.org/latest/)
2.  Clone o repositório:
    ```bash
    git clone https://github.com/rbalbinotti/smart-supply-chain-ai.git
    cd smart-supply-chain-ai
    ```
3.  Instale as dependências e crie o ambiente virtual com o PDM:
    ```bash
    pdm install
    ```
    Isso vai instalar todas as bibliotecas necessárias no ambiente virtual do PDM.
4.  Execute o script principal com o PDM:
    ```bash
    pdm run python main.py
    ```
> Após a execução, a interface do dashboard estará acessível localmente.

## Gestão e Versionamento de Dados com DVC
Este projeto utiliza o **Data Version Control (DVC)** para versionar e gerenciar grandes conjuntos de dados e modelos. O DVC trabalha em conjunto com o Git, permitindo que o repositório de código permaneça leve, enquanto os arquivos grandes são armazenados separadamente.
> Observação:  Os arquivos de dados não são incluídos no repositório.

O que o DVC versiona:
- Datasets Brutos: data/raw/
- Datasets Processados: data/processed/
- Modelos de ML: models/

### Configuração do DVC
Para configurar o armazenamento remoto do DVC, execute o seguinte comando. Isso irá configurar o local onde os dados serão armazenados.

1. Crie o armazenamento local:

    ```bash
    # Para o armazenamento local que usamos neste projeto
    dvc remote add -d [nome_do_repositorio] [caminho_local]

    ex.: dvc remote add -d mylocalstorage /caminho/para/o/seu/dvc_local_storage
    ```
> **Observação**: O comando acima assume que você já possui a pasta /caminho/para/o/seu/dvc_local_storage criada.

2. Obtendo os Dados e Modelos
Após clonar o repositório, os arquivos de dados e modelos (que são pesados) não estarão presentes. Para baixá-los a partir do armazenamento remoto, use o comando dvc pull.
    ```bash
    pdm run dvc pull
    ```
> Este comando irá baixar todos os arquivos de dados e modelos referenciados nos arquivos .dvc e restaurá-los na estrutura de diretórios do seu projeto. (arquivos não fornecidos neste projeto)

3. Atualizando os Dados e Modelos
Sempre que você modificar um arquivo de dados ou modelo, use o comando dvc add para versioná-lo. Em seguida, envie os arquivos para o armazenamento remoto com dvc push.

    ```bash
    # Exemplo: Adicionando um novo arquivo de dados processados
    pdm run dvc add data/processed/new_processed_data.csv

    # Faça o commit do arquivo .dvc com o Git
    git add data/processed/new_processed_data.csv.dvc
    git commit -m "Adiciona dados processados atualizados"

    # Envie os dados para o armazenamento remoto do DVC
    pdm run dvc push
    ```

## Resultados e Impacto Esperados
O projeto visa transformar a gestão de estoque e compras, alcançando:

* **Redução de Custos:** Diminuição dos custos com excesso de estoque e logística de armazenamento.
* **Redução da Ruptura de Estoque:** Menor porcentagem de itens em falta.
* **Aumento da Liquidez:** Liberação de capital que estaria preso em estoque parado.



