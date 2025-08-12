# Estrutura do Projeto

**Estrutura de Diretórios Sugerida:**

Serve como base para organização do projeto, pode não ser necessária todas as derivações.

project/
├──.github/	# Configurações de CI/CD 
├──.gitignore	# Arquivos e pastas ignorados pelo Git
├── data/	# Dados
│   ├── raw/	# Dados brutos
│   ├── processed/	# Dados pré-processados
│   └── synthetic/	# Dados sintéticos gerados
├── notebooks/	# Jupyter notebooks para exploração e prototipagem
├── src/	# Código fonte principal do projeto
│   ├── data_processing/	# Módulos EDA
│   │   ├── __init__.py
│   │   └── preprocess.py
│   ├── synthetic_data_gen/	# Módulos de dados sintéticos
│   │   ├── __init__.py
│   │   └── timegan_model.py
│   ├── cv_models/	# Módulos para modelos de Visão Computacional
│   │   ├── __init__.py
│   │   └── ocr_expiry.py
│   │   └── freshness_classifier.py
│   ├── ts_models/	# Módulos Séries Temporais
│   │   ├── __init__.py
│   │   └── demand_forecaster.py
│   ├── optimization/ 	# Módulos para lógica de otimização
│   │   ├── __init__.py
│   │   └── supplier_optimizer.py
│   └── utils/ 	# Funções utilitárias gerais
│       └── __init__.py
│       └── metrics.py
├── models/ 	# Modelos treinados (privados ou DVC)
├── api/ 	# Código para a API 
│   ├── __init__.py
│   ├── main.py 	# Ponto de entrada da API
│   └── Dockerfile	# Dockerfile para a API
├── tests/ 	# Testes unitários e de integração
├── docs/ 	# Documentação do projeto
├──.env	# Variáveis de ambiente (SEMPRE PRIVADO)
├── requirements.txt	# Dependências do projeto
├── README.md	# Descrição do projeto
└── dvc.yaml	# Configuração do DVC (se usar)

