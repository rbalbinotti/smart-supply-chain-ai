# Estrutura do Projeto

.  
├── data  
│	 	├── external  
│	 	├── processed  
│	 	└── raw  
├── docs  
├── LICENSE  
├── models  
├── notebooks  
├── pyproject.toml  
├── README.md  
├── src  
│	 	└── smart_supply_chain_ai  
└── tests  


***data/***: Área de dados.

	data/raw/: Para os dados originais e imutáveis.

	data/processed/: Para os dados que foram limpos, processados e prontos para serem usados na modelagem.

	data/external/: Para dados de terceiros que não fazem parte do seu conjunto de dados original.

***docs/***: Ideal para colocar qualquer documentação do projeto, como guias de instalação, explicações sobre o modelo ou até mesmo relatórios técnicos.

***models/***: Esta pasta deve ser usada para armazenar os modelos de IA treinados.

***notebooks/***: O lugar perfeito para seus notebooks Jupyter.

***pyproject.toml***: O arquivo principal do seu projeto, gerenciado pelo PDM.

***src/***: Código-fonte que será instalado como um pacote.

	src/smart_supply_chain_ai/: Este é o seu pacote Python.

***tests/***: Onde ficam os seus testes.

***README.md***: O arquivo de "leia-me".

