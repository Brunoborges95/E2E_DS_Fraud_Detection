# Projeto de Detecção de Fraudes

Este projeto é uma abordagem abrangente para a detecção de fraudes utilizando técnicas avançadas de modelagem e avaliação. Ele inclui notebooks para Exploração de Dados (EDA), modelagem, avaliação e deployment dos modelos, utilizando ZenML e MLflow para o pipeline de dados e registro dos modelos, e AWS para deployment na nuvem.

### Descrição dos Arquivos

- **EDA.ipynb**: Notebook contendo a Análise Exploratória dos Dados (EDA). Este notebook examina a distribuição das variáveis, a presença de valores ausentes, outliers, e outras estatísticas descritivas para entender melhor os dados antes da modelagem.
  
- **Notebook.ipynb**: Notebook principal de modelagem e avaliação. Contém as seguintes seções:
  - **Exploração dos Dados**: Visualização e análise preliminar dos dados.
  - **Treinamento de Modelos**: Implementação de três algoritmos de machine learning:
    - XGBoost
    - CatBoost
    - Regressão Logística
  - **Teste de Calibração**: Avaliação da calibração dos modelos utilizando técnicas como Platt Scaling ou isotonic regression.
  - **Avaliação**: Comparação dos modelos utilizando diversas métricas:
    - Estatística KS
    - Curva ROC
    - Curva Lift
    - Métricas para bases desbalanceadas, com ênfase em falsos negativos, que custam 10x mais que falsos positivos.
    - Curva de Lucro
  - **Inferência**: Teste de tempo de inferência para garantir rapidez na detecção de fraudes.

- **Desafio - Scoring.pdf**: Documento descrevendo o desafio e os critérios de scoring para o projeto.

- **Makefile**: Arquivo para automação de tarefas comuns no projeto, como instalação de dependências e execução de testes.

- **dados.zip**: Arquivo contendo os dados utilizados no projeto.

- **requirements.txt**: Lista de dependências necessárias para executar os notebooks e scripts do projeto.

- **test-eda.pdf**: Relatório em PDF gerado a partir do notebook de EDA.

- **tree_high_res.png**: Imagem de alta resolução da árvore de decisão gerada por um dos modelos.

- **utils.py**: Script contendo funções utilitárias usadas nos notebooks e scripts do projeto.

- **zenml_cat_app.py**: Script para deployment do modelo CatBoost utilizando ZenML e MLflow.

- **zenml_xgb_app.py**: Script para deployment do modelo XGBoost utilizando ZenML e MLflow.

## Pipeline de Dados e Deployment

Utilizamos ZenML para criar um pipeline de dados que facilita a manutenção e reprodutibilidade do processo de modelagem. O MLflow é utilizado para o registro e rastreamento dos modelos treinados.

### Deployment na AWS

Para o deployment, configuramos a infraestrutura na AWS para hospedar os modelos XGBoost e CatBoost, garantindo alta disponibilidade e escalabilidade. Os scripts `zenml_cat_app.py` e `zenml_xgb_app.py` cuidam da integração e deploy dos modelos na nuvem.

## Como Executar

1. Clone o repositório:
   ```bash
   git clone https://github.com/seu-usuario/projeto-deteccao-fraudes.git
   cd projeto-deteccao-fraudes
   ```

2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

3. Execute a análise exploratória:
   ```bash
   jupyter notebook EDA.ipynb
   ```

4. Treine e avalie os modelos:
   ```bash
   jupyter notebook Notebook.ipynb
   ```

5. Faça o deploy dos modelos:
   ```bash
   python zenml_xgb_app.py
   python zenml_cat_app.py
   ```
