# Análise de Vinhos Tinto e Vinhos Branco portugueses

O conjunto de dados empregado no presente projeto, acessível por meio do link [https://archive.ics.uci.edu/dataset/186/wine+quality](https://archive.ics.uci.edu/dataset/186/wine+quality), compreende cerca de 5.000 observações de dados relevantes que descrevem atributos do vinho branco e vinho tinto, incluindo, mas não se limitando a, acidez, teor de açúcar residual, pH, entre outros. Paralelamente, dispomos de aproximadamente 1.600 observações de dados que caracterizam o vinho tinto. Neste contexto, objetivamos a fusão destes dois conjuntos de dados para a elaboração do presente modelo.

### No total, temos 11 features dentro desse Dataset, são elas:

1. acidez_fixa (g(ácido_tartárico)/dm³)

2. acidez_volátil (g(ácido_acético)/dm³)

3. ácido_cítrico (g/dm³)

4. açúcar_residual (g/dm³)

5. cloretos (g(cloreto_de_sódio)/dm³)

6. dióxido de enxofre livre (mg/dm³)

7. dióxido de enxofre total (mg/dm³)

8. densidade (g/cm³)

9. pH

10. sulfatos (g(sulfato_de_potássio)/dm³)

11. teor alcoólico (% vol.)

12. Qualidade do vinho (número de 0 a 10).

## Membros

**[Clara Ferreira Batista](https://www.linkedin.com/in/clara-ferreira-batista/)**

**[Laura Muglia](https://www.linkedin.com/in/lauramuglia/)**

**[Luana Ferraz](https://www.linkedin.com/in/luanamariaferraz/)**

**[Pedro Elias](https://www.linkedin.com/in/pedro-elias-muniz-peres-378b41206/)**

**[Rafael Couto de Oliveira](https://www.linkedin.com/in/couto21/)**

**Gostou do projeto? Siga a gente no Linkedin e mande mensagem**

## 1. Pré-processamento dos Dados
- Leitura dos dados de um arquivo CSV.
- Remoção da primeira coluna que continha índices.
- Verificação de informações sobre o conjunto de dados, como dimensões e informações sobre as colunas.
- Verificação de valores nulos no conjunto de dados.

## 2. Análise Univariada
- Exibição de histogramas para visualizar a distribuição das características.
- Criação de um mapa de calor para mostrar a correlação entre diferentes características.

## 3. Preparação dos Dados
- Conversão da variável alvo (qualidade) em uma variável binária (ruim ou bom) com base em intervalos.
- Uso do LabelEncoder para transformar a variável alvo em valores numéricos.

## 4. Modelagem de Dados
- Divisão do conjunto de dados em treinamento e teste.
- Comparação de vários algoritmos de classificação, como Regressão Logística, SVM, K-Nearest Neighbors, Random Forest, Decision Tree, Gradient Boosting e Naive Bayes, sem escalonamento das características.

## 5. Escalonamento de Características
- Aplicação de técnicas de escalonamento, incluindo Min-Max Scaling e Standard Scaling.
- Avaliação do desempenho dos modelos após o escalonamento.

## 6. Ajuste de Parâmetros (Otimização de Hiperparâmetros)
- Otimização dos hiperparâmetros de alguns algoritmos, como Regressão Logística, K-Nearest Neighbors, Support Vector Machine (SVM), Random Forest e Gradient Boosting.
- Comparação dos modelos com os hiperparâmetros otimizados.

## 7. Resultados
- Apresentação das acurácias dos modelos de classificação antes e depois da otimização de hiperparâmetros.
- Identificação dos modelos mais eficazes.

## Conclusões
- O modelo SVM com kernel RBF obteve a maior acurácia após o ajuste de parâmetros, atingindo 91,75%.

## Requisitos

Antes de executar o código, certifique-se de que as seguintes bibliotecas estejam instaladas em seu ambiente Python:

- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn

Você pode instalar essas bibliotecas usando o pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Executando o Código

O código pode ser executado em um ambiente Jupyter notebook compatível com as bibliotecas mencionadas. Certifique-se de que o arquivo "analise_vinho_tinto.csv" e "analise_vinho_branco.csv" esteja no mesmo diretório do arquivo Jupyter Notebook (analise_e_modelagem_de_dados_vinho_tinto.ipynb) ou ajuste o caminho do arquivo de dados conforme necessário.

Execute o código para realizar a análise de dados e a modelagem de treinamento de máquina. Os resultados, incluindo a precisão dos modelos, serão exibidos no final da execução.

## Considerações Finais

Este código oferece uma visão geral do processo de análise de dados e modelagem de treinamento de máquina para prever a qualidade do vinho tinto. Você pode ajustar e expandir esse código para atender às suas necessidades específicas ou aplicá-lo a outros conjuntos de dados de classificação.

Lembre-se de que a escolha do modelo e os hiperparâmetros podem depender da natureza dos dados, portanto, é importante ajustar o código de acordo com o contexto do seu projeto.

## Notas

- Este código é um exemplo de análise de dados e modelagem de classificação em um conjunto de dados específico.
- Os resultados podem variar dependendo dos dados e dos hiperparâmetros escolhidos.
- Este README fornece uma visão geral do código, mas a análise completa e os resultados podem ser encontrados na saída do código Jupyter Notebook.

Divirta-se explorando a análise de qualidade do vinho tinto e ajustando os modelos para obter o melhor desempenho!