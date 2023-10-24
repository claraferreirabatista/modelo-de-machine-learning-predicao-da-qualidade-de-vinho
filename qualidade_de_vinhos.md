# Solicitação de Citação

Este conjunto de dados está publicamente disponível para fins de pesquisa. Os detalhes estão descritos em [Cortez et al., 2009]. Por favor, inclua esta citação se planeja usar este banco de dados:

**Citação:**

P. Cortez, A. Cerdeira, F. Almeida, T. Matos e J. Reis.
Modelagem de preferências de vinho por meio de mineração de dados a partir de propriedades físico-químicas.
Em *Decision Support Systems*, Elsevier, 47(4):547-553. ISSN: 0167-9236.

Disponível em:
- [@Elsevier](http://dx.doi.org/10.1016/j.dss.2009.05.016)
- [Pré-impressão (pdf)](http://www3.dsi.uminho.pt/pcortez/winequality09.pdf)
- [bib](http://www3.dsi.uminho.pt/pcortez/dss09.bib)

## Informações do Conjunto de Dados

**1. Título:** Qualidade do Vinho

**2. Fontes**
   - Criado por: Paulo Cortez (Univ. Minho), Antonio Cerdeira, Fernando Almeida, Telmo Matos e Jose Reis (CVRVV) @ 2009

**3. Uso Anterior:**

No estudo de referência acima, foram criados dois conjuntos de dados, usando amostras de vinho tinto e branco. As entradas incluem testes objetivos (por exemplo, valores de pH) e a saída é baseada em dados sensoriais (mediana de pelo menos 3 avaliações feitas por especialistas em vinho). Cada especialista classificou a qualidade do vinho entre 0 (muito ruim) e 10 (excelente). Vários métodos de mineração de dados foram aplicados para modelar esses conjuntos de dados sob uma abordagem de regressão. O modelo de máquina de vetores de suporte obteve os melhores resultados. Várias métricas foram calculadas: MAD, matriz de confusão para uma tolerância de erro fixa (T), etc. Também plotamos a importância relativa das variáveis de entrada (medida por um procedimento de análise de sensibilidade).

**4. Informações Relevantes:**

Os dois conjuntos de dados estão relacionados às variantes vermelha e branca do vinho "Vinho Verde" de Portugal. Para mais detalhes, consulte [aqui](http://www.vinhoverde.pt/en/) ou a referência [Cortez et al., 2009]. Devido a questões de privacidade e logística, apenas variáveis físico-químicas (entradas) e variáveis sensoriais (a saída) estão disponíveis (por exemplo, não há dados sobre tipos de uva, marca do vinho, preço de venda do vinho, etc.).

Esses conjuntos de dados podem ser vistos como tarefas de classificação ou regressão. As classes são ordenadas e não balanceadas (por exemplo, há muito mais vinhos normais do que excelentes ou ruins). Algoritmos de detecção de valores atípicos podem ser usados para detectar os poucos vinhos excelentes ou ruins. Além disso, não estamos certos se todas as variáveis de entrada são relevantes. Portanto, pode ser interessante testar métodos de seleção de características.

**5. Número de Instâncias:** Vinho tinto - 1599; Vinho branco - 4898.

**6. Número de Atributos:** 11 + atributo de saída

   Nota: vários dos atributos podem estar correlacionados, portanto, faz sentido aplicar algum tipo de seleção de características.

**7. Informações de Atributos:**

   Para mais informações, leia [Cortez et al., 2009].

   - Variáveis de entrada (baseadas em testes físico-químicos):
     1 - acidez fixa
     2 - acidez volátil
     3 - ácido cítrico
     4 - açúcar residual
     5 - cloretos
     6 - dióxido de enxofre livre
     7 - dióxido de enxofre total
     8 - densidade
     9 - pH
     10 - sulfatos
     11 - álcool
   - Variável de saída (baseada em dados sensoriais): 
     12 - qualidade (pontuação entre 0 e 10)

**8. Valores de Atributos Ausentes:** Nenhum
