# Concesión de Préstamos

Este es un proyecto de análisis de datos centrado en el proceso de concesión de préstamos, que forma parte de una prueba técnica de la empresa Mokka. El objetivo principal es desarrollar un modelo que prediga el valor de la variable objetivo **bad_flag**, o la concesion del préstamo.

## Introducción

El proyecto se divide en los siguientes notebooks principales: 

- **EDA**: Se realiza un analisis exploratorio de los datos para hacernos una idea general de cómo se distribuye la variable bad_flag con respecto al resto de variables (género, mes en el que se solicita el préstamo o región)

- **Main**: El notebook Main realiza una primera aproximación de diferentes modelos a usar, entre ellos BaggingClassifier, GaussianNB, RandomForestClassifier, o ExtraTreesClassifier. Los resultados de las metricas utilizadas (Accuracy, precision, recall y f1) muestran un resultado insatisfactorio debido al desbalanceamiento de clases, por lo que se tomarán dos posibles aproximaciones, realizadas en los notebook exp_1 y exp_2, utilizando las tecnicas de undersampling (con RandomUnderSampling) y oversampling (SMOTE).

- **Exp_1**: Este notebook usa el método de undersampling (para balancear las clases de la variable target bad_flag), entrena de nuevo los modelos KNeighborsClassifier, LogisticRegression, BaggingClassifier, GaussianNB, RandomForestClassifier, ExtraTreesClassifier y se utilizan las metricas citadas anteriormente.

- **Exp_2**: Este notebook usa el método de oversampling usando la técnica SMOTE (Synthetic Minority Over- Sampling), entrena de nuevo los modelos KNeighborsClassifier, LogisticRegression, BaggingClassifier, GaussianNB, RandomForestClassifier, ExtraTreesClassifier y se utilizan las metricas citadas anteriormente.

Se dividirá la estructuctura en dos partes diferenciadas, por un lado la primera parte, correspondera al EDA (Exploratory Data Analysis) y la segunda parte corresponderá a los notebooks Main, exp_1 y exp_2, donde se prueban diferentes modelos de clasificación.

## Parte 1: Análisis Exploratorio de Datos (EDA)

Con el análisis exploratorio de datos se han comprobado las distintas relaciones entre la concesión o no del préstamo, con diferentes características del solicitante, es decir, su género, la región de la que proviene, o qué mes pidió el préstamo. las siguientes gráficas ilustra lo anterior:

![Mes donde mas se solicitan préstamos](img/EDA/month.png)

Como se puede comprobar, octubre es el mes donde mas préstamos se solicitan.

En cuanto al género y la region del solicitante nos encontramos lo siguiente:

![Mes donde mas se solicitan préstamos](img/EDA/genre.png)
![Mes donde mas se solicitan préstamos](img/EDA/region.png)


<!-- <div style="display: flex;">
  <div style="flex: 50%;">
    <img src="img/EDA/genre.png" width="100%">
    <p align="center">Solicitud de préstamo por género</p>
  </div>
  <div style="flex: 50%;">
    <img src="img/EDA/region.png"  width="100%">
    <p align="center">Solicitud de préstamo por región</p>
  </div>
</div> -->

Hay una disparidad muy grande entre hombres y mujeres en cuanto a solicitud de préstamos sin embargo, el porcentaje de aceptados es mayor para las mujeres, un **14%** por un **9.35%** de los hombres. En cuanto a las regiones, las tres regiones que más prestamos solicitan son la región 3, 6 y 2, mientras que en porcentaje de aceptación, las tres primeras son la **region 4** con un **12.4%**, la **region 3** con un **11.4%**, y la **region 0** con un **10.5%**.


## Parte 2: Machine learning

En esta segunda parte se utiliza el dataset modificado obtenido desde el notebook EDA.ipynb "clean_dataset.csv" para probar diferentes algoritmos de clasificación. El primer notebook corresponde a Main.ipynb donde se prueban los algoritmos BaggingClassifier, GaussianNB, RandomForestClassifier y ExtraTreesClassifier, se grafican las matrices de confusion para cada uno de los algoritmos y se obtienen las siguientes métricas para cada uno de ellos: 

| Algoritmo             | F1-Score  | Accuracy  | Precisión | Recall    |
|-----------------------|-----------|-----------|-----------|-----------|
| BaggingClassifier     | 0.2095     | 0.90      | 0.5      |0.1325      |
| GaussianNB            | 0.1929      | 0.8891      | 0.3548      | 0.1325      |
| RandomForestClassifier| 0.1836      | 0.9036      | 0.6      | 0.1084      |
| ExtraTreesClassifier  | 0.1020      | 0.8939      | 0.3333      | 0.0602      |

Como se puede comprobar, las métricas reflejan un claro desbalanceamiento de clases, ya que aunque la métrica accuracy es muy alta para todas los algoritmos, el recall y la precisión son muy bajas, esto quiere decir que nuestro modelo no es capaz de clasificar correctamente los casos donde bad_flag es 1, ya que recall (cantidad de casos que nuestro modelo es capaz de predecir correctamente) y precision (si lo que predice nuestro modelo es correcto) tienen unos valores bajos. A modo de ejemplo, la siguiente matriz de confusión (Del RandomForestClassifier) muestra el numero de muestras predichas:


![Matriz de confusión del algoritmo RandomForestClassifier](img/main/randomforestclas_cm.png)
