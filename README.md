# Proyecto de Data Science: Concesión de Préstamos

Este es un proyecto de análisis de datos centrado en el proceso de concesión de préstamos, que forma parte de la prueba técnica de la empresa Mokka. El objetivo principal es desarrollar un modelo que prediga el valor de la variable objetivo **bad_flag**, o la concesion del préstamo.

## Introducción

El proyecto se divide en las siguientes partes: 

- **EDA**: Se realiza un analisis exploratorio de los datos para hacernos una idea general de cómo se distribuye la variable bad_flag con respecto al resto de variables (género, mes en el que se solicita el préstamo o región)

- **Main**: El notebook Main realiza una primera aproximación de diferentes modelos a usar, entre ellos BaggingClassifier, GaussianNB, RandomForestClassifier, o ExtraTreesClassifier. Los resultados de las metricas utilizadas (Accuracy, precision, recall y f1) muestran un resultado insatisfactorio debido al desbalanceamiento de clases, por lo que se tomarán dos posibles aproximaciones, realizadas en los notebook exp_1 y exp_2, utilizando las tecnicas de undersampling (con RandomUnderSampling) y oversampling (SMOTE).

- **Exp_1**: Este notebook usa el metodo de undersampling (para balancear las clases de la variable target bad_flag), entrena de nuevo los modelos KNeighborsClassifier, LogisticRegression, BaggingClassifier, GaussianNB, RandomForestClassifier, ExtraTreesClassifier y se utilizan las metricas citadas anteriormente.

- **Exp_2**: Este notebook usa el metodo de oversampling usando la técnica SMOTE (Synthetic Minority Over- Sampling), entrena de nuevo los modelos KNeighborsClassifier, LogisticRegression, BaggingClassifier, GaussianNB, RandomForestClassifier, ExtraTreesClassifier y se utilizan las metricas citadas anteriormente.

## Parte 1: Análisis Exploratorio de Datos (EDA)

Con el análisis exploratorio de datos se han comprobado las distintas relaciones entre la concesión o no del préstamo, con diferentes características del solicitante, es decir, su género, la región de la que proviene, o qué mes pidió el préstamo. las siguientes gráficas ilustra lo anterior:

![Mes donde mas se solicitan préstamos](img/EDA/month.png)

Como se puede comprobar, octubre es el mes donde mas préstamos se solicitan.

En cuanto al género y la region del solicitante nos encontramos lo siguiente:

<div style="display: flex;">
  <div style="flex: 50%;">
    <img src="img/EDA/genre.png" width="100%">
    <p align="center">Solicitud de préstamo por género</p>
  </div>
  <div style="flex: 50%;">
    <img src="img/EDA/region.png"  width="100%">
    <p align="center">Solicitud de préstamo por región</p>
  </div>
</div>

Hay una disparidad muy grande entre hombres y mujeres en cuanto a solicitud de préstamos, al igual que en las regiones.

| Género                         | Región                          |
|:----------------------------------:|:----------------------------------:|
| ![Solicitud por género](img/EDA/genre.png)      | ![Solicitud por región](img/EDA/region.png)      |