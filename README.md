# IA-BayesianNetworks

# Redes Bayesianas para el Análisis de Respuesta a Desastres Naturales

Este proyecto utiliza Redes Bayesianas para analizar e inferir la efectividad de la respuesta a desastres naturales en Estados Unidos del año 2000 al 2023. El modelo utiliza varios factores como el entrenamiento de preparación, la concienciación comunitaria y otras variables relacionadas con desastres para predecir el tipo de desastre y sus impactos.

## Tabla de Contenidos

1. [Descripción del Proyecto](#descripción-del-proyecto)
2. [Dependencias](#dependencias)
3. [Datos](#datos)
4. [Estructura del Modelo](#estructura-del-modelo)
5. [Métodos de Inferencia Utilizados](#métodos-de-inferencia-utilizados)
6. [Cómo Ejecutar](#cómo-ejecutar)
7. [Resultados](#resultados)
8. [Licencia](#licencia)

## Descripción del Proyecto

Este proyecto tiene como objetivo proporcionar un análisis probabilístico de escenarios de respuesta a desastres naturales utilizando Redes Bayesianas. El modelo predice el tipo de desastre basado en factores como el entrenamiento de preparación y la concienciación comunitaria, y evalúa el impacto en varios parámetros de respuesta como el personal involucrado, suministros médicos y daños económicos.

## Dependencias

Para ejecutar este proyecto, necesitas las siguientes dependencias:

- Python 3.x
- pandas
- pgmpy
- numpy

Puedes instalar estas dependencias usando `pip`:

Instalación de pandas:
```bash
pip install pandas
```
- `pandas` se ha utilizado para la manipulación y análisis de datos. Se ha leído el archivo csv que contiene los datos de desastres naturales y para realizar operaciones de discretización de los datos numéricos en categorías.

Instalación de pgmpy: 
```bash
pip install pgmpy
```
- `pgmpy` (Probabilistic Graphical Models using Python) es una biblioteca utilizada para construir y trabajar con modelos gráficos probabilísticos. Se ha utilizado para crear y definir la estructura de la red bayesiana, estimar las distribuciones de probabilidad condicional (CPDs) y realizar inferencias exactas y aproximadas sobre los datos.

Instalación de numpy: 
```bash
pip install numpy
```
- `numpy` se ha utilizado para generar datos de ejemplo, particularmente para crear datos aleatorios que simulan los valores de las variables numéricas en el conjunto de datos de desastres naturales.

## Datos
El conjunto de datos utilizado en nuestro proyecto contiene información sobre varios eventos de desastres naturales, incluyendo factores como el entrenamiento de preparación, la concienciación comunitaria y los impactos en recursos e infraestructura. Los datos están discretizados para adaptarse al modelo de Red Bayesiana.

### Ejemplo de Datos

Aquí hay una breve vista del conjunto de datos:

| EventID | DisasterType | Date       | Latitude | Longitude  | Region      | Personnel | MedicalSupplies | Shelters | FoodWater | ResponseTime | ResponseDuration | PeopleAffected | PeopleAssisted | InfrastructureCondition | PreparednessTraining | CommunityAwareness | AgenciesInvolved | CoordinationScore | CommunicationScore | EconomicDamage | Casualties | LongTermImpact |
|---------|--------------|------------|----------|------------|-------------|-----------|-----------------|----------|-----------|--------------|------------------|----------------|----------------|-------------------------|---------------------|--------------------|------------------|------------------|------------------|----------------|------------|----------------|
| 1       | Flood        | 2010-01-01 | 28.262237| -103.688502| New Orleans | 330       | 155             | 21       | 1279      | 3.97         | 79.14            | 9343           | 580            | Poor                    | No                  | Yes                | 9                | 7                | 5                | 470357593      | 408        | 6              |

Los diferentes tipos de datos que tenemos son los siguientes: 
- **EventID**: Identificador único que identifica cada desastre registrado en el dataset.
- **DisasterType**: Tipo de desastre (p. ej., inundación, tornado, terremoto, huracán).
- **Date**: Fecha del evento de desastre.
- **Latitude**, **Longitude**: Coordenadas geográficas del evento.
- **Region**: Región geográfica del evento.
- **Personnel**, **MedicalSupplies**, **Shelters**, **FoodWater**: Cantidad de personal, suministros médicos, refugios y suministros de alimentos/agua involucrados en el desastre.
- **ResponseTime**, **ResponseDuration**: Tiempo de respuesta (Cuanto tardaron en actuar) y duración de la respuesta al desastre.
- **PeopleAffected**, **PeopleAssisted**: Número de personas afectadas y asistidas por el desastre.
- **InfrastructureCondition**: Condición de la infraestructura (p. ej., pobre, justa, buena).
- **PreparednessTraining**, **CommunityAwareness**, **AgenciesInvolved**, **CoordinationScore**, **CommunicationScore**: Factores relacionados con la preparación para desastres y la coordinación de la respuesta.
- **EconomicDamage**: Daños económicos causados por el desastre.
- **Casualties**: Número de víctimas del desastre.
- **LongTermImpact**: Impacto a largo plazo del desastre.

### Estructura del Modelo

La Red Bayesiana está estructurada con `DisasterType` como el nodo central. El modelo incluye factores que influyen en el tipo de desastre y sus consecuencias. La estructura se define de la siguiente manera:

```python
model = BayesianNetwork([
    #Esto son las causas o factores que influyen en el tipo de desastre
    ("PreparednessTraining", "DisasterType"),
    ("CommunityAwareness", "DisasterType"),
    ("AgenciesInvolved", "DisasterType"),
    ("CoordinationScore", "DisasterType"),
    ("CommunicationScore", "DisasterType"),
    #Consecuencias o impactos que tienen diferentes aspectos del desastre
    ("DisasterType", "Personnel"),
    ("DisasterType", "MedicalSupplies"),
    ("DisasterType", "Shelters"),
    ("DisasterType", "FoodWater"),
    ("DisasterType", "ResponseTime"),
    ("DisasterType", "ResponseDuration"),
    ("DisasterType", "PeopleAffected"),
    ("DisasterType", "PeopleAssisted"),
    ("DisasterType", "InfrastructureCondition"),
    ("DisasterType", "EconomicDamage"),
    ("DisasterType", "Casualties"),
    ("DisasterType", "LongTermImpact")
])
```

## Métodos de Inferencia Utilizados

### Inferencia Exacta
- **Eliminación de Variables**: Método que elimina variables no necesarias mediante la suma marginal, simplificando la red y calculando probabilidades exactas.
```python
from pgmpy.inference import VariableElimination
```

- **Propagación de Creencias**: Algoritmo que utiliza la factorización de la distribución conjunta para actualizar las probabilidades de las variables en la red.
```python
from pgmpy.inference import BeliefPropagation
```
- **Inferencia Causal**: Técnica utilizada para determinar el efecto de intervenciones en la red, basada en la teoría de la causalidad.
```python
from pgmpy.inference import CausalInference
```

### Inferencia Aproximada
- **Muestreo de Importancia con Reasignación (SIR)**: Método que genera muestras ponderadas para aproximar distribuciones posteriores, ajustando las ponderaciones para mejorar la precisión.
```python
from pgmpy.inference import ApproxInference
```
- **Inferencia Aproximada usando ApproxInference de pgmpy**: Algoritmo que utiliza técnicas de muestreo y aproximación para estimar las probabilidades en redes complejas donde los métodos exactos son ineficaces.
```python
from pgmpy.sampling import BayesianModelSampling
```

## Cómo Ejecutar

### 1. Clonar el repositorio:
``` bash
git clone https://github.com/tuusuario/redes-bayesianas-respuesta-desastres.git
cd redes-bayesianas-respuesta-desastres
``` 

### 2. Instalar dependencias:
``` bash
pip install -r requirements.txt
``` 

### 3. Preparar los datos:
Asegúrate de que los datos estén en el formato correcto y discretizados según sea necesario. Se proporciona un conjunto de datos de ejemplo `natural_disasters_discretized_en.csv`.

### 4. Ejecutar el modelo:
``` bash
python model_training.py
```

## Resultados
El modelo proporciona predicciones probabilísticas de los tipos de desastres y sus impactos basados en la evidencia proporcionada. Ejemplo de salida:

- **Eliminación de variables:**

| DisasterType          | phi(DisasterType) |
|-----------------------|-------------------|
| Hurricane             | 0.2845            |
| Flood                 | 0.2199            |
| Tornado               | 0.2381            |
| Earthquake            | 0.2575            |

la tabla muestra la distribución de probabilidad de diferentes tipos de desastres naturales. Cada fila representa un tipo de desastre. La columna **"phi(DisasterType)"** indica la probabilidad de que ocurra cada tipo de desastre si la causa `PreparednessTraining` ocurre, es decir, `PreparednessTraining` = 'Yes'. Por ejemplo, la probabilidad de que ocurra un huracán si `PreparednessTraining` = 'Yes' es del 28.45%, mientras que la probabilidad de una inundación es del 21.99%, y así sucesivamente.

## Licencia 
Este proyecto está licenciado bajo la Licencia MIT.


### Archivo `requirements.txt`

Crea un archivo `requirements.txt` con las dependencias necesarias:

```txt
pandas
pgmpy
numpy
```

