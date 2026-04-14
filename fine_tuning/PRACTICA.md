# Fine-Tuning de Modelos en Azure AI Foundry

En esta práctica aplicarás los conocimientos sobre fine-tuning de modelos de lenguaje en Azure AI Foundry. Entrenarás un modelo personalizado utilizando tu propio dataset y evaluarás su rendimiento comparándolo con el modelo base.

Podrás elegir entre dos modalidades de implementación:
- **Modalidad Portal**: Realizar el fine-tuning desde Azure AI Foundry Studio (requiere video del proceso)
- **Modalidad Python SDK**: Implementar el fine-tuning programáticamente usando código Python

La práctica se divide en una única parte integral y el entregable final será un **Jupyter Notebook** que documente todo el proceso y demuestre el funcionamiento del modelo fine-tuned.

---

## Parte 1) Entrenamiento y Evaluación de Modelo Fine-Tuned
**Objetivo:** Entrenar un modelo de lenguaje personalizado mediante fine-tuning, desplegarlo y evaluar su rendimiento mediante pruebas comparativas y análisis de métricas.

### 1.1 - Preparación del Dataset de Fine-Tuning

Crea o selecciona un dataset personalizado que defina el comportamiento deseado para tu modelo fine-tuned.

**Requisitos del dataset:**
- Formato **JSONL** (JSON Lines) compatible con Chat Completions API
- Mínimo **50-100 ejemplos** de conversaciones (recomendado: 100-300 para mejores resultados)
- Estructura conversacional con roles: `system`, `user`, `assistant`
- División en dos archivos:
  - `training_set.jsonl` (80% de los datos)
  - `validation_set.jsonl` (20% de los datos)

> [!TIP]
> **Datasets Automatizados**: He incluido un script llamado `prepare_datasets.py` en esta carpeta que descarga automáticamente el dataset de rendimiento académico sugerido por el usuario y genera los archivos `.jsonl` con el reparto 80/20 solicitado. Solo tienes que ejecutarlo con `python prepare_datasets.py`.

### 1.2 - Entrenamiento del Modelo (elegir una modalidad)

#### 🖥️ Opción A - Modalidad Portal (Azure AI Foundry Studio)
Si eliges esta modalidad, graba un video mostrando la configuración y súbelo a SharePoint.

#### 💻 Opción B - Modalidad Python SDK
Implementa el proceso usando el SDK de OpenAI para Azure.

### 1.3 - Despliegue del Modelo Fine-Tuned
Despliega el modelo exitoso y obtén el `deployment_name`.

### 1.4 - Pruebas y Evaluación del Modelo
Compara el modelo base vs el fine-tuned y analiza las métricas de `loss`.

---

## Entregable:
Un Jupyter Notebook (`.ipynb`) con:
1. Introducción y Contexto
2. Proceso de Fine-Tuning (Video o Código)
3. Análisis de Métricas
4. Pruebas Comparativas
5. Conclusiones
