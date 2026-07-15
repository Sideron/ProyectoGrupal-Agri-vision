# Clasificación de hojas de cultivos mediante CNN, Vision Transformer y redes siamesas

**Curso:** _Deep Learning_
**Docente:** _Escobedo Cardenas, Edwin Jonathan_
**Fecha:** _17/07/2026_

**Integrantes:**
- Almerco Velita, Alonso Feliciano
- Montero Gutierrez, Eduardo Cristopher
- Oshiro Ugamoto, Ryuichi
- Sulca Ramirez, Rodrigo Fernando

---

## 1. Introducción

_[Redactar: 1-2 páginas]_

- Contexto y motivación del problema: importancia de la detección temprana de enfermedades
  en hojas de cultivos (impacto agrícola/económico) y el rol de la visión por computadora en
  la automatización de este diagnóstico.
- Problema abordado: clasificación de imágenes de hojas en múltiples clases (variedad de
  cultivo × estado sano/enfermedad), comparando distintas arquitecturas y estrategias de
  aprendizaje (CNN, Transformer, redes siamesas + métricas de distancia, aprendizaje profundo
  combinado con ML clásico).
- Objetivo general y objetivos específicos del trabajo (uno por cada escenario experimental,
  ver sección 2.2).
- Breve resumen de la estructura del informe.

---

## 2. Metodología

### 2.1 Descripción del dataset

| Característica | Detalle |
|---|---|
| Fuente | Journal *Data in Brief* (ScienceDirect) — _[completar título del artículo, autores, año y DOI]_ |
| Dominio | Imágenes de hojas de cultivos (Batol Gourd, Tomate, Zucchini, Papaya) |
| N.º de clases | 28 |
| N.º de imágenes totales | 28,121 (train: 26,536 · valid: 772 · test: 813) |
| Resolución original | _[completar]_ |
| Formato | _[completar, ej. JPG/PNG]_ |
| Split | 70 % train / 15 % valid / 15 % test (`split_dataset.ipynb`, generado en `Dataset_split_Aug/`) |

**Listado de clases** (obtenido de `CLASS_NAMES` en `clasificacion_hojas_siamese_triplet.ipynb`):

`Batol_Gourd_Alternaria_Leaf_Blight`, `Batol_Gourd_Anthracnose`, `Batol_Gourd_Downy_Mildew`,
`Batol_Gourd_Early_Alternaria_Leaf_Blight`, `Batol_Gourd_Fungal_Damage_Leaf`,
`Batol_Gourd_Healthy`, `Batol_Gourd_Mosaic_Virus`, `Tomato_Downy Mildew`, `Tomato_Healthy`,
`Tomato_Mosaic`, `Tomato_Spot`, `Tomato_white_spot`, `Zucchini_Angular_Leaf_Spot`,
`Zucchini_Anthracnose`, `Zucchini_Downy_Midew`, `Zucchini_Dry_Leaf`, `Zucchini_Healthy`,
`Zucchini_Insect_Damage`, `Zucchini_Iron_Chlorosis_Damage`, `Zucchini_Xanthomonas_Leaf_Spot`,
`Zucchini_Yellow_Mosaic_Virus`, `papaya_Bacterial_Blight`, `papaya_Carica_Insect_Hole`,
`papaya_Curled_Yellow_Spot`, `papaya_Mosaic_Virus`, `papaya_Yellow_Necrotic_Spots_Holes`,
`papaya_healthy_leaf`, `papaya_pathogen_symptoms`

**Distribución de clases:** _[insertar gráfico de barras de conteo por clase — celda "2.1
Distribución de clases y muestra visual" del notebook `clasificacion_hojas_resnet_vs_vit.ipynb`]_

**Preprocesamiento y aumento de datos:**
- Redimensionado a 224×224 px (requerido por ResNet-50 y ViT-B/16 preentrenados en ImageNet).
- Normalización con media/desviación estándar de ImageNet.
- Aumentos aplicados solo en entrenamiento: `RandomResizedCrop`, flips horizontal/vertical,
  rotación aleatoria (±20°), *color jitter*.

### 2.2 Protocolo experimental

Se evaluaron **6 escenarios** de clasificación sobre el mismo split de datos, todos usando
*transfer learning* con pesos preentrenados en ImageNet como punto de partida:

| # | Escenario | Notebook | Descripción |
|---|---|---|---|
| 1 | CNN (ResNet-50) | `clasificacion_hojas_resnet_vs_vit.ipynb` | Fine-tuning end-to-end de ResNet-50 con cabeza lineal de 28 clases. |
| 2 | Vision Transformer (ViT-B/16) | `clasificacion_hojas_resnet_vs_vit.ipynb` | Fine-tuning end-to-end de ViT-B/16 con cabeza lineal de 28 clases. |
| 3 | Siamesa + Contrastive Loss + FC | `clasificacion_hojas_siamese_contrastive.ipynb` | Backbone ResNet-50 entrenado con contrastive loss sobre pares (misma clase / distinta clase); luego se congela y se entrena una capa FC para las 28 clases. |
| 4 | Siamesa + Contrastive Loss + XGBoost | `clasificacion_hojas_siamese_contrastive.ipynb` | Mismo backbone congelado del escenario 3, usado para extraer embeddings; se entrena un clasificador XGBoost sobre esos vectores. |
| 5 | Siamesa + Triplet Loss + FC | `clasificacion_hojas_siamese_triplet.ipynb` | Backbone ResNet-50 (independiente) entrenado con triplet loss (anchor/positive/negative); luego se congela y se entrena una capa FC para las 28 clases. |
| 6 | Siamesa + Triplet Loss + XGBoost | `clasificacion_hojas_siamese_triplet.ipynb` | Mismo backbone congelado del escenario 5, usado para extraer embeddings; se entrena un clasificador XGBoost sobre esos vectores. |

**Detalles comunes a todos los escenarios:**
- Arquitectura base: ResNet-50 preentrenado en ImageNet (`ResNet50_Weights.IMAGENET1K_V2`).
- ViT-B/16 preentrenado en ImageNet (`ViT_B_16_Weights.IMAGENET1K_V1`) solo en el escenario 2.
- Optimizador: AdamW, *weight decay* = 1e-4.
- *Scheduler*: Cosine Annealing.
- Precisión mixta (AMP) cuando hay GPU disponible.
- Selección de mejor modelo por F1-macro (clasificadores) o menor pérdida de validación
  (redes siamesas) durante el entrenamiento.
- Semilla fija (`SEED = 42`) para reproducibilidad.

**Particularidades de las redes siamesas (escenarios 3-6):**
- Dimensión del embedding: 128, normalizado L2.
- Margen: 1.0 (igual en contrastive y triplet, para que ambos escenarios sean comparables).
- Contrastive loss: pares 50 % misma clase / 50 % clase distinta.
- Triplet loss: tripletas (anchor, positive de la misma clase, negative de clase distinta).
- En las Tareas A (FC) de ambos, el backbone queda **congelado** (incluyendo estadísticas de
  BatchNorm) y solo se entrena la capa de clasificación final, tal como exige la consigna.
- En las Tareas B (XGBoost), los embeddings se extraen sin aumento de datos, con `XGBClassifier`
  (`n_estimators=300`, `max_depth=6`, `learning_rate=0.05`, *early stopping* sobre el set de
  validación).

### 2.3 Métricas de evaluación

Para cada escenario, evaluado sobre el conjunto de **test** (nunca visto durante entrenamiento
ni selección de hiperparámetros):

- Accuracy
- Precision macro y ponderada (weighted)
- Recall macro
- F1-score macro y ponderado
- Matriz de confusión y reporte por clase
- Tiempo de entrenamiento y número de parámetros entrenables (para discutir costo
  computacional vs. desempeño)

---

## 3. Resultados y discusión

### 3.1 Tabla comparativa de los 6 escenarios

_[Completar con los valores de `checkpoints/comparacion_metricas_TODOS_los_escenarios.csv`,
generado automáticamente al correr los tres notebooks de entrenamiento]_

| Modelo | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | F1 (weighted) | Tiempo entren. (min) | Parámetros entrenables (M) |
|---|---|---|---|---|---|---|---|
| ResNet-50 (CNN) | 0.8844 | 0.8284 | 0.8283 | 0.8176 | 0.8814 | 87.4 | 23.5654 |
| ViT-B/16 | 0.8253 | 0.7713 | 0.7853 | 0.7683 | 0.8226 | 88.4 | 85.8202 |
| Siamese-Contrastive + FC | | | | | | | |
| Siamese-Contrastive + XGBoost | | | | | | | |
| Siamese-Triplet + FC | 0.6704 | 0.6143 | 0.6688 | 0.6135 | 0.6560 | 91.67 | 0.0036 |
| Siamese-Triplet + XGBoost | 0.7183 | 0.6514 | 0.6657 | 0.6525 | 0.7202 | 0.14 | — |

_[Insertar gráfico de barras comparativo]_

### 3.2 CNN vs. ViT (fine-tuning end-to-end)

_[Comparar accuracy/F1, curvas de entrenamiento, tiempo de entrenamiento, matrices de
confusión y F1 por clase — ver sección 8 de `clasificacion_hojas_resnet_vs_vit.ipynb`.
Discutir: ¿cuál generaliza mejor?, ¿en qué clases falla más cada uno?, ¿justifica el ViT su
mayor costo computacional (si aplica)?]_

### 3.3 Redes siamesas: contrastive vs. triplet

**Triplet (resultados disponibles):** la red se entrenó 10 épocas (1659 tripletas/batch de
32, 53,072 tripletas de entrenamiento por época; ~1326 min de entrenamiento total en CPU). La
pérdida de entrenamiento bajó de forma consistente de 0.2166 (época 1) a 0.0954 (época 10), y
la pérdida de validación de 0.1662 a 0.1181 (mejor checkpoint en la época 9, val_loss = 0.1181).
Las distancias en el espacio de embeddings (par fijo de validación) muestran una separación
clara y estable entre clases desde las primeras épocas:

| Época | d(a,p) (misma clase) | d(a,n) (clase distinta) | *triplet-acc* (d(a,p) < d(a,n)) |
|---|---|---|---|
| 1 | 0.2294 | 1.3578 | 94.20 % |
| 5 | 0.2577 | 1.3780 | 96.10 % |
| 10 | 0.2271 | 1.3864 | 96.90 % |

La distancia intra-clase se mantuvo baja (~0.22-0.28) mientras que la inter-clase creció y se
estabilizó cerca de 1.38-1.39 (el máximo posible entre embeddings normalizados L2 es 2), lo que
indica que el triplet loss separó bien las clases en el espacio de embeddings, con una
*triplet-accuracy* de validación de ~97 % al final del entrenamiento.

**Contrastive (pendiente):** _[completar cuando termine de correr
`clasificacion_hojas_siamese_contrastive.ipynb`, y comparar aquí distancias intra/inter-clase,
curva de pérdida y tiempo de entrenamiento contra los valores de triplet de arriba]_

### 3.4 Clasificador FC vs. XGBoost sobre los mismos embeddings

**Triplet:** sobre exactamente los mismos embeddings (128-d, backbone congelado), XGBoost
superó a la capa FC en todas las métricas de test: accuracy 0.7183 vs. 0.6704 (+4.8 pp),
F1-macro 0.6525 vs. 0.6135 (+3.9 pp) y F1-weighted 0.7202 vs. 0.6560 (+6.4 pp). Además, el
costo de entrenamiento de XGBoost fue drásticamente menor (8.2 s, 132 iteraciones con *early
stopping*) frente a los ~92 min de la capa FC (10 épocas sobre 26,536 imágenes). Esto sugiere
que, para este espacio de embeddings, un clasificador no-lineal basado en árboles aprovecha
mejor la geometría de los embeddings triplet que una simple proyección lineal, y lo hace con
un costo de entrenamiento órdenes de magnitud menor (aunque la extracción de embeddings en sí
—forward pass del backbone— sí requiere GPU/tiempo considerable, no reflejado en el tiempo de
XGBoost).

**Contrastive:** _[completar cuando termine de correr
`clasificacion_hojas_siamese_contrastive.ipynb`]_

### 3.5 Comparación global: fine-tuning end-to-end vs. redes siamesas + clasificador congelado

_[Discutir el trade-off principal del trabajo: los escenarios 1-2 actualizan todos los pesos
del backbone durante el entrenamiento de clasificación, mientras que 3-6 aprenden primero un
espacio de embeddings (con una señal de entrenamiento distinta a la clasificación directa) y
luego solo entrenan una cabeza ligera sobre pesos congelados. ¿Compensa el costo extra de
entrenar la red siamesa? ¿En qué escenario conviene más este enfoque (p. ej. pocos datos,
necesidad de embeddings reutilizables para otras tareas)?]_

### 3.6 Análisis por clase

**Triplet + FC — clases más débiles (F1 en test):** `Tomato_Mosaic` (0.06), `Zucchini_Downy_Midew`
(0.07), `Zucchini_Angular_Leaf_Spot` (0.15), `Zucchini_Insect_Damage` (0.30),
`Zucchini_Iron_Chlorosis_Damage` (0.48). Mejores clases: `papaya_Yellow_Necrotic_Spots_Holes`
(1.00), `Zucchini_Yellow_Mosaic_Virus` (0.97), `Tomato_Downy Mildew` (0.95).

**Triplet + XGBoost — clases más débiles (F1 en test):** `Zucchini_Angular_Leaf_Spot` (0.12),
`Zucchini_Insect_Damage` (0.23), `Zucchini_Iron_Chlorosis_Damage` (0.28),
`Zucchini_Downy_Midew` (0.33), `Zucchini_Dry_Leaf` (0.35). Mejores clases:
`papaya_Yellow_Necrotic_Spots_Holes` (1.00), `Tomato_Downy Mildew` (0.95),
`Tomato_Healthy` (0.95).

En ambos clasificadores triplet, las clases más problemáticas son casi todas de **Zucchini**
(`Angular_Leaf_Spot`, `Insect_Damage`, `Downy_Midew`, `Iron_Chlorosis_Damage`), lo que sugiere
que los síntomas visuales de estas categorías de Zucchini se confunden entre sí en el espacio
de embeddings, independientemente del clasificador final usado. `Tomato_Mosaic` es la más
débil solo en FC (recall 0.03) pero mejora notablemente con XGBoost (F1 0.53), lo que indica
que el problema ahí es más del clasificador lineal que del embedding en sí. Falta revisar si
esto se relaciona con desbalance de clases (número de imágenes por clase, sección 2.1) y
comparar contra CNN/ViT/contrastive para ver si son las mismas clases difíciles en todos los
escenarios. _[completar con CNN, ViT y contrastive]_

---

## 4. Conclusiones

_[Redactar a partir de los hallazgos de la sección 3. Debe incluir, como mínimo:]_

- Cuál fue el mejor escenario en términos de F1-macro / accuracy en test, y por qué.
- Cuál fue el más eficiente en tiempo/recursos de entrenamiento.
- Si el enfoque de redes siamesas (contrastive/triplet) aportó ventajas frente al fine-tuning
  directo (CNN/ViT), y en qué condiciones lo recomendarían.
- Si contrastive loss o triplet loss produjo mejores embeddings para este dataset específico.
- Si XGBoost superó consistentemente a la capa FC sobre los mismos embeddings.
- Limitaciones del estudio (tamaño del dataset, desbalance de clases, cómputo disponible,
  número de épocas) y trabajo futuro sugerido.

---

## Referencias

_[Formato según lo solicitado por el curso — incluir al menos:]_

1. Artículo del dataset (*Data in Brief*) — _[completar]_
2. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. *CVPR*.
3. Dosovitskiy, A. et al. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. *ICLR*.
4. Hadsell, R., Chopra, S., & LeCun, Y. (2006). Dimensionality Reduction by Learning an Invariant Mapping. *CVPR*.
5. Schroff, F., Kalenichenko, D., & Philbin, J. (2015). FaceNet: A Unified Embedding for Face Recognition and Clustering. *CVPR*. (triplet loss)
6. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD*.

---

## Anexos

- Anexo A: capturas de las curvas de entrenamiento completas de cada escenario.
- Anexo B: matrices de confusión completas (28×28) de cada escenario.
- Anexo C: enlace al repositorio del proyecto (notebooks, checkpoints, dataset).
