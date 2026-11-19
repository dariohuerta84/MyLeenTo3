# MyLeenTo3: Percepción 3D Medible y Compuerta de Confiabilidad Edge → VLM

**Repositorio abierto:** `https://github.com/dariohuerta84/MyLeenTo3`  
**Proyecto predecesor (2D Edge):** `https://github.com/dariohuerta84/C8286-Proyecto1-Vision-Clasica-vs-CNN`

---

## 1. ¿Qué problema resuelve MyLeenTo3 y qué valor genera?
Cuando una cámara común toma una foto plana (2D), un sistema tradicional puede clasificar objetos, pero no sabe a cuántos metros están, cuál es su volumen real ni si un obstáculo tapa a otro. Usar escáneres LiDAR es costoso, pero confiar ciegamente en modelos de IA grandes para cada fotograma también resulta lento y riesgoso si la IA inventa medidas o falla cuando hay sombra u oclusión.

**MyLeenTo3** convierte cualquier fotografía 2D en una escena 3D medible en metros usando **Google AI Studio**, un **visor 3D interactivo (Three.js)** y un verificador físico en Python que funciona como **Compuerta Adaptativa Edge → VLM**: aprueba tiers rápidos cuando la escena es limpia y escala a razonamiento 3D profundo (o rechaza capturas inválidas) cuando detecta oclusión, rotaciones fuertes o incoherencia física.

## 2. Origen y Evolución desde C8286 (Computación Paralela 2D)
Este sistema nace como la fase 3D directa de [`C8286-Proyecto1-Vision-Clasica-vs-CNN`](https://github.com/dariohuerta84/C8286-Proyecto1-Vision-Clasica-vs-CNN), donde se demostró experimentalmente un límite clave de la visión 2D:
- En imágenes limpias, los descriptores clásicos paralelizados con memoria compartida (`POSIX SharedMemory`) logran **786.1 FPS**, y una `TinyCNN INT8` de **87.5 KB** alcanza **2,527.1 FPS**.
- Sin embargo, bajo ruido de iluminación o oclusión parcial, la visión clásica colapsa (`F1 <= 0.104`), `MobileNetV2 INT8` llega a `F1 = 0.715` en oclusión y una **cascada adaptativa Edge ↔ Servidor** reduce **97.5%** del tráfico enviando a la GPU solo los casos dudosos.
- **Salto en MyLeenTo3:** trasladamos esa cascada costo–precisión al espacio 3D métrico, sustituyendo etiquetas planas por cajas 3D en metros, oclusiones por trazado de rayos y un filtro automático de confiabilidad física sin requerir LiDAR.

## 3. Veredicto Operativo de la Compuerta Edge → VLM (Sin Ground Truth)

| Escena Evaluada | Régimen C8286 | Veredicto | Modelo Recomendado | Plausibilidad |
| :--- | :---: | :---: | :---: | :---: |
| Mesa de Laboratorio | `clean` | **APROBADO** | `gemini-4-argon` | 89.4/100 |
| Corredor con Oclusiones | `occlusion_bg` | **ESCALAR_MODELO** | `gemini-4-hydrogen` | 79.8/100 |
| Taller con Giros 3D | `clean` | **ESCALAR_MODELO** | `gemini-4-hydrogen` | 86.1/100 |

*Índice medio de plausibilidad física en escenas evaluadas: **`85.1/100`**. Estabilidad temporal en secuencia multi-cuadro: **`58.5/100`**.*

## 4. Resultados Medidos (`SEED = 42`, 3 Escenas Calibradas)

| Modelo de Gemini | Razonamiento | Precisión Volumen 3D | Vista Superior (BEV) | Error de Distancia | Error de Giro |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Gemini 4 Hydrogen** | Alto (high) | **69.9%** | 76.3% | 4.6 cm | 2.6° |
| **Gemini 4 Argon** | Medio (medium) | **59.3%** | 67.5% | 7.4 cm | 4.7° |
| **Gemini 3.5 Flash-Lite** | Rápido (low) | **36.7%** | 47.5% | 16.6 cm | 9.9° |

- **Ajuste del lente de la cámara:** con el ángulo calibrado en `60°` las cajas calzan al **`80.7%`** sobre la foto (frente a solo `2.6%` si se descalibra a `100°`), con corrección instantánea en menos de 10 ms.
- **Brecha frente a modelos ligeros:** `Gemini 4 Hydrogen` logra **`4.6 cm`** de error medio de posición, reduciendo en más de 3 veces el error frente a `Flash-Lite` (`16.6 cm`) y superando también a `Argon` (`7.4 cm`).
