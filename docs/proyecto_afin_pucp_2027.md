# MyLeenTo3: Convirtiendo Fotos 2D Comunes en Escenas 3D Medibles con IA

**Postulación al Summer Camp PUCP en Inteligencia Artificial 2027**  
**Autor:** Matias Darío Huerta Cruz · 8vo ciclo de Ingeniería Informática, Universidad Peruana Cayetano Heredia (UPCH) · Miembro L.I.A. UPCH  
**Repositorio abierto:** `https://github.com/dariohuerta84/MyLeenTo3`

---

## 1. ¿Qué problema resuelve MyLeenTo3?
Cuando tomamos una fotografía normal con un celular o computadora, la imagen es plana (2D): podemos ver qué objetos aparecen, pero no sabemos a cuántos metros están, cuánto miden realmente ni qué objeto está tapando a otro. **MyLeenTo3** conecta **Google AI Studio** (`gemini-4-hydrogen`) con un **visor 3D interactivo (Three.js)** y un motor geométrico en Python para reconstruir la escena en 3D con medidas reales en metros, sin necesidad de sensores costosos como LiDAR.

## 2. ¿Cómo funciona paso a paso?
1. **Cualquier imagen 2D:** fotografías reales, renders 3D o capturas de videojuegos.
2. **IA Multimodal (Google AI Studio):** Gemini detecta los objetos principales y estima su ubicación, dimensiones y giro en 3D.
3. **Cálculo exacto en Python (cero cifras inventadas):** un Copiloto inteligente utiliza *Function Calling* para que Python calcule distancias en metros, colisiones y oclusiones de forma verificable.
4. **Visor 3D y Diagnóstico de Calidad:** el usuario explora la escena desde cualquier ángulo y detecta fallas físicas automáticamente.

## 3. ¿Qué hace innovador a este proyecto?
- **La IA no adivina la matemática:** la IA solo identifica los objetos; toda distancia, choque o rayo de oclusión lo calcula Python con geometría exacta.
- **Funciona con cualquier foto nueva (Diagnóstico Sin Referencia):** en fotos comunes sin medidas de laboratorio previas, evalúa si los bordes 3D coinciden con la foto real, si algún objeto parece "flotar" sobre el piso, si dos sólidos se atraviesan y cuánta incertidumbre hay por sombras o distancia (índice medio de plausibilidad física **`85.1/100`**).
- **Puente frente a videos generativos (como Google Genie / Veo):** evalúa si los objetos cambian de tamaño o tiemblan al mover la cámara entre cuadros consecutivos (estabilidad medida: **`58.5/100`**).

## 4. Resultados Medidos (`SEED = 42`, 3 Escenas Calibradas)

| Modelo de Gemini | Razonamiento | Precisión Volumen 3D | Vista Superior (BEV) | Error de Distancia | Error de Giro |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Gemini 4 Hydrogen** | Alto (high) | **69.9%** | 76.3% | 4.6 cm | 2.6° |
| **Gemini 4 Argon** | Medio (medium) | **59.3%** | 67.5% | 7.4 cm | 4.7° |
| **Gemini 3.5 Flash-Lite** | Rápido (low) | **36.7%** | 47.5% | 16.6 cm | 9.9° |

- **Ajuste del lente de la cámara:** con el ángulo calibrado en `60°` las cajas calzan al **`80.7%`** sobre la foto (frente a solo `2.6%` si se descalibra a `100°`), y el usuario puede corregirlo en tiempo real en menos de 10 ms.
- **Brecha frente a modelos más ligeros:** `Gemini 4 Hydrogen` logra **`4.6 cm`** de error medio de posición, reduciendo en más de 3 veces el error frente a `Flash-Lite` (`16.6 cm`) y superando también a `Argon` (`7.4 cm`).

## 5. Aporte al Summer Camp PUCP 2027
El proyecto une dos líneas activas del **Grupo IA-PUCP**: Visión por Computador 3D y Sistemas/Agentes Inteligentes aplicados con rigor verificable. Durante el campamento se busca extender la herramienta hacia fusión con mapas de profundidad densa y datasets públicos reales como *SUN RGB-D* y *ScanNet*.
