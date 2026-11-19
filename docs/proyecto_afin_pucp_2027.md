# MyLeenTo3: Percepción Espacial 3D Monocular Calibrada, Agentes Determinísticos y Auditoría Geométrica Multi-Frame

**Postulación al Summer Camp PUCP en Inteligencia Artificial 2027**  
*Grupo de Investigación en Inteligencia Artificial — Pontificia Universidad Católica del Perú (IA-PUCP)*  
**Autor:** Matias Darío Huerta Cruz · Estudiante de 8vo ciclo de Ingeniería Informática — Universidad Peruana Cayetano Heredia (UPCH) · Miembro del Laboratorio de Inteligencia Artificial (L.I.A. — UPCH)  
**Repositorio reproducible:** `https://github.com/dariohuerta84/MyLeenTo3`  
**Fecha:** Noviembre 2026

---

## 1. Resumen Ejecutivo y Motivación Científica

La percepción espacial 3D a partir de una única imagen RGB monocular constituye uno de los problemas inversos más desafiantes en visión por computador debido a la ambigüedad escala-profundidad y la pérdida de información métrica al proyectar la escena al plano imagen. Aunque los modelos fundacionales visión-lenguaje (*Vision-Language Models*, VLMs) contemporáneos han demostrado capacidad *zero-shot* para predecir cajas delimitadoras orientadas de 9 grados de libertad (9-DOF OBB), su adopción en investigación reproducible se ve limitada por tres factores: (i) errores sistemáticos de proyección *pinhole* cuando el campo de visión horizontal ($\text{FOV}_h$) no se calibra analíticamente; (ii) inversión de quiralidad o desalineación de rotaciones al mapear las poses del marco de cámara hacia motores gráficos 3D; y (iii) alucinaciones aritméticas cuando el propio modelo de lenguaje intenta estimar distancias euclidianas, *clearance* o relaciones de oclusión en texto libre.

**MyLeenTo3** propone un banco de pruebas abierto y reproducible que articula dos líneas centrales del grupo **IA-PUCP**: **Visión por Computador 2D/3D** (geometría proyectiva exacta, evaluación volumétrica 3D/BEV y trazado de rayos de oclusión) e **Ingeniería del Conocimiento y Agentes** (acoplamiento neuro-simbólico entre la `Interactions API` de Google AI Studio y un núcleo geométrico determinístico en Python). En los tres *benchmarks* calibrados (`SEED = 42`, 16 objetos 3D y 12 relaciones de oclusión), `gemini-4-hydrogen` alcanza un promedio de **`mIoU_3D = 0.699`**, **`mIoU_BEV = 0.763`**, **`AP@0.25 = 1.000`** y **`ATE = 0.046 m`**, frente a `0.593` (`gemini-4-argon`) y `0.367` (`gemini-3.5-flash-lite`), mientras que el módulo de auditoría temporal cuantifica un índice de estabilidad geométrica de **`58.5 / 100`** en secuencias tipo *World Model*.

---

## 2. Formulación Matemática y Arquitectura de MyLeenTo3

### 2.1 Representación 9-DOF y Transformación Preservando Quiralidad
Siguiendo la convención oficial de cámara Gemini (**$X$ derecha, $Y$ profundidad hacia adelante, $Z$ arriba**), cada objeto se representa como $\mathbf{b} = [x, y, z, \Delta x, \Delta y, \Delta z, r, p, \psi] \in \mathbb{R}^9$ (metros y grados). Los ángulos de Euler intrínsecos ZYX $(r, p, \psi)$ se convierten a un cuaternión unitario $\mathbf{q} = (q_x, q_y, q_z, q_w)$ y matriz de rotación $R_{\text{gem}} \in SO(3)$.

Para renderizar sin espejado en **Three.js** ($X$ derecha, $Y$ arriba, $-Z$ profundidad), MyLeenTo3 aplica la transformación ortogonal pura $M = R_x(-90^\circ)$ con $\det(M) = +1$:

$$M = \begin{pmatrix} 1 & 0 & 0 \\ 0 & 0 & 1 \\ 0 & -1 & 0 \end{pmatrix}, \quad \mathbf{p}_{\text{three}} = M\mathbf{p}_{\text{gem}} = \begin{pmatrix} x \\ z \\ -y \end{pmatrix}, \quad R_{\text{three}} = M R_{\text{gem}} M^\top, \quad \mathbf{s}_{\text{three}} = (\Delta x, \Delta z, \Delta y)$$

### 2.2 Proyección Pinhole y Primitivas Geométricas Determinísticas
La reproyección 3D $\rightarrow$ 2D utiliza $R_{\text{view}} = R_x(+90^\circ)$ y matriz intrínseca $K$ con distancia focal $f = W / (2\tan(\text{FOV}_h / 2))$ y punto principal $(W/2, H/2)$. Las métricas se calculan íntegramente en Python (`NumPy`/`SciPy`):
- **$\text{IoU}_{\text{BEV}}$:** intersección poligonal exacta en el plano $XY$ mediante **Sutherland-Hodgman**.
- **$\text{IoU}_{\text{3D}}$:** volumen prismático exacto cuando $|r|,|p| < 2^\circ$; en cajas inclinadas, rechazo temprano mediante **SAT (15 ejes)** + muestreo estratificado determinístico (`SEED = 42`).
- **Oclusión 3D:** trazado de 9 rayos por objeto desde $(0,0,0)$ con intersección de losas **Kay-Kajiya** en espacio local OBB.
- **Orquestación Neuro-Simbólica:** el backend utiliza el SDK `google-genai >= 3.6.1` (`client.interactions.create` en `v1`) con *Structured Output* y un bucle multi-turno de *Function Calling* con 6 herramientas determinísticas (`focus_object`, `measure_distance`, `set_camera_view`, `analyze_occlusions_and_collisions`, `filter_objects`, `run_fov_sensitivity_sweep`), eliminando por diseño las alucinaciones aritméticas del LLM.

---

## 3. World Models Implícitos (Google Genie / Veo) vs. Percepción 3D Explícita

Modelos fundacionales de mundo como **Google DeepMind Genie** y **Veo** aprenden dinámicas latentes de video condicionadas a acciones para generar entornos interactivos fotorrealistas. Sin embargo, desde la perspectiva de la visión 3D métrica existen diferencias fundamentales: (1) Genie opera sobre representaciones implícitas en espacio de píxeles/latentes y no dispone de una API pública que retorne poses métricas 9-DOF ni mallas/OBBs explícitas; (2) la síntesis autorregresiva cuadro a cuadro induce deriva de volumen (*scale drift*) y oscilación angular (*yaw jitter*) en objetos rígidos al rotar la cámara.

| Dimensión | *World Models* Implícitos (Genie / Veo) | Percepción 3D Explícita (MyLeenTo3) |
| :--- | :--- | :--- |
| Representación interna | Dinámica latente de video / tokens visuales | Poses explícitas 9-DOF en $SE(3)$ y geometría OBB |
| Métricas espaciales | No expone distancias, *clearance* ni IoU 3D | Distancias exactas, SAT, Sutherland-Hodgman, Kay-Kajiya |
| Invariancia de cuerpo rígido | Susceptible a deriva de escala y *yaw jitter* | Auditable cuadro a cuadro (`geometric_stability_score`) |
| Disponibilidad técnica | Demostración cerrada / sin API pública 3D | API abierta (`Interactions API v1`) + núcleo offline reproducible |

Para tender un puente cuantitativo entre ambos paradigmas, MyLeenTo3 incorpora el módulo **`Multi-Frame Geometric Consistency Audit`** (`world_model_corridor_audit`, 4 cuadros de paneo de cámara), que mide deriva porcentual de escala, coeficiente de variación de volumen y *yaw jitter* por trayectoria.

---

## 4. Evaluación Experimental Reproducible (`SEED = 42`)

### 4.1 Resumen Agregado por Modelo (3 Benchmarks Calibrados, $960\times 640$, $\text{FOV}_h = 60^\circ$)

| Modelo | `mIoU_3D` | `mIoU_BEV` | `AP@0.25` | `AP@0.50` | `ATE (m)` | `AOE (°)` | `Reproj IoU 2D` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `gemini-4-hydrogen` | **0.699** | 0.763 | 1.000 | 0.920 | 0.046 | 2.57 | 0.832 |
| `gemini-4-argon` | **0.593** | 0.675 | 1.000 | 0.724 | 0.074 | 4.71 | 0.784 |
| `gemini-3.5-flash-lite` | **0.367** | 0.475 | 0.582 | 0.206 | 0.166 | 9.94 | 0.562 |

### 4.2 Desglose por Escena Calibrada e Intervalo de Confianza Bootstrap 95% ($N_{\text{boot}} = 1000$)

| Escena | Modelo | Thinking | `mIoU_3D` | `mIoU_BEV` | `AP@0.25` | `ATE (m)` | `AOE (°)` | IC 95% `mIoU_3D` |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Lab Desk | `gemini-4-hydrogen` | high | 0.662 | 0.726 | 1.000 | 0.043 | 1.83 | [0.559, 0.753] |
| Lab Desk | `gemini-4-argon` | medium | 0.616 | 0.687 | 1.000 | 0.055 | 4.97 | [0.507, 0.725] |
| Lab Desk | `gemini-3.5-flash-lite` | low | 0.351 | 0.461 | 0.440 | 0.133 | 8.07 | [0.212, 0.512] |
| Occlusion Corridor | `gemini-4-hydrogen` | high | 0.765 | 0.793 | 1.000 | 0.048 | 4.63 | [0.694, 0.818] |
| Occlusion Corridor | `gemini-4-argon` | medium | 0.582 | 0.642 | 1.000 | 0.097 | 5.06 | [0.500, 0.641] |
| Occlusion Corridor | `gemini-3.5-flash-lite` | low | 0.453 | 0.510 | 0.806 | 0.201 | 10.48 | [0.271, 0.591] |
| Rotated Workshop | `gemini-4-hydrogen` | high | 0.669 | 0.769 | 1.000 | 0.048 | 1.25 | [0.603, 0.729] |
| Rotated Workshop | `gemini-4-argon` | medium | 0.581 | 0.697 | 1.000 | 0.070 | 4.09 | [0.556, 0.610] |
| Rotated Workshop | `gemini-3.5-flash-lite` | low | 0.297 | 0.454 | 0.500 | 0.164 | 11.26 | [0.189, 0.407] |

### 4.3 Sensibilidad al $\text{FOV}_h$ y Auditoría Multi-Frame

- **Calibración $\text{FOV}_h$ (`Lab Desk · gemini-4-hydrogen`):** el ajuste proyectivo alcanza su máximo exacto en $\text{FOV}_h = 60^\circ$ con **`Reproj IoU 2D = 0.807`** y error medio de **`10.70 px`**. Una descalibración hacia $\text{FOV}_h = 100^\circ$ degrada el IoU 2D hasta **`0.026`** (`106.06 px` de error medio), confirmando la importancia de desacoplar la inferencia VLM del parámetro intrínseco $f$.
- **Estabilidad Multi-Frame (`world_model_corridor_audit`, 4 cuadros):** el puntaje compuesto de estabilidad geométrica es **`58.5 / 100`**, con una deriva media de escala de **`10.87%`** y *yaw jitter* medio de **`5.78°`**. Diagnóstico del auditor: *Deriva Moderada (Típica de World-Model Implícito): deriva volumétrica media 10.9% y jitter de orientación 5.8°/frame. Mayor deriva observada en 'transport_crate' (17.0% de cambio volumétrico, CV 5.9%).*

| Objeto (Track) | Cuadros | Deriva Escala (%) | CV Volumen (%) | Yaw Jitter (°) | Paso Centroide (m) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `calibration_table` | 4 | 4.50% | 2.22% | 3.30° | 0.163 |
| `equipment_rack` | 4 | 8.49% | 3.06% | 4.00° | 0.163 |
| `laser_bench` | 4 | 13.49% | 4.83% | 6.83° | 0.163 |
| `transport_crate` | 4 | 17.00% | 5.94% | 9.00° | 0.163 |

---

## 5. Limitaciones, Trabajo Futuro en el Summer Camp PUCP y Referencias

### Limitaciones Actuales
1. Los *benchmarks* calibrados con iluminación Lambertiana garantizan trazabilidad exacta (`SEED = 42`), pero no capturan ruido fotométrico severo, reflejos especulares ni desorden (*clutter*) de interiores reales.
2. En inferencia monocular sin calibración previa persiste la ambigüedad fundamental entre tamaño físico del objeto y distancia focal $f$.
3. La auditoría multi-cuadro actual evalúa trayectorias cuadro a cuadro sin todavía optimizar conjuntamente la pose de cámara y objetos mediante *Bundle Adjustment* o filtros de Kalman en $SE(3)$.

### Agenda de Investigación Propuesta para el Summer Camp PUCP 2027
1. **Fusión con *priors* de profundidad monocular métrica:** combinar la semántica 9-DOF de VLMs con mapas de profundidad densa (*Metric3D v2* / *Depth Anything V2*) para reducir el error $\text{ATE}$ en objetos a más de $4\text{ m}$ y bajo oclusión parcial.
2. **Auto-calibración de intrínsecos $K$:** estimación automática de $\text{FOV}_h$ y horizonte mediante geometría de puntos de fuga antes de la reproyección *pinhole*.
3. **Evaluación en datasets reales anotados:** extender el protocolo reproducible (emparejamiento Húngaro + *bootstrap* 95%) hacia **SUN RGB-D**, **ScanNet** y **ARKitScenes**, incorporando restricciones físicas de soporte y no-interpenetración.

### Referencias
1. Google AI for Developers. *Gemini Spatial Understanding 3D & Interactions API Documentation* (`google-genai` SDK v3.6+, 2025–2026).
2. Parker-Holder, J. et al. *Genie 2: A Large-Scale Foundation World Model*. Google DeepMind Technical Report (2024).
3. Song, S., Lichtenberg, S. P., & Xiao, J. *SUN RGB-D: A RGB-D Scene Understanding Benchmark Suite*. Proc. IEEE/CVF CVPR (2015), pp. 567–576.
4. Dai, A. et al. *ScanNet: Richly-Annotated 3D Reconstructions of Indoor Scenes*. Proc. IEEE/CVF CVPR (2017), pp. 5828–5839.
5. Kuhn, H. W. *The Hungarian Method for the Assignment Problem*. Naval Research Logistics Quarterly, 2(1–2), 83–97 (1955).
6. Sutherland, I. E., & Hodgman, G. W. *Reentrant Polygon Clipping*. Communications of the ACM, 17(1), 32–42 (1974).
7. Kay, T. L., & Kajiya, J. T. *Ray Tracing Complex Scenes*. ACM SIGGRAPH Computer Graphics, 20(4), 269–278 (1986).
