# Catálogo de Publicaciones y Portafolio de Proyectos de GitHub
**Autor:** Darío Matías Huerta Quispe  
**Perfil Oficial:** [github.com/dariohuerta84](https://github.com/dariohuerta84)  
**Fecha de Actualización:** Noviembre 2026  

---

> Este documento contiene el resumen técnico, la propuesta de valor y las **publicaciones redactadas y optimizadas** (listas para LinkedIn, X/Twitter, Dev.to o presentaciones académicas) para cada uno de los proyectos alojados en el ecosistema GitHub de Darío Huerta.

---

## Índice General de Proyectos

1. [MyLeenTo3 · Percepción Espacial 3D y Optimización Cuántica (Harvard HQI)](#1-myleento3)
2. [OdysMedic · Plataforma de Entrenamiento Clínico Adaptativo (UPCH)](#2-odysmedic)
3. [C8286 · Visión Clásica Paralela vs CNN (CUDA RTX / TFLite INT8)](#3-c8286-visión-clásica-vs-cnn)
4. [QUIM-DETECT2 · Detección de Pesticidas por Espectroscopía VIS/NIR](#4-quim-detect2)
5. [AURA · Avatares 3D con TripoSR y Gamificación Conductual](#5-aura)
6. [HeredIAns / RutaSegura · Agente de IA Vial para MTC / SUTRAN con Gemma](#6-heredians--rutasegura)
7. [Motor Tarifario Inteligente para inDrive (Lima Metropolitana)](#7-motor-tarifario-inteligente-indrive)
8. [Archlinux · Entorno Caelestia Shell y Arquitectura Multi-GPU](#8-archlinux--caelestia-shell)
9. [GESTIÓN HINTON · MLOps y Contenerización Docker en Servidores de Cómputo](#9-gestión-hinton)
10. [Proyecto-Calidad · Sistema de Auditoría y Control de Calidad](#10-proyecto-calidad)
11. [SIVAPRE (v3) · Sistema de Valoración Clínica y Triaje Preventivo](#11-sivapre-v3)
12. [DeliverySystem · Plataforma de Logística y Rutas de Despacho](#12-deliverysystem)

---

### 1. MyLeenTo3
* **Repositorio:** [github.com/dariohuerta84/MyLeenTo3](https://github.com/dariohuerta84/MyLeenTo3)
* **Área:** Visión por Computadora 3D · Metrología Óptica · Computación Cuántica Combinatoria
* **Stack:** Python, PyTorch (MoGe-2 ViT-L), OpenCV (ArUco / SIFT), React, Three.js, WebGL, FastAPI, QuEra/Braket QUBO.

#### Resumen Técnico
MyLeenTo3 transforma cualquier cámara RGB estándar en un instrumento metrológico y auditor físico tridimensional con precisión milimétrica sin sensores LiDAR. Infiere nubes surfel densas, calcula cuboides orientados 9-DOF, evalúa el Teorema del Eje Separador (SAT en 15 ejes) para descartar colisiones y valida la estabilidad gravitacional (GFI) sobre planos RANSAC. En su Fase 4, incorpora un solver combinatorio QUBO-MWIS inspirado en las investigaciones de átomos neutros (Rydberg) de la Universidad de Harvard (HQI), resolviendo el registro multi-vista 360° inmune a falsos consensos en piezas simétricas.

#### Texto Listo para Publicación (LinkedIn / Portafolio)
> 🚀 **¿Es posible lograr metrología 3D y control de calidad milimétrico con un teléfono móvil y $0 en sensores LiDAR?**
>
> Presento **MyLeenTo3**, un laboratorio de percepción espacial 3D y auditoría física en el borde (Edge GPU) que traslada los principios de la física y la optimización combinatoria a la visión por computadora:
>
> 🔹 **Reconstrucción Surfel y Normales Monoculares:** Estimación de hasta 80,000 elementos de superficie y relieve a partir de una única imagen RGB mediante MoGe-2.
> 🔹 **Auditoría Física Inquebrantable:** Validación de no penetración mediante el Teorema del Eje Separador (SAT 15-ejes) y verificación de sustentación gravitatoria (GFI).
> 🔹 **Calibración Métrica Absoluta:** Integración con marcadores ArUco y tarjetas estándar ISO para mediciones reales en milímetros.
> 🔹 **Optimización Cuántica Inspirada en Harvard (HQI):** Mapeo del registro 360° a problemas QUBO / Maximum Independent Set (MWIS), resolviendo la fusión multi-vista sin desviaciones.
>
> Código abierto y ejecutable 100% On-Premise para preservar la privacidad médica e industrial:
> 👉 https://github.com/dariohuerta84/MyLeenTo3
>
> #ComputerVision #3DVision #MachineLearning #QuantumComputing #Metrology #OpenSource #Python

---

### 2. OdysMedic
* **Repositorio:** [github.com/dariohuerta84/OdysMedic](https://github.com/dariohuerta84/OdysMedic)
* **Área:** Salud Digital · Educación Médica Adaptativa · EdTech
* **Stack:** TypeScript, React, Next.js, Node.js, Tailwind CSS, PostgreSQL.

#### Resumen Técnico
Desarrollado en el marco del curso de Proyectos de Ingeniería 2 de la Universidad Peruana Cayetano Heredia (UPCH), OdysMedic es una plataforma de simulación clínica adaptativa. Evalúa de manera retrospectiva y en tiempo real el razonamiento diagnóstico de estudiantes de medicina e internos, clasificando sesgos cognitivos (cierre prematuro, heurística de disponibilidad) y modulando la complejidad de los casos clínicos según el desempeño individual.

#### Texto Listo para Publicación
> 🩺 **Reduciendo el error diagnóstico en la formación médica mediante tecnología adaptativa: OdysMedic**
>
> El error diagnóstico es una de las principales causas de eventos adversos en salud. Para abordar este desafío, desarrollamos **OdysMedic** en la Universidad Peruana Cayetano Heredia (UPCH): una plataforma de entrenamiento clínico interactivo que evalúa no solo el resultado del diagnóstico, sino el *proceso de toma de decisiones del médico en formación*.
>
> 💡 Mediante una taxonomía de brechas cognitivas, la plataforma detecta patrones de cierre prematuro y anclaje heurístico, adaptando la dificultad del caso de forma personalizada.
>
> Explora el repositorio:
> 👉 https://github.com/dariohuerta84/OdysMedic
>
> #HealthTech #EdTech #Medicina #SistemasDeSalud #SoftwareEngineering #UPCH

---

### 3. C8286 · Visión Clásica vs CNN
* **Repositorio:** [github.com/dariohuerta84/C8286-Proyecto1-Vision-Clasica-vs-CNN](https://github.com/dariohuerta84/C8286-Proyecto1-Vision-Clasica-vs-CNN)
* **Área:** Computación Paralela y Distribuida · Sistemas Embebidos · Benchmarking GPU
* **Stack:** Python, multiprocessing (loky / POSIX SharedMemory), MobileNetV2, PyTorch CUDA, NVIDIA NVML, TFLite INT8, OpenCV.

#### Resumen Técnico
Estudio comparativo y de ingeniería de alto rendimiento desarrollado en el curso de Computación Paralela y Distribuida (UPCH). Enfrenta pipelines clásicos de visión computacional altamente paralelizados en CPU multi-core (segmentación morfológica, contornos, memoria compartida POSIX sin sobrecosto de serialización) contra redes neuronales convolucionales profundas (MobileNetV2) ejecutadas en GPU NVIDIA GeForce RTX 3050 Laptop bajo CUDA, cuantización INT8 y cascadas adaptativas de inferencia. Mide consumo energético (W), temperatura, ocupación de VRAM y latencia por cuadro.

#### Texto Listo para Publicación
> ⚡ **¿Cuándo conviene una red neuronal profunda frente a algoritmos de visión clásica paralelizados en CPU?**
>
> En este proyecto de Computación Paralela y Distribuida (UPCH), analizamos la compensación (*trade-off*) entre latencia, precisión y eficiencia energética:
>
> 📊 **Resultados clave:**
> 1. Pipelines clásicos optimizados con `POSIX SharedMemory` y `loky` reducen el overhead de IPC a casi cero, superando en latencia a modelos CNN pesados para tareas de inspección controlada.
> 2. MobileNetV2 con cuantización `TFLite INT8` y telemetría de GPU en tiempo real con `NVIDIA NVML` demostró el punto óptimo de eficiencia por vatio consumido en hardware de borde (RTX 3050).
>
> Revisa la metodología, código y benchmarks completos:
> 👉 https://github.com/dariohuerta84/C8286-Proyecto1-Vision-Clasica-vs-CNN
>
> #ParallelComputing #HPC #CUDA #ComputerVision #DeepLearning #Optimization

---

### 4. QUIM-DETECT2
* **Repositorio:** [github.com/dariohuerta84/QUIM-DETECT2](https://github.com/dariohuerta84/QUIM-DETECT2)
* **Área:** Espectroscopía VIS/NIR · Quimiometría · Inocuidad Alimentaria
* **Stack:** Python, Jupyter Notebook, SciPy, Scikit-learn, PCA, PLS-DA, Baseline Correction (SNV/MSC).

#### Resumen Técnico
Sistema portátil para la detección no destructiva de residuos de pesticidas en fresas mediante firmas espectrales en el rango Visible e Infrarrojo Cercano (VIS/NIR: 400 nm – 1000 nm). Integra preprocesamiento espectral riguroso (Derivadas Savitzky-Golay, Standard Normal Variate - SNV) y modelos multivariados supervisados (PLS-DA y SVM) para discriminar muestras tratadas y muestras de control en campo sin destruir la fruta.

#### Texto Listo para Publicación
> 🍓 **Inocuidad alimentaria y espectroscopía NIR: Detección no invasiva de pesticidas en cultivos con QUIM-DETECT2**
>
> Los métodos químicos tradicionales (cromatografía HPLC/GC-MS) para medir pesticidas en alimentos tardan días y destruyen el producto. Con **QUIM-DETECT2**, implementamos un pipeline quimiométrico que analiza la reflectancia espectral en el rango VIS/NIR:
>
> 🔬 Mediante algoritmos de preprocesamiento espectral (SNV, Savitzky-Golay) y clasificación multivariada (PLS-DA), el sistema identifica firmas anómalas de pesticidas en segundos sin alterar la fruta.
>
> Repositorio del proyecto:
> 👉 https://github.com/dariohuerta84/QUIM-DETECT2
>
> #Chemometrics #DataScience #Spectroscopy #Agriculture #MachineLearning #FoodSafety

---

### 5. AURA
* **Repositorio:** [github.com/dariohuerta84/AURA](https://github.com/dariohuerta84/AURA)
* **Área:** Hábitos Conductuales · Modelado Generativo 3D · Gamificación
* **Stack:** JavaScript, React, Three.js, Python, TripoSR, FastAPI, WebGL.

#### Resumen Técnico
AURA es una plataforma interactiva de bienestar que combate la procrastinación transformando el progreso de hábitos cotidianos en una experiencia visual inmersiva. A partir de una única fotografía del usuario, un pipeline generativo acelerado por GPU genera en segundos un avatar tridimensional estilizado mediante TripoSR, el cual evoluciona visualmente de acuerdo con el cumplimiento de objetivos del usuario.

#### Texto Listo para Publicación
> 🎮 **Transformando hábitos y productividad en avatares 3D interactivos: Conoce AURA**
>
> ¿Cómo convertir la disciplina diaria en una experiencia tangible y gratificante?
>
> **AURA** utiliza inteligencia artificial generativa 3D para materializar el progreso personal:
> 👤 A partir de una sola fotografía, el motor genera un modelo tridimensional personalizado con **TripoSR** en segundos.
> 📈 Cada hábito completado nutre y transforma el entorno y estado visual del avatar en un visor WebGL en tiempo real.
>
> Código y arquitectura interactiva:
> 👉 https://github.com/dariohuerta84/AURA
>
> #GenerativeAI #3D #ThreeJS #Productivity #WebGL #FullStack

---

### 6. HeredIAns / RutaSegura
* **Repositorio:** [github.com/dariohuerta84/HeredIAns](https://github.com/dariohuerta84/HeredIAns)
* **Área:** Inteligencia Artificial Aplicada a Políticas Públicas · Seguridad Vial
* **Stack:** Python, Google Gemma (VLM / LLM), Function Calling, GeoPandas, FastAPI.

#### Resumen Técnico
Desarrollado para el análisis y mitigación de siniestros viales en el Perú, RutaSegura es un agente conversacional basado en la familia de modelos abiertos Gemma de Google. Permite a funcionarios del Ministerio de Transportes y Comunicaciones (MTC), SUTRAN y gobiernos regionales consultar en lenguaje natural sobre puntos críticos de accidentabilidad. El agente ejecuta *tool use / function calling* contra bases de datos geoespaciales para ponderar frecuencia, gravedad y horarios de siniestros, sugiriendo intervenciones prioritarias de infraestructura.

#### Texto Listo para Publicación
> 🚦 **IA para la toma de decisiones en seguridad vial: RutaSegura**
>
> En el transporte público y las carreteras nacionales, priorizar los recursos de fiscalización e infraestructura salva vidas.
>
> Desarrollamos **RutaSegura**, un agente impulsado por **Gemma** que traduce datos complejos de siniestralidad del MTC y SUTRAN en respuestas accionables mediante lenguaje natural:
>
> 🗺️ Con *Function Calling* y análisis geoespacial, el funcionario puede consultar directamente: *"¿Cuáles son los 3 tramos de la Panamericana Sur con mayor tasa de siniestros en horario nocturno?"* y obtener la justificación analítica en segundos.
>
> Código abierto del proyecto:
> 👉 https://github.com/dariohuerta84/HeredIAns
>
> #GovTech #Safety #Gemma #AI #OpenData #DataScience

---

### 7. Motor Tarifario Inteligente (inDrive)
* **Repositorio:** `Motor-Tarifario-Inteligente-para-inDrive-en-Lima-Metropolitana`
* **Área:** Movilidad Urbana · Algoritmos de Precios Dinámicos · Machine Learning Geoespacial
* **Stack:** Python, XGBoost, Scikit-learn, OSMnx, Folium, Pandas, NumPy.

#### Resumen Técnico
Modelo analítico y predictivo diseñado para el ecosistema de transporte por aplicativo en Lima Metropolitana. Modela la puja tarifaria bilateral entre conductor y pasajero, integrando variables exógenas como congestión vial en tiempo real, hora pico, zonas de riesgo y distancia de ruta calculada sobre la red vial de OpenStreetMap.

#### Texto Listo para Publicación
> 🚗 **Optimización tarifaria y movilidad urbana en Lima: Modelo Predictivo para inDrive**
>
> El modelo de negociación bilateral de viajes plantea un reto fascinante de ciencia de datos: ¿cuál es el precio justo que equilibra la aceptación del conductor y la satisfacción del usuario?
>
> Desarrollé un **Motor Tarifario Inteligente** que predice bandas de precios recomendadas analizando la red vial de Lima Metropolitana, distritos de origen/destino y congestión en tiempo real con modelos de Gradient Boosting (`XGBoost`).
>
> #DataScience #MachineLearning #Mobility #Lima #SmartCities #Algorithms

---

### 8. Archlinux / Caelestia Shell
* **Repositorio:** [github.com/dariohuerta84/Archlinux](https://github.com/dariohuerta84/Archlinux)
* **Área:** Sistemas Operativos · Entornos de Escritorio Wayland · Linux Kernel
* **Stack:** Arch Linux, Hyprland, Quickshell (Qt6 / QML), Bash, Caelestia Shell, NVIDIA PRIME.

#### Resumen Técnico
Configuración de bajo nivel y dotfiles reproducibles para estaciones de trabajo basadas en Arch Linux sobre laptops de arquitectura híbrida (AMD HawkPoint1 iGPU + NVIDIA GeForce RTX 3050 dGPU). Integra el entorno modular **Caelestia Shell** con renderizado QML, gestión de multiplexación de hardware con `AQ_DRM_DEVICES` y scripts automatizados de compilación y rescate de paquetes AUR.

#### Texto Listo para Publicación
> 🐧 **Ingeniería de entornos Linux de alto rendimiento: Mi configuración de Arch Linux + Hyprland + Caelestia Shell**
>
> Para desarrollo intensivo en visión computacional y CUDA, contar con un sistema operativo ligero, predecible y estéticamente refinado marca la diferencia.
>
> En este repositorio mantengo la infraestructura como código de mi entorno en Arch Linux:
> ⚙️ Configuración multi-GPU sin conflictos para inferencia local con PyTorch.
> 🎨 Interfaz gráfica moderna y ligera sobre Wayland con Caelestia Shell y QuickShell.
> 🛡️ Respaldo y mantenimiento automatizado del sistema.
>
> Repositorio de dotfiles:
> 👉 https://github.com/dariohuerta84/Archlinux
>
> #Linux #ArchLinux #Hyprland #Wayland #DevOps #SystemAdministration

---

### 9. GESTIÓN HINTON
* **Repositorio:** [github.com/dariohuerta84/gestion_hinton](https://github.com/dariohuerta84/gestion_hinton)
* **Área:** MLOps · Infraestructura de Servidores · Contenedores Docker
* **Stack:** Docker, Docker Compose, NVIDIA Container Toolkit, Linux Server, SSH, Bash.

#### Resumen Técnico
Marco de estandarización y administración de entornos de experimentación de Machine Learning y Visión por Computadora para los servidores de cómputo HINTON 1 y HINTON 2. Define imágenes Docker desacopladas con soporte CUDA para aislamiento de dependencias entre investigadores, gestión de puertos y asignación equitativa de recursos GPU.

#### Texto Listo para Publicación
> 🐳 **Estandarizando entornos de Machine Learning en servidores de investigación: Proyecto GESTIÓN HINTON**
>
> Uno de los mayores cuellos de botella en laboratorios de IA universitarios es el conflicto de versiones de CUDA, PyTorch y dependencias entre investigadores que comparten un servidor común.
>
> Con **GESTIÓN HINTON**, implementamos una infraestructura reproducible basada en Docker y NVIDIA Container Toolkit para los servidores del laboratorio:
> 🔒 Ambientes aislados por investigador con acceso seguro a GPU.
> 🚀 Despliegue de contenedores en un solo comando sin ensuciar el sistema host.
>
> 👉 https://github.com/dariohuerta84/gestion_hinton
>
> #MLOps #Docker #MachineLearning #DeepLearning #DevOps #GPUComputing

---

### 10. Proyecto-Calidad
* **Repositorio:** [github.com/dariohuerta84/Proyecto-Calidad](https://github.com/dariohuerta84/Proyecto-Calidad)
* **Área:** Gestión de Calidad · Desarrollo Web de Auditoría
* **Stack:** HTML5, CSS3, JavaScript, Bootstrap.

#### Resumen Técnico
Plataforma orientada al aseguramiento de la calidad en procesos organizacionales. Proporciona herramientas interactivas para el seguimiento de métricas operativas, control documental, checklist de auditorías y cálculo de indicadores de rendimiento (KPIs).

#### Texto Listo para Publicación
> 📊 **Aseguramiento de Calidad y Control Operacional Digitalizado: Proyecto-Calidad**
>
> La gestión de calidad requiere trazabilidad, claridad de métricas y cumplimiento normativo sin fricciones. Desarrollé esta plataforma para centralizar la auditoría de procesos y la visualización de indicadores clave de rendimiento (KPIs) en una interfaz web ágil y accesible.
>
> 👉 https://github.com/dariohuerta84/Proyecto-Calidad
>
> #QualityManagement #WebDevelopment #SoftwareEngineering #KPIs

---

### 11. SIVAPRE (v3)
* **Repositorio:** `sivapre-v3` / [github.com/dariohuerta84/SIVAPRE](https://github.com/dariohuerta84/SIVAPRE)
* **Área:** Sistemas Clínicos · Triaje y Valoración Preventiva
* **Stack:** TypeScript, React, Tailwind CSS, Node.js.

#### Resumen Técnico
Sistema integral de valoración médica preventiva y triaje. Digitaliza formularios clínicos de evaluación de riesgo cardiovascular, comorbilidades y estilos de vida, automatizando el cálculo de scores clínicos estandarizados para apoyar la toma de decisiones del personal de salud en atención primaria.

#### Texto Listo para Publicación
> 🏥 **Digitalización del triaje y la prevención clínica en atención primaria: SIVAPRE**
>
> Identificar a tiempo los factores de riesgo en el primer nivel de atención es fundamental para prevenir complicaciones mayores. **SIVAPRE** digitaliza la evaluación clínica del paciente, calculando de forma automatizada los índices de riesgo y facilitando una derivación oportuna y protocolizada.
>
> #HealthTech #DigitalHealth #ClinicalDecisionSupport #PrimaryCare

---

### 12. DeliverySystem
* **Repositorio:** [github.com/dariohuerta84/DeliverySystem](https://github.com/dariohuerta84/DeliverySystem)
* **Área:** Logística · Sistemas de Despacho Web
* **Stack:** HTML5, CSS3, JavaScript, REST APIs.

#### Resumen Técnico
Plataforma web para la gestión de despachos, pedidos y seguimiento de entregas en tiempo real. Diseñada para comercios y servicios locales que requieren control del estado de envíos, asignación de repartidores y confirmación de recepción.

#### Texto Listo para Publicación
> 📦 **Gestión logística y control de despachos en tiempo real: DeliverySystem**
>
> Una solución web ágil y directa para gestionar órdenes, monitorear el flujo de repartos y asegurar la trazabilidad del pedido desde su preparación hasta la entrega final al cliente.
>
> 👉 https://github.com/dariohuerta84/DeliverySystem
>
> #Logistics #WebDevelopment #Frontend #DeliverySolutions
