# Catálogo de Datasets de Prueba por Temática · MyLeenTo3

> **Propósito:** Colección organizada de imágenes de prueba para demostraciones, validaciones técnicas y presentaciones del laboratorio de percepción espacial 3D **MyLeenTo3**.

Cada subcarpeta contiene ejemplos reales optimizados para los umbrales algorítmicos del motor (nitidez Laplaciana, fotometría anti-glare, segmentación YOLOE y fusión multi-vista QUBO-MWIS).

---

## Índice de Temáticas y Carpetas

| Carpeta | Caso de Uso / Proyecto | N° Archivos | Propósito Principal en Presentación |
| :--- | :--- | :---: | :--- |
| [`01_biomedico_y_ortopedia`](file:///home/matias/Github/MyLeenTo3/sample_datasets/01_biomedico_y_ortopedia) | **Biomédico, Ortopedia y Tele-Monitoreo Tisular** | 6 | Mostrar la malla densa 3D de una extremidad o modelo dental, las curvas de nivel... |
| [`02_inspeccion_industrial_y_mecanica`](file:///home/matias/Github/MyLeenTo3/sample_datasets/02_inspeccion_industrial_y_mecanica) | **Inspección Industrial, Metrología y Mecánica** | 6 | Comprobar el ajuste del cuboide 9-DOF, las holguras entre componentes mecánicos ... |
| [`03_logistica_y_cubitaje_paqueteria`](file:///home/matias/Github/MyLeenTo3/sample_datasets/03_logistica_y_cubitaje_paqueteria) | **Logística, Cubitaje Volumétrico y Paquetería** | 6 | Demostrar la cubificación instantánea (largo × ancho × alto en cm) y el cálculo ... |
| [`04_peritaje_forense_y_siniestros`](file:///home/matias/Github/MyLeenTo3/sample_datasets/04_peritaje_forense_y_siniestros) | **Peritaje Forense, Aseguradoras y Daño Estructural** | 6 | Mostrar el mapa de calor de deformación y la profundidad máxima del impacto en m... |
| [`05_patrimonio_arqueologia_y_arte`](file:///home/matias/Github/MyLeenTo3/sample_datasets/05_patrimonio_arqueologia_y_arte) | **Patrimonio Cultural, Arqueología y Piezas de Museo** | 6 | Visualizar la nube surfel densa de una escultura antigua, rotándola en tiempo re... |
| [`06_multivista_360_activepose`](file:///home/matias/Github/MyLeenTo3/sample_datasets/06_multivista_360_activepose) | **Secuencia Multi-Vista 360° y Optimización Cuántica (Harvard HQI)** | 6 | Subir las 6 imágenes al endpoint /api/multiview-reconstruct y ver cómo se fusion... |
| [`07_calibracion_metrica_aruco_fiducials`](file:///home/matias/Github/MyLeenTo3/sample_datasets/07_calibracion_metrica_aruco_fiducials) | **Dianas de Calibración Métrica ArUco y Estándares ISO** | 4 | Imprimir o mostrar en una tablet este marcador junto a cualquier objeto para for... |

---

## Cómo Usar Estas Imágenes en la Presentación del Laboratorio

1. **Modo Monocular (1 Imagen):** Arrastra cualquier imagen de `01`, `02`, `03`, `04` o `05` a la zona de carga de MyLeenTo3 (`http://localhost:5173` o `http://localhost:8085`). Observa la segmentación 2D, el ajuste del cuboide 9-DOF, el plano RANSAC y la nube de surfels 3D.
2. **Modo Calibración Métrica Real:** Abre en tu teléfono la imagen `aruco_dict4x4_id0_50mm.png` de `07_calibracion_metrica_aruco_fiducials` y colócala al lado de un objeto al tomar la foto. MyLeenTo3 detectará la escala métrica automáticamente.
3. **Modo Multi-Vista 360°:** Sube las 6 vistas contenidas en `06_multivista_360_activepose` de forma simultánea. El motor ejecutará el emparejamiento SIFT, filtrado QUBO-MWIS y registro ICP cerrado.