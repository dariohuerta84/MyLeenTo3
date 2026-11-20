# Secuencia Multi-Vista 360° y Optimización Cuántica (Harvard HQI)

> **Caso de Uso:** Secuencia de 6 tomas orbitales de una escultura clásica tomadas desde diferentes ángulos de rotación.

## ¿Por qué estas imágenes son óptimas para MyLeenTo3?

- **Solapamiento visual superior al 50% entre tomas contiguas, ideal para emparejamiento SIFT con ratio de Lowe < 0.76.**
- **Permite demostrar el solver combinatorio QUBO-MWIS eliminando correspondencias falsas en escenas simétricas.**
- **Ejemplifica la recomendación de la brújula ActivePose NBV para planificar la siguiente toma óptima.**

## Enfoque Recomendado para Presentaciones
> Subir las 6 imágenes al endpoint /api/multiview-reconstruct y ver cómo se fusionan en una nube unificada 360° sin artefactos de deriva.

## Inventario de Imágenes y Telemetría Óptica

| Archivo | Resolución | Varianza Laplaciana | Evaluación de Calidad |
| :--- | :---: | :---: | :--- |
| [view_01_Aphrodite_of_Knido.jpg](file:///home/matias/Github/MyLeenTo3/sample_datasets/06_multivista_360_activepose/view_01_Aphrodite_of_Knido.jpg) | 6016 × 4016 px | 513.1 | Excelente (Sharp) |
| [view_02_Aphrodite_of_Knido.jpg](file:///home/matias/Github/MyLeenTo3/sample_datasets/06_multivista_360_activepose/view_02_Aphrodite_of_Knido.jpg) | 1150 × 2458 px | 78.8 | Moderada |
| [view_03_Aphrodite_of_Knido.jpg](file:///home/matias/Github/MyLeenTo3/sample_datasets/06_multivista_360_activepose/view_03_Aphrodite_of_Knido.jpg) | 2736 × 3648 px | 18.0 | Suave / Difusa |
| [view_04_Aphrodite_of_Knido.jpg](file:///home/matias/Github/MyLeenTo3/sample_datasets/06_multivista_360_activepose/view_04_Aphrodite_of_Knido.jpg) | 2648 × 4288 px | 120.2 | Moderada |
| [view_05_Aphrodite_of_Knido.jpg](file:///home/matias/Github/MyLeenTo3/sample_datasets/06_multivista_360_activepose/view_05_Aphrodite_of_Knido.jpg) | 1425 × 1900 px | 181.8 | Excelente (Sharp) |
| [view_06_Aphrodite_of_Knido.jpg](file:///home/matias/Github/MyLeenTo3/sample_datasets/06_multivista_360_activepose/view_06_Aphrodite_of_Knido.jpg) | 3456 × 5184 px | 54.9 | Moderada |

---
*Generado automáticamente para el laboratorio de verificación 3D MyLeenTo3.*