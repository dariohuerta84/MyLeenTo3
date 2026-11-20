# Dianas de Calibración Métrica ArUco y Estándares ISO

> **Caso de Uso:** Marcadores ArUco de alta resolución (Dict 4x4 y Dict 5x5) y tarjeta estándar ISO/IEC 7810 ID-1 (85.60 × 53.98 mm) listos para imprimir o proyectar.

## ¿Por qué estas imágenes son óptimas para MyLeenTo3?

- **Bordes de contraste absoluto (100% blanco/negro) con reglas milimétricas graduadas incorporadas.**
- **Detección inmediata por cv2.aruco con supresión de falsos positivos en el soporte RANSAC.**
- **Proporciona la constante métrica de escala k_scale = D_real / D_est para anclar milímetros exactos.**

## Enfoque Recomendado para Presentaciones
> Imprimir o mostrar en una tablet este marcador junto a cualquier objeto para forzar a MyLeenTo3 a calibrar en escala métrica milimétrica real.

## Inventario de Imágenes y Telemetría Óptica

| Archivo | Resolución | Varianza Laplaciana | Evaluación de Calidad |
| :--- | :---: | :---: | :--- |
| [aruco_dict4x4_id0_50mm.png](file:///home/matias/Github/MyLeenTo3/sample_datasets/07_calibracion_metrica_aruco_fiducials/aruco_dict4x4_id0_50mm.png) | 1100 × 1200 px | 1157.0 | Excelente (Sharp) |
| [aruco_dict4x4_id1_100mm.png](file:///home/matias/Github/MyLeenTo3/sample_datasets/07_calibracion_metrica_aruco_fiducials/aruco_dict4x4_id1_100mm.png) | 1100 × 1200 px | 1364.7 | Excelente (Sharp) |
| [aruco_dict5x5_id42_75mm.png](file:///home/matias/Github/MyLeenTo3/sample_datasets/07_calibracion_metrica_aruco_fiducials/aruco_dict5x5_id42_75mm.png) | 1100 × 1200 px | 1381.2 | Excelente (Sharp) |
| [iso_standard_card_85x54mm.png](file:///home/matias/Github/MyLeenTo3/sample_datasets/07_calibracion_metrica_aruco_fiducials/iso_standard_card_85x54mm.png) | 1000 × 700 px | 669.0 | Excelente (Sharp) |

---
*Generado automáticamente para el laboratorio de verificación 3D MyLeenTo3.*