# SDV-VISION

Sistema de visión basado en ROS 2 diseñado para la percepción del entorno mediante detección de objetos y estimación de profundidad, con proyección a un espacio 3D (Bird's Eye View). El proyecto está concebido principalmente para su despliegue final en una NVIDIA Jetson Orin Nano (16GB), aprovechando la aceleración por hardware. Adicionalmente, cuenta con una rama estructurada para optimizar la ejecución en equipos que dependen exclusivamente del procesamiento por CPU.

## Getting Started

Estas instrucciones te permitirán obtener una copia del proyecto y configurarlo en tu máquina local para desarrollo y pruebas.

### Prerequisites

*   **ROS 2:** Humble.
*   **Python:** 3.10+
*   **Librerías de Python:** `ultralytics`, `opencv-python`, `numpy`.
*   **Paquetes de ROS 2:** `rclpy`, `sensor_msgs`, `vision_msgs`, `visualization_msgs`, `cv_bridge`, `message_filters`.

### Installing

1.  Clona este repositorio dentro de la carpeta `src` de tu espacio de trabajo (workspace) de ROS 2:
    ```bash
    git clone [https://github.com/Santiagortegab/SDV-VISION.git](https://github.com/Santiagortegab/SDV-VISION.git)
    ```
2.  Asegúrate de descargar y colocar los pesos preentrenados de YOLO (`yolo26n.pt` y `yolo26n-depth.pt`) en la ruta `../../weights/` relativa a la ejecución del nodo, o ajusta la ruta en el código fuente.
3.  Regresa a la raíz de tu espacio de trabajo y compila el paquete:
    ```bash
    colcon build --packages-select sdv_pkg
    ```
4.  Aplica las variables de entorno:
    ```bash
    source install/setup.bash
    ```

## Deployment Architecture

El ciclo de desarrollo y despliegue se gestiona a través de dos ramas principales para adaptarse al hardware de destino:

*   **Rama `master` (Despliegue Jetson Orin Nano):** Arquitectura distribuida que ejecuta la detección 2D (`yolo_detection.py`) y la estimación de profundidad (`yolo_depth.py`) en nodos separados. Esta estructura está ideada para sistemas con GPUs dedicadas y memoria unificada, donde los modelos se exportarán a motores de TensorRT para alcanzar inferencia en tiempo real.
*   **Rama `pruebas-cpu` (Optimización para Laptops):** Implementa un "Nodo de Percepción" unificado que procesa ambos modelos secuencialmente utilizando un único frame decodificado. Esta rama mitiga los cuellos de botella por concurrencia y evita la pérdida de mensajes al probar el sistema sin aceleración CUDA.

## Usage

Para inicializar el sistema de visión completo —incluyendo el feed de la dashcam, los nodos de inferencia, la transformación TF2 y la visualización en RViz2— ejecuta el archivo de lanzamiento:

```bash
ros2 launch sdv_pkg sdv.launch.py
```

### Componentes Principales

*   **Publicador de Video (`video_pub.py`):** Lee un archivo MP4 local y publica los frames en la red de ROS 2 emulando una cámara.
*   **Módulo YOLO (`yolo_detection.py` / `yolo_depth.py`):** Genera arreglos de detecciones 2D (`Detection2DArray`) y matrices flotantes (`32FC1`) para el mapa de profundidad.
*   **Generador BEV (`BEV.py`):** Utiliza `ApproximateTimeSynchronizer` para fusionar las bounding boxes con los recortes de profundidad, estimando la distancia absoluta (Z) de cada objeto y publicando marcadores 3D (`MarkerArray`) representativos de cada clase (Persona, Vehículo, Bicicleta) en el entorno virtual.

## Built With

*   [ROS 2 Humble](https://docs.ros.org/en/humble/) - Framework de robótica.
*   [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - Inferencia de detección y profundidad.
*   [OpenCV](https://opencv.org/) - Manipulación matricial de imágenes.

## Authors

*   **Santiago Burgueño Ortega** [Santiagortegab](https://github.com/Santiagortegab)
