[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/o8XztwuW)
# Proyecto Final 2025-1: AI Neural Network
## **CS2013 Programación III** · Informe Final

### **Descripción**

> Implementación de una red neuronal multicapa en C++ para la predicción de enfermedades cardíacas.

### Contenidos

1. [Datos generales](#datos-generales)
2. [Requisitos e instalación](#requisitos-e-instalación)
3. [Investigación teórica](#1-investigación-teórica)
4. [Diseño e implementación](#2-diseño-e-implementación)
5. [Ejecución](#3-ejecución)
6. [Análisis del rendimiento](#4-análisis-del-rendimiento)
7. [Trabajo en equipo](#5-trabajo-en-equipo)
8. [Conclusiones](#6-conclusiones)
9. [Bibliografía](#7-bibliografía)
---
### Datos generales

* **Tema**: Redes Neuronales en Análisis y Proyección de enfermedades cardíacas en hospitales
* **Grupo**: `The frontiend`
* **Integrantes**:
  * José Ignacio Benalcázar Ferro – 202410208
---

### Requisitos e instalación

1. **Compilador**: C++ 17 o superior, GCC 11 o superior
2. **Dependencias**:

   * CMake 3.18+
   * Eigen 3.4
   * wget (Para la descarga del dataset)
3. **Instalación**:
    Paso 1: Clonar repositorio que contiene el dataset
   ```bash
   git clone https://github.com/sharmaroshan/Heart-UCI-Dataset.git
   cd Heart-UCI-Dataset
   ````
   Paso 2: Verificar el dataset y ajustar el formato
   ```bash
   wc -l heart.csv  # Debe mostrar ~304 líneas (303 datos + 1 header)
   head -1 heart.csv  # Debe mostrar los nombres de columnas
   head -5 heart.csv
   ````
    Paso 3: Compilar el proyecto 
   ```bash
   git clone https://github.com/proyecto-final-2025-2-the_frontiend.git
   cd proyecto-final-2025-2-the_frontiend
   mkdir build && cd build
   cmake .. -DCMAKE_BUILD_TYPE=Release
   make j4
   cd build
   ./heart_disease_train ../heart.csv
   ```
---

### 1. Investigación teórica

* **Objetivo**: Explorar fundamentos y arquitecturas de redes neuronales y sus potenciales aplicaciones presentes y futuras.

  1. Historia y evolución de las NNs.
   * Las redes neuronales artificiales tienen sus orígenes en 1943 con el trabajo de Warren McCulloch y Walter Pitts, quienes crearon el primer modelo computacional inspirado en neuronas biológicas. En 1958,          Frank  Rosenblatt desarrolló el perceptrón, la primera red neuronal capaz de clasificación binaria, marcando el inicio de modelos que podían aprender de datos. Sin embargo, en 1969, Marvin Minsky y Seymour       Papert demostraron limitaciones fundamentales del perceptrón, como su incapacidad para resolver problemas no lineales como XOR, lo que condujo al primer "invierno de la IA" y estancamiento en la                  investigación por más de una década.
   * El algoritmo de backpropagation fue derivado inicialmente en los años 1960 de forma ineficiente, pero Seppo Linnainmaa publicó la versión moderna en 1970. El verdadero renacimiento llegó en 1986 cuando           David Rumelhart, Geoffrey Hinton y Ronald Williams formalizaron y popularizaron backpropagation, permitiendo entrenar redes multicapa y resolver problemas no lineales. A partir de 2006, Geoffrey Hinton           introdujo métodos de pre-entrenamiento no supervisado que inauguraron la era del deep learning, seguido por avances en GPUs que aceleraron el entrenamiento de redes profundas. En 2012, AlexNet ganó la            competencia ImageNet con una red convolucional profunda, catalizando el actual "spring de IA" y expansión masiva del aprendizaje profundo en múltiples industrias.

2. Principales arquitecturas: MLP, CNN, RNN.
   **Multilayer Perceptron (MLP):** Los MLPs son redes completamente conectadas donde cada neurona de una capa se conecta con todas las neuronas de la siguiente. Son ideales para datos tabulares y problemas de        regresión o clasificación simple, pero requieren vectores de entrada aplanados y no capturan relaciones espaciales o temporales en los datos. Su naturaleza "structure-agnostic" las hace versátiles pero           menos eficientes que arquitecturas especializadas para dominios específicos.

   **Convolutional Neural Networks (CNN):** Las primeras redes convolucionales fueron desarrolladas por Fukushima en 1979, utilizando capas convolucionales y de pooling similares a las arquitecturas modernas.         Las CNNs  utilizan filtros convolucionales que detectan patrones como bordes, formas y texturas de manera jerárquica, siendo especialmente efectivas para procesamiento de imágenes y datos espaciales.             Estudios comparativos demuestran que CNNs superan a MLPs en reconocimiento de caracteres complejos por factores de 2x en velocidad de clasificación. Sus limitaciones incluyen alto costo computacional, falta      de interpretabilidad, y dificultad para manejar datos secuenciales sin combinarlos con RNNs.
 
   **Recurrent Neural Networks (RNN):** Las RNNs procesan datos secuenciales manteniendo un estado oculto que captura dependencias temporales en el historial de datos. Son apropiadas para modelado de lenguaje,        traducción automática, reconocimiento de voz y series temporales, pero sufren del problema de gradiente desvaneciente que limita el aprendizaje de dependencias a largo plazo. Long Short-Term Memory (LSTM),       una variante de RNN, usa mecanismos de compuertas para resolver este problema y mantener dependencias temporales extensas. A diferencia de CNNs que comparten parámetros espacialmente, RNNs comparten              parámetros temporalmente, pero su procesamiento secuencial las hace lentas de entrenar.

3. Algoritmos de entrenamiento: backpropagation, optimizadores.
     
   **Backpropagation:** Backpropagation es una aplicación eficiente de la regla de la cadena derivada por Leibniz en 1673, que calcula gradientes de funciones anidadas propagando errores hacia atrás capa por          capa. El método evita cálculos redundantes al reutilizar gradientes parciales de capas posteriores para calcular gradientes de capas anteriores mediante programación dinámica. Aunque backpropagation fue          derivado múltiples veces desde los años 1960, su importancia no fue completamente apreciada hasta el paper seminal de Rumelhart, Hinton y Williams en 1986. Sus aplicaciones exitosas incluyeron NETtalk            (1987) para conversión de texto a voz, ALVINN (1989) para conducción autónoma, y LeNet (1989) para reconocimiento de dígitos manuscritos.

   **Optimizadores SGD y Momentum:** Stochastic Gradient Descent (SGD) actualiza parámetros con gradientes calculados en mini-batches, permitiendo convergencia más rápida que gradient descent batch. Momentum          acelera SGD al acumular un promedio móvil exponencial de gradientes pasados, reduciendo oscilaciones y permitiendo escapar de mínimos locales y mesetas. Sin embargo, SGD con momentum tiende a encontrar           mínimos más planos que generalizan mejor que los mínimos agudos a los que convergen métodos adaptativos.
 
   **Optimizadores Adaptativos**: AdaGrad adapta tasas de aprendizaje individualmente para cada parámetro basándose en gradientes históricos acumulados, siendo efectivo para features
 
4. Aplicaciones de las NNs y proyección a futuro.
   **Salud**: Se utilizan para el diagnóstico médico mediante la clasificación de imágenes médicas (como la detección de lesiones cutáneas o retinopatía diabética) y para el autodiagnóstico potencial en el          futuro.

  **Finanzas**: Ayudan en las predicciones financieras y la detección de fraudes mediante el procesamiento de datos históricos de instrumentos financieros.
  
  **Procesamiento del Lenguaje Natural (PLN)**: Hacen posible la traducción automática, el análisis de sentimientos y el reconocimiento de voz en asistentes virtuales y smartphones.
  Visión Artificial: Son la base de la identificación de objetos en imágenes, lo que permite aplicaciones en vehículos autónomos, sistemas de guía de automóviles y organización automática de fotos en redes         sociales.

  **Marketing y Comercio**: Permiten el marketing dirigido mediante el análisis de datos de comportamiento y el filtrado de redes sociales.
   
  **Automatización y Robótica**: Se usan en la detección de fallos en aeronaves y en sistemas de casas inteligentes. 

4.1 Proyección a Futuro:

   La investigación actual busca superar las limitaciones computacionales y la dependencia de datos etiquetados para expandir aún más las capacidades de las redes neuronales. 
  **Eficiencia Mejorada**: El enfoque está en hacer que los modelos sean más eficientes, requiriendo menos recursos computacionales y energéticos para el entrenamiento y la operación.

  **Aprendizaje Autosupervisado**: Se espera un mayor desarrollo del aprendizaje autosupervisado, donde los sistemas puedan aprender de grandes cantidades de datos sin etiquetas explícitas.

  **Integración Multimodal**: Las futuras redes neuronales probablemente integrarán y procesarán diversos tipos de datos simultáneamente (imágenes, texto, audio) para una comprensión más completa del entorno.

  **Operadores Neuronales**: El uso de operadores neuronales para aprender mapas sustitutos de ecuaciones diferenciales parciales permitirá un modelado más rápido y eficiente de fenómenos naturales complejos,        superando a los métodos computacionales estándar.

  **Automatización Avanzada**: Se anticipa una mayor automatización de tareas complejas en los negocios y la industria, así como la creación de nuevos roles laborales centrados en la ética y la integración de la     IA.
  **Avances Creativos**: En el futuro, las redes neuronales podrían permitir la composición musical avanzada y la conversión automática de documentos manuscritos a formatos digitales perfectamente formateados.

### 2. Diseño e implementación

#### 2.1 Arquitectura de la solución

* **Patrones de diseño**:
  - Strategy en `nn_loss.h, nn_optimization.h` y `nn_activation.h` para la agregación de funciones sin modificar código previo.
  - Composite en `neural_network.h` para la separación adecuada de objetos y composiciones de objetos.
  - Dependency Injection en `nn_layer.h` para la inicialización de pesos y mocking para testeos.
* **Estructura de carpetas**:
  
  ```
  proyecto-final-2025-2-the_frontiend/
  ├── include/
  │   ├── tensor.h                    
  │   ├── neural_network.h            
  │   ├── nn_interfaces.h             
  │   ├── nn_dense.h
  │   ├── nn_activation.h             
  │   ├── nn_loss.h                   
  │   ├── nn_optimizer.h              
  │   ├── data_loader.h               
  │   └── evaluation.h                
  ├── heart_disease_train.cpp        
  ├── tests/
  │   └── test_suite.cpp              
  ├── CMakeLists.txt                  
  ├── download_dataset.sh             
  ├── heart.csv                       
  └── README.md                       
  ```

#### 2.2 Manual de uso y casos de prueba

* **Cómo ejecutar**: `# Desde el directorio build/`
* ` ./heart_disease_train ../heart.csv`
* **Casos de prueba**:

  * Test Suite de operaciones, activaciones, optimización/pérdida y batches.
  * Test del problema XOR.
  * Test de entrenamiento, validación y rendimiento.
---

#### 2.3 Arquitectura de Red
**Arquitectura Base**:
`Input(13) → Dense(32) → ReLU → Dense(16) → ReLU → Dense(1) → Sigmoid`

Justificación:

- **Capa 1 (32 neuronas)**: Captura relaciones complejas entre las 13 features.
- **Capa 2 (16 neuronas)**: Reduce dimensionalidad y extrae patrones de alto nivel.
- **Capa 3 (1 neurona + Sigmoid)**: Clasificación binaria con probabilidad [0,1].

#### 2.4 Explicación del Dataset a utilizar


### 3. Ejecución

> **Demo de ejemplo**: Video/demo alojado en `Demo_NN_Proyecto.mp4`.
> Pasos:
>
> 1. Preparar datos de entrenamiento (formato CSV).
> 2. Ejecutar comando de entrenamiento.
> 3. Evaluar resultados con script de validación.

---

### 4. Análisis del rendimiento

* **Métricas de ejemplo**:

  * Iteraciones: 5000 épocas.
  * Tiempo total de entrenamiento: 1m20s.
  * Precisión final: 82.57%.
* **Ventajas**:

  * Control total y comprensión profunda
  * Sin dependencias externas
  * Eficiencia de memoria
  * Template programming avanzado
  * Patrones profesionales
  * Rendimiento competitivo en datasets pequeños
 
  * **Desventajas**:
  * Escalabilidad limitada (OpenMP, GPU)
  * Funcionalidad limitada (Extender gradualmente)
  * Debugging complejo (Tests exhaustivos)
  * Falta de herramientas (Scripts externos)
  * Operaciones subóptimas (Optimizar con compiler flags)
* **Mejoras futuras**:

  * Implementar logging detallado de gradientes
  * Crear visualizaciones simples con Python para una mayor apreciación de los datos y del rendimiento de la red en distintos escenarios de datos
  * Tests unitarios exhaustivos para poner a prueba la adaptabilidad del modelo y sus componentes para detectar todas las excepciones posibles y corregirlas en consecuencia

---

### 5. Trabajo en equipo

| Tarea                     | Miembro  | Rol                                            |
| ------------------------- | -------- | ---------------------------------------------- |
| Investigación teórica     | José Ignacio Benalcázar Ferro | Documentar bases teóricas |
| Diseño de la arquitectura | José Ignacio Benalcázar Ferro | Estructuración de clases  |
| Implementación del modelo | José Ignacio Benalcázar Ferro | Código C++ de la NN       |
| Pruebas y benchmarking    | José Ignacio Benalcázar Ferro | Generación de métricas    |
| Documentación y demo      | José Ignacio Benalcázar Ferro | Tutorial y video demo     |


---

### 6. Conclusiones

Se logró realizar la implementación completa desde cero, incluyendo los siguientes componentes:
- Red neuronal funcional sin dependencias externas
- Backpropagation manual con cálculo correcto de gradientes
- Optimizadores SGD y Adam implementados correctamente

La Red Neuronal es efectiva para la predicción de datos en áreas como la medicina, con un dataset real de enfermedad cardíaca (303 pacientes), y con estadísticas relativamente favorables
(Accuracy competitivo: 80-90% (literatura: 85-93%))

La Optimización SGD Demostró ser más efectiva que Adam, con mayor estabilidad y menos propenso a overfitting, pese a que sea más lento. Se recomienda empezar con SGD para baseline, posteriormente experimentar con Adam.


---

### 7. Bibliografía

Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. Nature, 323(6088), 533-536. https://doi.org/10.1038/323533a0
Cong, S., & Zhou, Y. (2023). A review of convolutional neural network architectures and their optimizations. Artificial Intelligence Review, 56(3), 1905-1969. https://doi.org/10.1007/s10462-022-10213-5
Bredikhin, A. I. (2019). Training algorithms for convolutional neural networks. Yugra State University Bulletin, 15(1), 41-54. https://doi.org/10.17816/byusu20190141-54
Galván, E., & Mooney, P. (2021). Neuroevolution in deep neural networks: Current trends and future challenges. IEEE Transactions on Artificial Intelligence, 2(6), 476-493. https://doi.org/10.1109/TAI.2021.3067574
