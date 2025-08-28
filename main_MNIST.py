import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os 
# --- 1. Cargar datos ---
def load_and_preprocess_data(file_path="digit-recognizer/train.csv"):
    """
    Carga el dataset de Kaggle MNIST, lo preprocesa y lo divide en conjuntos
    de entrenamiento y validación según la lógica del usuario.
    """
    if not os.path.exists(file_path):
        print(f"Error: El archivo '{file_path}' no se encontró.")
        print("Asegúrate de que el archivo 'train.csv' esté en la carpeta 'digit-recognizer'.")
        return None, None, None, None

    data = pd.read_csv(file_path)
    data = np.array(data)
    m, n = data.shape # m = número de muestras, n = número de características + 1 (para la etiqueta)
    np.random.shuffle(data) # Mezclar los datos para asegurar una buena distribución

    # División de los datos según la lógica del usuario
    # data_dev se usará como conjunto de validación
    data_dev = data[0:5000].T # Transponer para tener (características, muestras)
    Y_val = data_dev[0]      # La primera fila es la etiqueta
    X_val = data_dev[1:n] / 255.0 # El resto son píxeles, normalizados a [0, 1]

    # data_train se usará como conjunto de entrenamiento
    data_train = data[5000:m].T # Transponer para tener (características, muestras)
    Y_train = data_train[0]    # La primera fila es la etiqueta
    X_train = data_train[1:n] / 255.0 # El resto son píxeles, normalizados a [0, 1]

    print(f"Dimensiones de X_train: {X_train.shape}")
    print(f"Dimensiones de Y_train: {Y_train.shape}")
    print(f"Dimensiones de X_val: {X_val.shape}")
    print(f"Dimensiones de Y_val: {Y_val.shape}")

    return X_train, Y_train, X_val, Y_val

# --- 2. Inicialización de Parámetros ---
def init_params(neurons_hidden_number):
    """
    Inicializa los pesos (W) y sesgos (b) para una red neuronal de dos capas.
    Utiliza la inicialización He para las capas con activación ReLU.
    W1: (número de neuronas en la capa oculta, número de características de entrada)
    b1: (número de neuronas en la capa oculta, 1)
    W2: (número de neuronas de salida, número de neuronas en la capa oculta)
    b2: (número de neuronas de salida, 1)
    """
    # W1: (neurons_hidden_number, 784)
    W1 = np.random.randn(neurons_hidden_number, 784) * np.sqrt(2. / 784)
    b1 = np.zeros((neurons_hidden_number, 1))

    # W2: (10, neurons_hidden_number)
    W2 = np.random.randn(10, neurons_hidden_number) * np.sqrt(2. / neurons_hidden_number)
    b2 = np.zeros((10, 1))

    return W1, b1, W2, b2

# --- 3. Funciones de Activación ---
def ReLU(Z):
    """Función de activación Rectified Linear Unit (ReLU)."""
    return np.maximum(0, Z)

def softmax(Z):
    """
    Función de activación Softmax.
    Aplica la función exponencial y normaliza para obtener probabilidades.
    Se resta el máximo de Z para estabilidad numérica y evitar desbordamientos.
    """
    exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

# --- 4. Propagación Hacia Adelante ---
def forward_prop(W1, b1, W2, b2, X):
    """
    Realiza la propagación hacia adelante a través de la red.
    Z1 = W1 * X + b1 (entrada ponderada a la capa oculta)
    A1 = ReLU(Z1)   (activación de la capa oculta)
    Z2 = W2 * A1 + b2 (entrada ponderada a la capa de salida)
    A2 = softmax(Z2) (activación de la capa de salida - probabilidades)
    """
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

# --- 5. Codificación One-Hot ---
def one_hot(Y):
    """
    Convierte un vector de etiquetas numéricas en una matriz one-hot.
    Ejemplo: 0 -> [1,0,0,...], 1 -> [0,1,0,...]
    Necesario para el cálculo de la pérdida de entropía cruzada.
    """
    # Crea una matriz de ceros con dimensiones (número de clases, número de muestras)
    one_hot_Y = np.zeros((Y.max() + 1, Y.size))
    # Establece un 1 en la posición correspondiente a la etiqueta real
    one_hot_Y[Y, np.arange(Y.size)] = 1
    return one_hot_Y

# --- 6. Derivadas de Funciones de Activación ---
def deriv_ReLU(Z):
    """Derivada de la función ReLU."""
    return Z > 0 # Retorna una matriz booleana que se convierte a 0 o 1

# --- 7. Función de Pérdida (Cross-Entropy) ---
def cross_entropy_loss(Y_true_one_hot, A2_pred):
    """
    Calcula la pérdida de entropía cruzada categórica.
    Y_true_one_hot: Etiquetas verdaderas en formato one-hot (num_classes, num_samples)
    A2_pred: Probabilidades predichas por la red (num_classes, num_samples)
    """
    m = Y_true_one_hot.shape[1] # Número de muestras

    # Recortar las predicciones para evitar log(0) o log(1) que son indefinidos.
    # Esto mejora la estabilidad numérica.
    epsilon = 1e-12
    A2_pred = np.clip(A2_pred, epsilon, 1. - epsilon)

    # Cálculo de la pérdida de entropía cruzada.
    # np.sum(Y_true_one_hot * np.log(A2_pred)) suma solo los logaritmos
    # de las probabilidades de las clases correctas.
    loss = -np.sum(Y_true_one_hot * np.log(A2_pred)) / m
    return loss

# --- 8. Propagación Hacia Atrás ---
def back_prop(Z1, A1, Z2, A2, W2, X, Y):
    """
    Realiza la propagación hacia atrás para calcular los gradientes de los pesos y sesgos.
    Se basa en la minimización de la pérdida de entropía cruzada.
    """
    m = Y.size # Número de muestras
    one_hot_Y = one_hot(Y) # Convertir Y a formato one-hot

    # Gradientes de la capa de salida
    dZ2 = A2 - one_hot_Y # Error de la capa de salida (predicciones - reales)
    dW2 = (1 / m) * dZ2.dot(A1.T) # Gradiente de W2
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True) # Gradiente de b2

    # Gradientes de la capa oculta
    # dZ1 = dZ2 propagado hacia atrás a través de W2, multiplicado por la derivada de ReLU
    dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)
    dW1 = (1 / m) * dZ1.dot(X.T) # Gradiente de W1
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True) # Gradiente de b1

    return dW1, db1, dW2, db2

# --- 9. Actualización de Parámetros ---
def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    """
    Actualiza los pesos y sesgos utilizando el descenso de gradiente.
    alpha es la tasa de aprendizaje.
    """
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2

# --- 10. Predicciones y Precisión ---
def get_predictions(A2):
    """
    Obtiene las predicciones de clase a partir de las probabilidades de salida (A2).
    Selecciona el índice con la probabilidad más alta.
    """
    return np.argmax(A2, axis=0)

def get_accuracy(predictions, Y_true):
    """
    Calcula la precisión del modelo comparando las predicciones con las etiquetas verdaderas.
    """
    # Asegurarse de que Y_true sea un array 1D
    Y_true = Y_true.flatten()
    accuracy = np.sum(predictions == Y_true) / Y_true.size
    return accuracy

# --- 11. Entropía de Shannon ---
def shannon_entropy(W):
    """
    Calcula la entropía de Shannon de una matriz de pesos W.
    """
    # Normalizar los pesos para convertirlos en probabilidades
    W = W.flatten()
    if len(W) < 2:
        return 0  # No se puede calcular la entropía con menos de 2 valores
    hist, bin_edges = np.histogram(W, bins=100, density=False)
    if len(W) == 0:
        return 0  # Evitar división por cero
    probabilities = hist / np.sum(hist)

    # Calcular entropía solo para valores no nulos (log(0) es indefinido)
    probabilities = probabilities[probabilities > 0]

    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

# --- 12. Descenso de Gradiente (Función de Entrenamiento Principal) ---
def gradient_descent(X_train, Y_train, X_val, Y_val, iterations, alpha, neurons_hidden_number):
    """
    Implementa el algoritmo de descenso de gradiente para entrenar la red neuronal.
    Registra la precisión y la pérdida durante el entrenamiento y guarda los resultados.
    También visualiza el progreso y la entropía de los pesos.
    """
    W1, b1, W2, b2 = init_params(neurons_hidden_number)
    Y_train_one_hot = one_hot(Y_train) # One-hot encode Y_train una vez

    # Listas para almacenar métricas para la visualización
    train_accuracies = []
    val_accuracies = []
    train_losses = []
    val_losses = []

    output_filename1 = f"results_MNIST/MNIST_results_iter{iterations}_alpha{alpha}_hidden{neurons_hidden_number}.dat"
    output_filename2 = f"MNIST_results_iter{iterations}_alpha{alpha}_hidden{neurons_hidden_number}"
    with open(output_filename1, 'w') as f:
        f.write("Iteration,Train_Accuracy,Val_Accuracy,Train_Loss,Val_Loss,W1_Entropy,W2_Entropy,W3_Entropy\n")

        for i in range(1, iterations + 1):
            # Propagación hacia adelante (entrenamiento)
            Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X_train)

            # Cálculo de la pérdida de entrenamiento
            current_train_loss = cross_entropy_loss(Y_train_one_hot, A2)
            train_losses.append(current_train_loss)

            # Cálculo de la precisión de entrenamiento
            predictions_train = get_predictions(A2)
            current_train_accuracy = get_accuracy(predictions_train, Y_train)
            train_accuracies.append(current_train_accuracy)

            # Propagación hacia atrás
            dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W2, X_train, Y_train)

            # Actualización de parámetros
            W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)

            # Evaluación en el conjunto de validación (sin actualización de pesos)
            _, _, _, A2_val = forward_prop(W1, b1, W2, b2, X_val)
            Y_val_one_hot = one_hot(Y_val) # One-hot encode Y_val
            current_val_loss = cross_entropy_loss(Y_val_one_hot, A2_val)
            val_losses.append(current_val_loss)

            predictions_val = get_predictions(A2_val)
            current_val_accuracy = get_accuracy(predictions_val, Y_val)
            val_accuracies.append(current_val_accuracy)

            W3=W2@W1
            # Cálculo de la entropía de los pesos
            # Normalizamos por el log2 del número total de elementos en la matriz
            # para que la entropía esté en el rango [0, 1].
            W1_entropy = shannon_entropy(W1) / np.log2(W1.size)
            W2_entropy = shannon_entropy(W2) / np.log2(W2.size)
            W3_entropy = shannon_entropy(W3) / np.log2(W3.size)


            # Imprimir y guardar resultados periódicamente
            if i % 200 == 0 or i == iterations:
                print(f"Iteración: {i/iterations*100:.2f} %")
                np.savez(f"results_MNIST/pesos/{i}_{output_filename2}", W1=W1, W2=W2, b1=b1, b2=b2)
                f.write(f"{i},{current_train_accuracy},{current_val_accuracy},{current_train_loss},{current_val_loss},{W1_entropy},{W2_entropy},{W3_entropy}\n")
    return W1, b1, W2, b2

# --- 13. Función de Predicción para Nuevas Muestras ---
def predict(X, W1, b1, W2, b2):
    """
    Realiza una predicción sobre nuevas muestras utilizando los parámetros entrenados.
    X: Datos de entrada (784, num_samples)
    """
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

# --- Ejecución del Modelo ---
# Cargar y preprocesar los datos
X_train, Y_train, X_val, Y_val = load_and_preprocess_data()

if X_train is not None:
    # Definir hiperparámetros
    iterations = 50000
    alpha = 0.1
    neurons_hidden_number = 256

    # Entrenar el modelo
    print("\nIniciando el entrenamiento del modelo...")
    W1, b1, W2, b2 = gradient_descent(X_train, Y_train, X_val, Y_val,
                                          iterations, alpha, neurons_hidden_number)

    # Evaluar el modelo final en el conjunto de validación
    final_predictions_val = predict(X_val, W1, b1, W2, b2)
    final_accuracy_val = get_accuracy(final_predictions_val, Y_val)
    print(f"\nPrecisión final en el conjunto de validación: {final_accuracy_val:.4f}")

    # Opcional: Visualizar una predicción de ejemplo
    # Puedes cambiar el índice para ver diferentes imágenes
    index = 0 # Cambia este índice para ver diferentes imágenes de validación
    image_data = X_val[:, index].reshape(28, 28)
    true_label = Y_val[index]
    predicted_label = predict(X_val[:, index].reshape(-1, 1), W1, b1, W2, b2)[0]

    plt.imshow(image_data, cmap='gray')
    plt.title(f"Verdadero: {true_label}, Predicho: {predicted_label}")
    plt.axis('off')
    plt.show()
