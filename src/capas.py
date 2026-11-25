"""
capas.py - Capas propias de la CNN v3 (se importan antes de cargar el modelo).

Keras necesita conocer las capas que no son suyas para poder leer el archivo .keras;
por eso PoolingLSE vive aquí y la importan el entrenamiento, la evaluación y las demos.
"""
import keras


@keras.saving.register_keras_serializable(package="cnn_v3")
class PoolingLSE(keras.layers.Layer):
    """
    "Máximo suave" de un mapa de puntajes (N, alto, ancho, clases) -> (N, clases):
        (1/r) * log( promedio( exp(r * s) ) )
    Con r grande se parece al máximo (basta UNA zona con vaca para decir "hay vaca");
    con r -> 0 se parece al promedio (GlobalAveragePooling). Usamos r = 5.
    Funciona con cualquier tamaño de mapa, por eso la red acepta imágenes de cualquier tamaño.
    """

    def __init__(self, r=5.0, **kw):
        super().__init__(**kw)
        self.r = r

    def call(self, s):
        s = keras.ops.cast(s, "float32") * self.r
        hw = keras.ops.cast(keras.ops.shape(s)[1] * keras.ops.shape(s)[2], "float32")
        return (keras.ops.logsumexp(s, axis=(1, 2)) - keras.ops.log(hw)) / self.r

    def get_config(self):
        return {**super().get_config(), "r": self.r}
