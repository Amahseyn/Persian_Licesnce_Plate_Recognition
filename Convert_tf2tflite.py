import tensorflow as tf
model = tf.keras.models.load_model('Plate_1Dec.h5',compile=False)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model_quant = converter.convert()
with open('Plate_1Dec.tflite', 'wb') as f:
  f.write(tflite_model_quant)