
import tensorflow as tf

print(f"TensorFlow version: {tf.__version__}")

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"Found {len(gpus)} GPUs.")
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        
        with tf.device('/GPU:0'):
            print("Performing a sample computation on GPU:0")
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
            c = tf.matmul(a, b)
            print("Result of matrix multiplication:")
            print(c)

    except RuntimeError as e:
        print(e)
else:
    print("No GPU found. Please ensure TensorFlow is installed with GPU support and that your environment is configured correctly.")
