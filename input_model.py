#D'après https://stackoverflow.com/questions/54590363/create-tensorflow-dataset-from-image-local-directory
import tensorflow as tf
import os
from google.colab import files
from google.colab import drive
drive.mount('/content/drive')
os.system("cd '/content/drive/My Drive/TIPE'")
dossier_TIPE = "/content/drive/My Drive/TIPE/"
class ArtificialDataset(tf.data.Dataset):
    def open_imgs(x):
        img = tf.io.read_file(x)
        img = tf.image.decode_jpeg(img)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, [256, 256])
        img = tf.clip_by_value(img,0,1)
        return img
        
    def _generator():
        for img_batch in ArtificialDataset.donnees:
            cleans = []
            for i,img in enumerate(img_batch):
                clean = ArtificialDataset.open_imgs(img)
                cleans.append(clean)
            cleans_t = tf.stack(cleans)
            yield (cleans_t)
    
    def __new__(cls,nom):
        ArtificialDataset.donnees = tf.data.Dataset.list_files(dossier_TIPE+"Galaxies_resized/"+nom+"/*.jpg").shuffle(len(os.listdir(dossier_TIPE+"Galaxies_resized/"+nom))).batch(10)
        #voir doc from_generator https://github.com/tensorflow/tensorflow/blob/f2b2563c6ce2001a117cd7adb36f303903e907ec/tensorflow/python/data/ops/dataset_ops.py#L669
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_types=(tf.dtypes.float32),
            output_shapes=(tf.TensorShape([None,256,256,3])), #https://www.tensorflow.org/api_docs/python/tf/TensorShape?version=nightly
            args=None
        )


#Bruit gaussien origine : https://stackoverflow.com/questions/59286171/gaussian-blur-image-in-dataset-pipeline-in-tensorflow
def _gaussian_kernel(kernel_size, sigma, n_channels, dtype):
    """Génère le noyau de la couche de convolution du flou gaussien"""
    x = tf.range(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=dtype)
    g = tf.math.exp(-(tf.pow(x, 2) / (2 * tf.pow(tf.cast(sigma, dtype), 2))))
    g_norm2d = tf.pow(tf.reduce_sum(g), 2)
    g_kernel = tf.tensordot(g, g, axes=0) / g_norm2d
    g_kernel = tf.expand_dims(g_kernel, axis=-1)
    return tf.expand_dims(tf.tile(g_kernel, (1, 1, n_channels)), axis=-1)

def apply_blur(img):
    """Applique le flou gaussien"""
    img = tf.reshape(img,[1,256,256,3])
    blur = _gaussian_kernel(3, 2, 3, img.dtype)
    img = tf.nn.depthwise_conv2d(img, blur, [1,1,1,1], 'SAME')
    return tf.reshape(img,[256,256,3])
def bruitage(img):
    #Bruitage
    img_noise = tf.image.adjust_saturation(img,.8)
    img_noise = apply_blur(img_noise)
    img_noise = tf.image.adjust_contrast(img_noise,tf.reduce_sum(tf.random.uniform(shape=(1,),dtype=tf.float32,minval=0.1,maxval=0.2)))
    ajout = 0.1
    img_noise = img_noise+tf.concat([tf.ones(shape=[256,256,1])*ajout,tf.zeros(shape=[256,256,2])],axis=-1)
    img_noise = tf.image.adjust_brightness(img_noise,-ajout)
    img_noise = tf.image.adjust_hue(img_noise,tf.reduce_sum(tf.random.uniform(shape=(1,),dtype=tf.float32,minval=0.06,maxval=0.09)))
    img_noise = tf.clip_by_value(img_noise,0,1)
    return img_noise
def traitement(img_batch):
    """Modification par image"""
    noised = tf.map_fn(bruitage,img_batch)
    img_noise_batch = tf.concat([tf.slice(noised,[0,0,0,0],[len(img_batch)//2,256,256,3]),tf.slice(img_batch,[len(img_batch)//2,0,0,0],[len(img_batch)-len(img_batch)//2,256,256,3])],axis=0)
    return img_noise_batch,img_batch