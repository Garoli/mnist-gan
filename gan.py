import tensorflow as tf
import time
import matplotlib.pyplot as plt
import numpy as np
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization

def make_generator():
    model = Sequential()
    model.add(Dense(256 * 7 * 7, use_bias=False,input_shape=(100,)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Reshape((7, 7, 256)))

    model.add(Conv2DTranspose(128, (5,5),use_bias=False, strides=(2,2), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(64, (5,5), use_bias=False,strides=(2,2), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(1, (5,5), use_bias=False,activation='tanh', padding='same'))
    return model

def generator_loss( fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def make_discriminator():
    model = Sequential()  # Initiate the network
    #Deux couches entonoirs
    model.add(Conv2D(64, (5, 5), strides=(2, 2), padding="same", input_shape=(28,28,1)))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (5, 5), strides=(2, 2), padding="same", input_shape=(28,28,1)))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(1,activation='sigmoid')) # we want binary output
    return model

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def load_models():
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    generator_optimizer = checkpoint.generator_opti
    discriminator_optimizer = checkpoint.discriminator_opti
    generator = checkpoint.generator
    discriminator = checkpoint.discriminator

def save_models():
    checkpoint.save(checkpoint_prefix)

@tf.function
def train_one_step(images):
    noise = tf.random.normal([BATCH_SIZE, 100])
    with tf.GradientTape() as gradientGenerator, tf.GradientTape() as gradientDiscriminator:
        fake_images = generator(noise,training=True)
        fake_decision = discriminator(fake_images,training=True)
        real_decision = discriminator(images,training=True)
        loss_discriminator = discriminator_loss(real_decision,fake_decision)
        loss_generator = generator_loss(fake_decision)
    gradG = gradientGenerator.gradient(loss_generator, generator.trainable_variables)
    gradD = gradientDiscriminator.gradient(loss_discriminator, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(gradD,discriminator.trainable_variables))
    generator_optimizer.apply_gradients(zip(gradG,generator.trainable_variables))

def train():
    size = len(list(dataset))
    for epoch in range(EPOCH):
        start = time.time()
        i = 0
        print('EPOCH N°{}'.format(epoch+1));
        for image_batch in dataset:
            i += 1
            start_batch = time.time()
            train_one_step(image_batch)
            #print('Time for epoch {} is {} sec {}batch/{}'.format(epoch + 1, time.time() - start_batch, i, size))

        if (epoch + 1) % SAVE_EPOCH == 0:
            for x in range(0, 4):
                img = generator(tf.random.normal([1, 100]))
                plt.imshow(np.array(img).reshape(28, 28, 1) * 127.5 + 127.5, cmap='gray')
                plt.savefig("img/" + ( str(0) if tf.train.latest_checkpoint(checkpoint_dir) is None else tf.train.latest_checkpoint(checkpoint_dir)[28:])  + "-"+str(x)+ ".png")
            save_models()


#We declare everything we need for the different network
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
#We built the two networks
discriminator = make_discriminator()
generator = make_generator()

#We can configure different option
EPOCH = 50
SAVE_EPOCH = 1
BATCH_SIZE = 256
NUMBER_DATA = 60000

#We get the data and modify them
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5
dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(NUMBER_DATA).batch(BATCH_SIZE)

#To save the network
checkpoint = tf.train.Checkpoint(generator_opti=generator_optimizer,
                                 discriminator_opti=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

print("---------BEGINNING OF THE TRAINING LOOP---------------------")
train()
