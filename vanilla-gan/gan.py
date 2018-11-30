from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt

import sys

import numpy as np

class GAN():
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        ## notice binary crossentropy has 2 terms..
        ## (p(x))^{y} * (1-p(x))^{1-y} is max liklihood formulation
        ## i.e. we aim to maximise this,,,,
        ##  if we take log of expression then we can say we are summing 
        ## the log of each terms... also we add a negative sign
        ## so we get negative log likehood formulation which we will now MINIMISE
        ## - (log(p(x))^{y} + log(1-p(x))^{1-y})
        ##
        ##
        ## this means if y is 0 then first term disappears i.e. =1
        ## and our loss is just the second term
        ## which will always be >= 0 because.. if
        ## p(x) >= 0 then 1-p(x) <= 1 thus..
        ## log(...) term will always give NEGATIVE value for
        ## '...' being <= 1 and thus
        ## negative*negative = ppos
        ## so our ideal 2nd term is 0 and in practice it'll be > 0
        ## 
        ## same way when y=1, 2nd term =1 and first term
        ## log(...) is ONLY =0 when p(x) = 1 otherwise its.
        ## < 0 when p(x) < 1 and thus 
        ## - (log(...)) is ideal at zero and otherwise has a
        ## positive value


        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        validity = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)


    def build_generator(self):

        model = Sequential()

        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Flatten(input_shape=self.img_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=128, sample_interval=50):

        # Load the dataset
        (X_train, _), (_, _) = mnist.load_data()

        # Rescale -1 to 1
        X_train = X_train / 127.5 - 1.
        X_train = np.expand_dims(X_train, axis=3)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))  ##  all list of 1,1,1..,1
        fake = np.zeros((batch_size, 1))  ##  all list of 0,0,0..,0

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Generate a batch of new images
            ## G(z) we want this to be ideally be an unbiased estimator of x? 
            ## i.e. Exp[ G(z) ] == Exp[ x ]
            ##
            ## Specifically:
            ## Given G(z) ~ g(G(z)), and x ~ f(x)
            ## where G(z) is a random var as generated image sample
            ## x is original image
            ## we want g === f i.e. 
            ## the prob_density_of_G(z) == prob_density_of_x
            gen_imgs = self.generator.predict(noise)

            ## from paper the minmax loss fn is formulated as:
            ## min_G__max_D V(D, G)
            ## i.e. find a G that MINIMISES V(D, G)
            ## and simultaneously find a D that MAXIMISES V(D, G)
            ## where V(D, G) = Exp_{x}[log(D(x))] + Exp_{z}[log(1-D(G(z)))]
            ##
            ## notice how this equation is very muhc similar to the log-likelihood we say earlier
            ## specifically the Exp_{..} term in practical life simply means summing and dividing by
            ## number of samples, so if we ignore this for now assume stochastic update or per sample update
            ## what we want is to find a 'D(...)' or the parameters/weights that parametrizes/forms/defines D
            ## such that the log likelihood is maximised. And thats the likelihood that if input to D
            ## is 'x' then D == 1 and if its G(z) then D == 0 i.e. D is meant to tell us
            ## how 'likely' is the input REAL
            ## in logistic regression the network tells us how likely is the output/event true i.e. 
            ## how likely is y=1 given data x same idea!.
            ## whereas when we look from G's perspective we dont case about the first term because any
            ## gradients will be zero so we only need to focus in this term. 
            ## the objective wants to minimise G so minimise Exp_{z}[log(1-D(G(z)))] which 
            ## only happens when D is 'fooled' to think that G(z) is real i.e. as
            ## D(G(z)) -> 1, 1 - D(G(z)) -> 0 then log(...) -> -inf!
            ## so at start log(..) will be almost 0 or very small neg and G(z) wants to maximise this
            ##
            ## note exact same objective would be to MAXIMISE log(D(G(z))) which intuitevely makes sense
            ## as 0 <= D <= 1
            ## now this is same as MINIMISE -log(D(G(z)))

            # Train the discriminator

            ## the line below is exactly the first term in V(D,G)
            ## Exp_{x}[log(D(x))] because the loss fn is binary cross entropy
            ## and u supply all labels as 'TRUE' i.e. y=1 always so 2nd term in
            ## binary cross entropy disappears and u are left with
            ## standard loss func would MINIMISE -log(p(x)) this is
            ## same as maximising the first term of V(D,G)
            ## 'p(x)' === D(x)
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)


            ## same situation as before but this time the binary crossentropy
            ## will have first term ==1 so only thing left is
            ## -log(1-p(G(z))) as all y supplied is y=0
            ## this is minimised.. which is same as 
            ## maximising log(1-D(G(z)))
            ## remember G(z) is the input here
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)

            ## the two terms are just added together and dived by two
            ## division is extra not same as paper but this is just a scale factor!
            ## i.e. doesn't affect optimal D or G in theoretical terms
            ## maybe used for stability
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Train the generator (to have the discriminator label samples as valid)
            ## now interesting things are happening here...
            ## first of all we only case about second term in our loss function
            ## because first is 0 w.r.t G when differentiated so is essentially
            ## like a constant when choosing optimal G...
            ## what happens in internally this function will use 'combined' model
            ## which calcs D(G(z))
            ## but also keeps weights of D fixed so in back-prop only G is updated here
            ## furthermore you can see that the valid or 1,...,1 tag is supplied so in loss
            ## function only 1 term will remain namely the loss obj will be 
            ## minimise -log(D(G(z)))
            ## 
            ## the original paper objective was min log(1-D(G(z)))
            ## which is same as maximise log(D(G(z)))
            ## which is same as minimise -log(D(G(z)))!!
            ## notice we ignre expectation here as its just a summation
            ## of minibatch losses and divide by minibatch size.
            g_loss = self.combined.train_on_batch(noise, valid)

            ## this 'train_on_batch' command computes the weights and updates the
            ## model first it updates weights of D by only supplying {x_batch, ones} and {G(z)_batch, zeros}
            ##
            ## next it updates weights of G by supplying {z_new_batch, ones} 
            ## generating/converting the list of z's to G(z)'s
            ## computing weights of G only (by freezing D in model)
            ## notice how in both stages the z_batch samples are DIFFERENT but they come from same 
            ## distribution so we want it to match the distributions of G(z) and x and not the exact
            ## transformation so transformation function shuld look to transform
            ## the distribution of G(z) to distribution of x

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)

    def sample_images(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    gan = GAN()
    gan.train(epochs=30000, batch_size=32, sample_interval=200)
