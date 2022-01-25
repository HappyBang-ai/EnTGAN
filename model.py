from collections import namedtuple
from utils import *
from ops import *
import time
from glob import glob
import cv2
import numpy as np

def EGVector_Extractor(input_tensor, Bmin, Bmax, Cmin, Cmax, Smin, Smax):
    img = tf.squeeze(input_tensor)
    Gray_img = 0.299*img[:,:,0] +0.587*img[:,:,1]+0.114*img[:,:,2]
    Gray_img = Gray_img / 255.
    Brightness = tf.math.reduce_mean(Gray_img)
    Contrast = tf.math.sqrt(tf.math.reduce_mean(tf.math.square(Gray_img - Brightness)))
    hsv = tf.image.rgb_to_hsv(img)    
    hsv = hsv[:,:,1]
    Sat = tf.math.reduce_mean(hsv)
    vector = [Brightness, Contrast, Sat]
    
    vector[0] = 2*((vector[0]-Bmin)/(Bmax-Bmin))-1
    vector[1] = 2*((vector[1]-Cmin)/(Cmax-Cmin))-1
    vector[2] = 2*((vector[2]-Smin)/(Smax-Smin))-1
    
    return tf.reshape(vector, shape=[1,1,3,1])


def gaussian_noise_layer(input_layer, std):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)
    return input_layer+noise

def fully_connected(x, units, use_bias=True, scope='fully_connected'):
    with tf.variable_scope(scope):
        x = flatten(x)
        x = tf.layers.dense(x, units=units, kernel_initializer=weight_init,
                            kernel_regularizer=weight_regularizer,
                            use_bias=use_bias)
        return x

def flatten(x) :
    return tf.layers.flatten(x)


def adaptive_instance_norm(content, gamma, beta, epsilon=1e-5):


    c_mean, c_var = tf.nn.moments(content, axes=[1, 2], keep_dims=True)
    c_std = tf.sqrt(c_var + epsilon)

    return gamma * ((content - c_mean) / c_std) + beta


def MLP(style, scope='MLP'):
        channel = 256
        with tf.variable_scope(scope) :
            x = style

            for i in range(2):
                x = fully_connected(x, channel, scope='FC_' + str(i))
                x = tf.nn.relu(x)
            
            mu_list = []
            var_list = []
            mu = fully_connected(x, channel, scope='FC_mu_' + str(i))
            var = fully_connected(x, channel, scope='FC_var_' + str(i))
            mu = tf.reshape(mu, shape=[-1, 1, 1, channel])
            var = tf.reshape(var, shape=[-1, 1, 1, channel])
            mu_list.append(mu)
            var_list.append(var)
            return mu, var


def generator_resnet(image, style, options, reuse=False, name="generator"):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
        def residule_block_dilated(x, dim, ks=3, s=1, name='res'):
            y = instance_norm(dilated_conv2d(x, dim, ks, s, padding='SAME', name=name + '_c1'), name + '_bn1')
            y = tf.nn.relu(y)
            y = instance_norm(dilated_conv2d(y, dim, ks, s, padding='SAME', name=name + '_c2'), name + '_bn2')
            return y + x

        
        
        ### Encoder architecture
        c0 = tf.pad(image, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        c1 = tf.nn.relu(instance_norm(conv2d(c0, options.gf_dim, 7, 1, padding='VALID', name='g_e1_c'), 'g_e1_bn'))
        c2 = tf.nn.relu(instance_norm(conv2d(c1, options.gf_dim * 2, 3, 2, name='g_e2_c'), 'g_e2_bn'))
        c3 = tf.nn.relu(instance_norm(conv2d(c2, options.gf_dim * 4, 3, 2, name='g_e3_c'), 'g_e3_bn'))
        r1 = residule_block_dilated(c3, options.gf_dim * 4, name='g_r1')
        r2 = residule_block_dilated(r1, options.gf_dim * 4, name='g_r2')
        r3 = residule_block_dilated(r2, options.gf_dim * 4, name='g_r3')
        r4 = residule_block_dilated(r3, options.gf_dim * 4, name='g_r4')
        r5 = residule_block_dilated(r4, options.gf_dim * 4, name='g_r5')
        

        
        ### translation decoder architecture
        mu, var = MLP(style)
        adain_r5 = adaptive_instance_norm(r5, mu, var)
        
        r6 = residule_block_dilated(adain_r5, options.gf_dim * 4, name='g_r6')
        r7 = residule_block_dilated(r6, options.gf_dim * 4, name='g_r7')
        r8 = residule_block_dilated(r7, options.gf_dim * 4, name='g_r8')
        r9 = residule_block_dilated(r8, options.gf_dim * 4, name='g_r9')
        d1 = deconv2d(r9, options.gf_dim * 2, 3, 2, name='g_d1_dc')
        d1 = tf.nn.relu(instance_norm(d1, 'g_d1_bn'))
        d2 = deconv2d(d1, options.gf_dim, 3, 2, name='g_d2_dc')
        d2 = tf.nn.relu(instance_norm(d2, 'g_d2_bn'))
        d2 = tf.pad(d2, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        pred = tf.nn.tanh(conv2d(d2, options.output_c_dim, 7, 1, padding='VALID', name='g_pred_c'))

        return pred,r5

def domain_agnostic_classifier(percep,options, reuse=False,name="percep"):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
        h1 = lrelu(instance_norm(conv2d(percep, options.df_dim * 4, name='d_h1_conv'), 'd_bn1'))
        h2 = lrelu(instance_norm(conv2d(h1, options.df_dim * 2, name='d_h2_conv'), 'd_bn2'))
        h3 = lrelu(instance_norm(conv2d(h2, options.df_dim * 2, name='d_h3_conv'), 'd_bn3'))
        h4 = conv2d(h3, 2, s=1, name='d_h3_pred')
        return tf.reshape(tf.reduce_mean(h4,axis=[0,1,2]),[-1,1,1,2])

def abs_criterion(in_, target):
    return tf.reduce_mean(tf.abs(in_ - target))


def mae_criterion(in_, target):
    return tf.reduce_mean((in_ - target) ** 2)


def sce_criterion(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))



def mae_criterion_list(in_, target):
    loss = 0.0
    for i in range(len(target)):
        loss+=tf.reduce_mean((in_[i]-target[i])**2)
    return loss / len(target)


def sce_criterion_list(logits, labels):
    loss = 0.0
    for i in range(len(labels)):
        loss+=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits[i], labels=labels[i]))
    return loss/len(labels)



epsilon = 1e-9

class Entgan(object):
    def __init__(self, sess, args):
        self.sess = sess
        self.batch_size = args.batch_size
        self.image_size = args.fine_size
        self.input_c_dim = args.input_nc
        self.output_c_dim = args.output_nc
        self.L1_lambda = args.L1_lambda
        self.dataset_dir = args.dataset_dir
        self.n_d = args.n_d
        self.n_scale = args.n_scale
        self.ndf= args.ndf
        self.load_size =args.load_size
        self.fine_size =args.fine_size
        self.generator = generator_resnet
        self.MLP = MLP
        self.domain_agnostic_classifier=domain_agnostic_classifier
        if args.use_lsgan:
            self.criterionGAN = mae_criterion
            self.criterionGAN_list = mae_criterion_list
        else:
            self.criterionGAN = sce_criterion
            self.criterionGAN_list = sce_criterion_list

        OPTIONS = namedtuple('OPTIONS', 'batch_size image_size \
                              gf_dim df_dim output_c_dim is_training')
        self.options = OPTIONS._make((args.batch_size, args.fine_size,
                                      args.ngf, args.ndf//args.n_d, args.output_nc,
                                      args.phase == 'train'))

        self._build_model()
        self.saver = tf.train.Saver(max_to_keep = None)
        self.pool = ImagePool(args.max_size)

    def discriminator(self,image, options, reuse=False, name="discriminator"):
        images = []
        for i in range(self.n_scale):
            images.append(tf.image.resize_bicubic(image, [get_shape(image)[1]//(2**i),get_shape(image)[2]//(2**i)]))
        with tf.variable_scope(name):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse is False
            images = dis_down(images,4,2,self.n_scale,options.df_dim,'d_h0_conv_scale_')
            images = dis_down(images,4,2, self.n_scale, options.df_dim*2, 'd_h1_conv_scale_')
            images = dis_down(images,4,2, self.n_scale, options.df_dim * 4, 'd_h2_conv_scale_')
            images = dis_down(images,4,2, self.n_scale, options.df_dim * 8, 'd_h3_conv_scale_')
            images = final_conv(images,self.n_scale,"d_pred_scale_")
            return images

    def _build_model(self):
        
        self.real_data = tf.placeholder(tf.float32,[self.batch_size, self.image_size, self.image_size*2,self.input_c_dim + self.output_c_dim],name='real_A_and_B_images')
        self.Bmax = tf.placeholder(tf.float32, name='Bmax')
        self.Bmin = tf.placeholder(tf.float32, name='Bmin')
        self.Cmax = tf.placeholder(tf.float32, name='Cmax')
        self.Cmin = tf.placeholder(tf.float32, name='Cmin')
        self.Smax = tf.placeholder(tf.float32, name='Smax')
        self.Smin = tf.placeholder(tf.float32, name='Smin')
        
        self.real_A = self.real_data[:, :, :, :self.input_c_dim]
        self.real_B = self.real_data[:, :, :, self.input_c_dim:self.input_c_dim + self.output_c_dim]

        img_A = (self.real_A+1)*127.5
        img_B = (self.real_B+1)*127.5
        
        self.A_vector = EGVector_Extractor(img_A, self.Bmin, self.Bmax, self.Cmin, self.Cmax, self.Smin, self.Smax)
        self.B_vector = EGVector_Extractor(img_B, self.Bmin, self.Bmax, self.Cmin, self.Cmax, self.Smin, self.Smax) 
    

        self.fake_A, self.realA_percep = self.generator(self.real_A, self.A_vector, self.options, False, name="generator") #A label 추가
        self.fake_A2B, _ = self.generator(self.real_A, self.B_vector, self.options,  True, name="generator") #B label 추가
        self.fake_A_rec, self.fake_A2B_percep = self.generator(self.fake_A2B, self.A_vector, self.options, True, name="generator") #A label 추가
        self.fake_B, self.realB_percep = self.generator(self.real_B, self.B_vector, self.options, True, name="generator") #B label 추가
        self.fake_B2A, _ = self.generator(self.real_B, self.A_vector, self.options, True, name="generator") #A label 추가
        self.fake_B_rec, self.fake_B2A_percep = self.generator(self.fake_B2A, self.B_vector, self.options, True, name="generator") #B label 추가


        fake_img_A = (self.fake_A+1)*127.5
        fake_img_B = (self.fake_B+1)*127.5
        
        fake_A_vector = EGVector_Extractor(fake_img_A, self.Bmin, self.Bmax, self.Cmin, self.Cmax, self.Smin, self.Smax)
        fake_B_vector = EGVector_Extractor(fake_img_B, self.Bmin, self.Bmax, self.Cmin, self.Cmax, self.Smin, self.Smax)
        
        fake_img_A2B = (self.fake_A2B+1)*127.5
        fake_img_B2A = (self.fake_B2A+1)*127.5
        
        fake_A2B_vector = EGVector_Extractor(fake_img_A2B, self.Bmin, self.Bmax, self.Cmin, self.Cmax, self.Smin, self.Smax)
        fake_B2A_vector = EGVector_Extractor(fake_img_B2A, self.Bmin, self.Bmax, self.Cmin, self.Cmax, self.Smin, self.Smax)


        fake_img_A_rec = (self.fake_A_rec+1)*127.5
        fake_img_B_rec = (self.fake_B_rec+1)*127.5
        
        fake_A_rec_vector = EGVector_Extractor(fake_img_A_rec, self.Bmin, self.Bmax, self.Cmin, self.Cmax, self.Smin, self.Smax)
        fake_B_rec_vector = EGVector_Extractor(fake_img_B_rec, self.Bmin, self.Bmax, self.Cmin, self.Cmax, self.Smin, self.Smax) 



        self.Environment_translation_loss = 0.5*(tf.reduce_sum(tf.abs(self.A_vector-fake_A_vector))+tf.reduce_sum(tf.abs(self.B_vector-fake_B_vector))\
                        +tf.reduce_sum(tf.abs(self.A_vector-fake_B2A_vector))+tf.reduce_sum(tf.abs(self.B_vector-fake_A2B_vector))\
                        +tf.reduce_sum(tf.abs(self.A_vector-fake_A_rec_vector))+tf.reduce_sum(tf.abs(self.B_vector-fake_B_rec_vector)))

        self.g_adv = 0.0
                
        self.percep_loss = 0.01*(tf.reduce_sum(tf.abs(self.realA_percep-self.fake_A2B_percep))\
                           +tf.reduce_sum(tf.abs(self.realB_percep-self.fake_B2A_percep)))

                           
        for i in range(self.n_d):
            self.DB_fake = self.discriminator(self.fake_B, self.options, reuse=False, name=str(i)+"_discriminator")
            self.DA_fake = self.discriminator(self.fake_A, self.options, reuse=True, name=str(i)+"_discriminator")

            self.D_B2A = self.discriminator(self.fake_B2A, self.options, reuse=True, name=str(i) + "_discriminator")
            self.D_A2B = self.discriminator(self.fake_A2B, self.options, reuse=True, name=str(i) + "_discriminator")

            self.D_fakeA_rec = self.discriminator(self.fake_A_rec, self.options, reuse=True, name=str(i) + "_discriminator")
            self.D_fakeB_rec = self.discriminator(self.fake_B_rec, self.options, reuse=True, name=str(i) + "_discriminator")

            self.g_adv += 0.5*((self.criterionGAN_list(self.DA_fake, get_ones_like(self.DA_fake))+ self.criterionGAN_list(self.DB_fake, get_ones_like(self.DB_fake)))+
                               (self.criterionGAN_list(self.D_B2A,get_ones_like(self.D_B2A))+self.criterionGAN_list(self.D_A2B,get_ones_like(self.D_A2B)))+
                               (self.criterionGAN_list(self.D_fakeA_rec,get_ones_like(self.D_fakeA_rec)) + self.criterionGAN_list(self.D_fakeB_rec, get_ones_like(self.D_fakeB_rec))))


        self.Reconstruction_loss = abs_criterion(self.real_A, self.fake_A) + abs_criterion(self.real_B, self.fake_B)

        self.Cycle_consistency_loss = abs_criterion(self.fake_A_rec, self.real_A) + abs_criterion(self.fake_B_rec, self.real_B)

        self.g_loss = self.g_adv + self.L1_lambda * self.Reconstruction_loss + self.L1_lambda * self.Cycle_consistency_loss + self.percep_loss + self.Environment_translation_loss

        self.fake_A_sample = tf.placeholder(tf.float32,[self.batch_size, self.image_size, self.image_size*2,  self.output_c_dim], name='fake_A_sample')
        self.fake_B_sample = tf.placeholder(tf.float32,[self.batch_size, self.image_size, self.image_size*2,  self.output_c_dim], name='fake_B_sample')
        self.fake_A2B_sample = tf.placeholder(tf.float32,[self.batch_size, self.image_size, self.image_size * 2, self.output_c_dim],name='fake_A2B_sample')
        self.fake_B2A_sample = tf.placeholder(tf.float32,[self.batch_size, self.image_size, self.image_size * 2, self.output_c_dim],name='fake_B2A_sample')
        self.fake_A_rec_sample = tf.placeholder(tf.float32, [self.batch_size, self.image_size, self.image_size * 2,self.output_c_dim], name='fake_A_rec_sample')
        self.fake_B_rec_sample = tf.placeholder(tf.float32, [self.batch_size, self.image_size, self.image_size * 2,self.output_c_dim], name='fake_B_rec_sample')

        self.d_loss_item=[]

        for i in range(self.n_d):
            self.DB_real = self.discriminator(self.real_B, self.options, reuse=True, name=str(i)+"_discriminator")
            self.DA_real = self.discriminator(self.real_A, self.options, reuse=True, name=str(i)+"_discriminator")
            self.DB_fake_sample = self.discriminator(self.fake_B_sample, self.options, reuse=True, name=str(i)+"_discriminator")
            self.DA_fake_sample = self.discriminator(self.fake_A_sample, self.options, reuse=True, name=str(i)+"_discriminator")
            self.db_loss_real = self.criterionGAN_list(self.DB_real, get_ones_like(self.DB_real))
            self.db_loss_fake = self.criterionGAN_list(self.DB_fake_sample, get_zeros_like(self.DB_fake_sample))
            self.db_loss = (self.db_loss_real * 0.5 + self.db_loss_fake * 0.5)
            self.da_loss_real = self.criterionGAN_list(self.DA_real, get_ones_like(self.DA_real))
            self.da_loss_fake = self.criterionGAN_list(self.DA_fake_sample, get_zeros_like(self.DA_fake_sample))
            self.da_loss = (self.da_loss_real * 0.5 + self.da_loss_fake * 0.5)
            
        
            self.DB_fake_A2B_sample = self.discriminator(self.fake_A2B_sample, self.options, reuse=True, name=str(i)+"_discriminator")
            self.DA_fake_B2A_sample = self.discriminator(self.fake_B2A_sample, self.options, reuse=True, name=str(i)+"_discriminator")
            
            self.db_loss_a2b = self.criterionGAN_list(self.DB_fake_A2B_sample, get_zeros_like(self.DB_fake_A2B_sample))
            self.db_loss2 = (self.db_loss_real * 0.5 + self.db_loss_a2b * 0.5)
            
            self.da_loss_b2a = self.criterionGAN_list(self.DA_fake_B2A_sample, get_zeros_like(self.DA_fake_B2A_sample))
            self.da_loss2 = (self.da_loss_real * 0.5 + self.da_loss_b2a * 0.5)
            
            self.DB_fake_B_rec_sample = self.discriminator(self.fake_B_rec_sample, self.options, reuse=True, name=str(i)+"_discriminator")
            self.DA_fake_A_rec_sample = self.discriminator(self.fake_A_rec_sample, self.options, reuse=True, name=str(i)+"_discriminator")
            
            self.db_loss_rec_B = self.criterionGAN_list(self.DB_fake_B_rec_sample, get_zeros_like(self.DB_fake_B_rec_sample))
            self.db_loss3 = (self.db_loss_real * 0.5 + self.db_loss_rec_B * 0.5)
            
            self.da_loss_rec_A = self.criterionGAN_list(self.DA_fake_A_rec_sample, get_zeros_like(self.DA_fake_A_rec_sample))
            self.da_loss3 = (self.da_loss_real * 0.5 + self.da_loss_rec_A * 0.5)
            
            
            self.d_loss = (self.da_loss + self.db_loss)+(self.da_loss2 + self.db_loss2)+(self.da_loss3 + self.db_loss3)
            self.d_loss_item.append(self.d_loss)
        
        self.test_A = tf.placeholder(tf.float32,[self.batch_size, self.image_size, self.image_size*2,self.input_c_dim], name='test_A')
        self.B_vector_test = tf.placeholder(tf.float32, name='B_vector_test')

        img_A_test = (self.test_A+1)*127.5
        
        
        self.A_vector_test = EGVector_Extractor(img_A_test, self.Bmin, self.Bmax, self.Cmin, self.Cmax, self.Smin, self.Smax)
        self.B_vector_test = tf.reshape(self.B_vector_test, shape=[1,1,3,1])
        
        
        
        self.test_A2B, self.testA_percep = self.generator(self.test_A, self.B_vector_test, self.options, True, name="generator") #B label 추가
        self.test_A_fake, _ = self.generator(self.test_A, self.A_vector_test, self.options, True, name="generator") #A label 추가
        self.test_A_rec, _ = self.generator(self.test_A2B, self.A_vector_test, self.options, True, name="generator") #A label 추가

        t_vars = tf.trainable_variables()

        
        self.g_vars = [var for var in t_vars if 'generator' in var.name]
        self.d_vars_item=[]
        for i in range(self.n_d):
            self.d_vars=[var for var in t_vars if str(i)+'_discriminator' in var.name]
            self.d_vars_item.append(self.d_vars)

    def train(self, args):
        """Train Entgan"""
        self.lr = tf.placeholder(tf.float32, None, name='learning_rate')

        
        ### generator
        self.g_optim = tf.train.AdamOptimizer(self.lr, beta1=args.beta1) \
            .minimize(self.g_loss, var_list=self.g_vars)
        
        
        ### translation
        self.d_optim_item=[]
        for i in range(self.n_d):
            self.d_optim = tf.train.AdamOptimizer(self.lr, beta1=args.beta1) \
                .minimize(self.d_loss_item[i], var_list=self.d_vars_item[i])
            self.d_optim_item.append(self.d_optim)
        
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        self.writer = tf.summary.FileWriter(os.path.join(args.checkpoint_dir,"logs"), self.sess.graph)

        counter = 1
        start_time = time.time()
        img_path_A = './datasets/{}/*.*'.format(self.dataset_dir + '/trainA')
        img_path_B = './datasets/{}/*.*'.format(self.dataset_dir + '/trainB')
        Bmax, Cmax, Smax, Bmin, Cmin, Smin = vector_max_min(img_path_A, img_path_B)

        print("Brightness max : {}".format(Bmax))
        print("Brightness min : {}".format(Bmin))
        print("Contrast max : {}".format(Cmax))
        print("Contrast min : {}".format(Cmin))
        print("Saturation max : {}".format(Smax))
        print("Saturation min : {}".format(Smin))

        if args.continue_train:
            if self.load(args.checkpoint_dir):
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")

        for epoch in range(args.epoch):
            dataA = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/trainA'))
            dataB = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/trainB'))
            np.random.shuffle(dataA)
            np.random.shuffle(dataB)
            
            
            batch_idxs = min(min(len(dataA), len(dataB)), args.train_size) // self.batch_size
            lr = args.lr if epoch < args.epoch_step else args.lr * (args.epoch - epoch) / (args.epoch - args.epoch_step)

            for idx in range(0, batch_idxs):
                batch_files = list(zip(dataA[idx * self.batch_size:(idx + 1) * self.batch_size],dataB[idx * self.batch_size:(idx + 1) * self.batch_size]))
                
                batch_images = [load_train_data(batch_file, args.load_size, args.fine_size) for batch_file in batch_files]
                
                
                # Update G network and record fake outputs
                fake_A,fake_B,fake_A2B,fake_B2A,fake_A_rec,fake_B_rec,_,g_loss,gan_loss,percep , A_vector_tensor, B_vector_tensor, Environment_translation_loss = self.sess.run(
                    [self.fake_A, self.fake_B,self.fake_A2B,self.fake_B2A,self.fake_A_rec,self.fake_B_rec, self.g_optim,self.g_loss,self.g_adv,self.percep_loss, self.A_vector, self.B_vector,self.Environment_translation_loss],
                    feed_dict={self.real_data: batch_images,self.lr: lr, self.Bmax: Bmax, self.Bmin: Bmin, self.Cmax: Cmax, self.Cmin: Cmin, self.Smax: Smax, self.Smin: Smin})
                
                [fake_A, fake_B] = self.pool([fake_A, fake_B])
                [fake_A2B, fake_B2A] = self.pool([fake_A2B, fake_B2A])
                [fake_A_rec, fake_B_rec] = self.pool([fake_A_rec, fake_B_rec])

                # Update D network
                loss_print=[]
                for i in range(self.n_d):
                    _, d_loss = self.sess.run(
                        [self.d_optim_item[i], self.d_loss_item[i]],
                        feed_dict={self.real_data: batch_images,
                                   self.fake_A_sample: fake_A,
                                   self.fake_B_sample: fake_B,
                                   self.fake_A2B_sample: fake_A2B,
                                   self.fake_B2A_sample: fake_B2A,
                                   self.fake_A_rec_sample: fake_A_rec,
                                   self.fake_B_rec_sample: fake_B_rec,self.lr: lr})
                    
                    loss_print.append(d_loss)

                counter += 1
                print(("Epoch: [%2d] [%4d/%4d] time: %4.4f g_loss: %4.4f g_adv loss:%4.4f g_percep_loss:%4.4f g_env_loss:%4.4f" % (
                    epoch, idx, batch_idxs, time.time() - start_time,g_loss,gan_loss,percep,Environment_translation_loss)))
                print(loss_print)

                if np.mod(counter, args.print_freq) == 1:
                    self.sample_model(args.sample_dir, epoch, idx)

                if np.mod(counter, args.save_freq) == 2:
                    self.save(args.checkpoint_dir, counter)

    def save(self, checkpoint_dir, step):
        model_name = "Entgan.model"
        model_dir = "%s_%s" % (self.dataset_dir, self.image_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        model_dir = "%s_%s" % (self.dataset_dir, self.image_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def sample_model(self, sample_dir, epoch, idx):
        img_path_A = './datasets/{}/*.*'.format(self.dataset_dir + '/trainA')
        img_path_B = './datasets/{}/*.*'.format(self.dataset_dir + '/trainB')
        Bmax, Cmax, Smax, Bmin, Cmin, Smin = vector_max_min(img_path_A, img_path_B)
       
        dataA = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/trainA'))
        dataB = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/trainB'))
        np.random.shuffle(dataA)
        np.random.shuffle(dataB)
        batch_files = list(zip(dataA[:self.batch_size], dataB[:self.batch_size]))
        sample_images = [load_train_data(batch_file,self.load_size,self.fine_size, is_testing=True) for batch_file in batch_files]
        sample_images = np.array(sample_images).astype(np.float32)


        fake_A,fake_B,fake_A2B,fake_B2A,fake_A_rec,fake_B_rec = self.sess.run(
            [self.fake_A, self.fake_B,self.fake_A2B,self.fake_B2A,self.fake_A_rec,self.fake_B_rec],
            feed_dict={self.real_data: sample_images,self.Bmax: Bmax, self.Bmin: Bmin, self.Cmax: Cmax, self.Cmin: Cmin, self.Smax: Smax, self.Smin: Smin}
        )
        real_A = sample_images[:, :, :, :3]
        real_B = sample_images[:, :, :, 3:]


        merge_A = np.concatenate([real_B, fake_B,fake_B2A,fake_B_rec], axis=2)
        merge_B = np.concatenate([real_A, fake_A,fake_A2B,fake_A_rec], axis=2)
        check_folder('./{}/{:02d}'.format(sample_dir, epoch))
        save_images(merge_A, [self.batch_size, 1],
                    './{}/{:02d}/A_{:04d}.jpg'.format(sample_dir, epoch, idx))
        save_images(merge_B, [self.batch_size, 1],
                    './{}/{:02d}/B_{:04d}.jpg'.format(sample_dir, epoch, idx))

    def test(self, args):
        """Test Entgan"""
        img_path_A = './datasets/{}/*.*'.format(self.dataset_dir + '/trainA')
        img_path_B = './datasets/{}/*.*'.format(self.dataset_dir + '/trainB')
        Bmax, Cmax, Smax, Bmin, Cmin, Smin = vector_max_min(img_path_A, img_path_B)
        
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        
        sample_files = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/test'))

        if self.load(args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        
        out_var,fake_var,in_var,rec_var,vector_var = (self.test_A2B, self.test_A_fake, self.test_A,self.test_A_rec,self.B_vector_test)
        
        for sample_file in sample_files:
            print('Processing image: ' + sample_file)
            sample_image = [load_test_data(sample_file, args.fine_size)]
            sample_image = np.array(sample_image).astype(np.float32)
            image_path = os.path.join(args.test_dir,'{0}_{1}'.format("Entgan_", os.path.basename(sample_file)))
            
            B_vector_test = np.random.uniform(-1.0, 1.0, 3)
            print("Environment Guide Vector : {}".format(B_vector_test))
            B_vector_test = np.reshape(B_vector_test, (1,1,3,1))
            
            fake_img, trans_img,rec_img = self.sess.run([fake_var, out_var,rec_var], feed_dict={in_var: sample_image, vector_var: B_vector_test, self.Bmax: Bmax, self.Bmin: Bmin, self.Cmax: Cmax, self.Cmin: Cmin, self.Smax: Smax, self.Smin: Smin })
            merge=np.concatenate([sample_image,fake_img,trans_img,rec_img],axis=2)
            image_path_trans = os.path.join(args.test_dir,'{0}_{1}'.format("Translation_", os.path.basename(sample_file)))
            save_images(trans_img, [1, 1], image_path_trans)
            save_images(merge, [1, 1], image_path)
