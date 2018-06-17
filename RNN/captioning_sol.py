import tensorflow as tf
import numpy as np
import os

#os.environ['CUDA_VISIBLE_DEVICES'] = "0"
#conf = tf.ConfigProto()
#conf.gpu_options.per_process_gpu_memory_fraction = 0.2


class Captioning():
    def __init__(self, word_to_idx, dim_in, dim_embed, dim_hidden, n_lstm_steps, n_words, init_b=None):
        self._null  = word_to_idx['<NULL>']
        self._start = word_to_idx['<START>']
        self._end   = word_to_idx['<END>']
        
        self.dim_in = dim_in
        self.dim_embed = dim_embed
        self.dim_hidden = dim_hidden
        self.n_lstm_steps = n_lstm_steps
        self.n_words = n_words
        
        # declare the variables to be used for our word embeddings
        self.word_embedding = tf.Variable(tf.random_uniform([self.n_words, self.dim_embed], -0.1, 0.1), name='word_embedding')
        self.embedding_bias = tf.Variable(tf.zeros([dim_embed]), name='embedding_bias')
        
        # declare the LSTM itself
        self.lstm = tf.contrib.rnn.BasicLSTMCell(dim_hidden)
        
        # declare the variables to be used to embed the image feature embedding to the word embedding space
        self.img_embedding = tf.Variable(tf.random_uniform([dim_in, dim_hidden], -0.1, 0.1), name='img_embedding')
        self.img_embedding_bias = tf.Variable(tf.zeros([dim_hidden]), name='img_embedding_bias')

        # declare the variables to go from an LSTM output to a word encoding output
        self.word_encoding = tf.Variable(tf.random_uniform([dim_hidden, self.n_words], -0.1, 0.1), name='word_encoding')
        
        # optional initialization setter for encoding bias variable 
        if init_b is not None:
            self.word_encoding_bias = tf.Variable(init_b, name='word_encoding_bias')
        else:
            self.word_encoding_bias = tf.Variable(tf.zeros([self.n_words]), name='word_encoding_bias')

    def build_model(self, batch_size=128):
        # declaring the placeholders for our extracted image feature vectors, our caption, and our mask
        # (describes how long our caption is with an array of 0/1 values of length `maxlen`  
        self.batch_size = batch_size
        img = tf.placeholder(tf.float32, [self.batch_size, self.dim_in])
        caption_placeholder = tf.placeholder(tf.int32, [self.batch_size, self.n_lstm_steps])
        mask = tf.placeholder(tf.float32, [self.batch_size, self.n_lstm_steps])
        
        # getting an initial LSTM embedding from our image_imbedding
        image_embedding = tf.matmul(img, self.img_embedding) + self.img_embedding_bias
        
        # setting initial state of our LSTM
        state = self.lstm.zero_state(self.batch_size, dtype=tf.float32)

        total_loss = 0.0
        with tf.variable_scope("RNN"):
            for i in range(self.n_lstm_steps): 
                if i > 0:
                    # if this is not the first iteration of our LSTM we need to get the word_embedding corresponding
                    # to the (i-1)th word in our caption 
                    current_embedding = tf.nn.embedding_lookup(self.word_embedding, caption_placeholder[:,i-1])\
                                        + self.embedding_bias
                else:
                    #if this is the first iteration of our LSTM we utilize the embedded image as our input 
                    current_embedding = image_embedding
                if i > 0: 
                    # allows us to reuse the LSTM tensor variable on each iteration
                    tf.get_variable_scope().reuse_variables()

                out, state = self.lstm(current_embedding, state)

                
                if i > 0:
                    #get the one-hot representation of the next word in our caption 
                    labels = tf.expand_dims(caption_placeholder[:, i], 1)
                    ix_range=tf.range(0, self.batch_size, 1)
                    ixs = tf.expand_dims(ix_range, 1)
                    concat = tf.concat([ixs, labels],1)
                    onehot = tf.sparse_to_dense(
                                    concat, tf.stack([self.batch_size, self.n_words]), 1.0, 0.0)

                    #perform a softmax classification to generate the next word in the caption
                    logit = tf.matmul(out, self.word_encoding) + self.word_encoding_bias
                    xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logit, labels=onehot)
                    xentropy = xentropy * mask[:,i]

                    loss = tf.reduce_sum(xentropy)
                    total_loss += loss

        total_loss = total_loss / tf.reduce_sum(mask[:,1:])
        return total_loss, img,  caption_placeholder, mask


    def predict(self, batch_size=1):

        #same setup as `build_model` function 
        self.batch_size = batch_size
        img = tf.placeholder(tf.float32, [self.batch_size, self.dim_in])
        image_embedding = tf.matmul(img, self.img_embedding) + self.img_embedding_bias
        state = self.lstm.zero_state(self.batch_size,dtype=tf.float32)

        N = self.batch_size
        #declare list to hold the words of our generated captions
        captions = self._null*np.ones((N, 1), dtype=np.int32)
        captions[:,0]=self._start

        with tf.variable_scope("RNN"):
            # in the first iteration we have no previous word, so we directly pass in the image embedding
            # and set the `previous_word` to the embedding of the start token ([0]) for the future iterations
            output, state = self.lstm(image_embedding, state)
            previous_word = tf.nn.embedding_lookup(self.word_embedding, captions[:,0]) + self.embedding_bias
            for i in range(self.n_lstm_steps-1): 
                tf.get_variable_scope().reuse_variables()

                out, state = self.lstm(previous_word, state)

                # get a get maximum probability word and it's encoding from the output of the LSTM
                logit = tf.matmul(out, self.word_encoding) + self.word_encoding_bias
                best_word = tf.argmax(logit, 1)

                # get the embedding of the best_word to use as input to the next iteration of our LSTM 
                previous_word = tf.nn.embedding_lookup(self.word_embedding, best_word)

                previous_word += self.embedding_bias

                best_word = tf.reshape( best_word, [N,1])
                captions = tf.concat([captions,best_word],1)

        return img, captions 


### Parameters ###
dim_embed = 256
dim_hidden = 256
dim_in = 512
batch_size = 128 #128
momentum = 0.9
n_epochs = 3 

def cap_train(img_features, captions, maxlen, n_words, word_to_idx, model_path):
        dim_in = img_features.shape[1]
        learning_rate=0.001

        feats = img_features
        caption_matrix = captions
        caption_mask_matrix = np.zeros((caption_matrix.shape[0], caption_matrix.shape[1]))
        nonzeros = np.array([x for x in map(lambda x: (x != 0).sum(), caption_matrix )])
        for ind, row in enumerate(caption_mask_matrix): 
                row[:nonzeros[ind]] = 1
            
        tf.reset_default_graph()

        index = (np.arange(len(feats)).astype(int))
        np.random.shuffle(index)

        sess = tf.InteractiveSession( )

        caption_generator = Captioning(word_to_idx, dim_in, dim_hidden, dim_embed, maxlen, n_words)

        loss, image, sentence, mask = caption_generator.build_model(batch_size)

        saver = tf.train.Saver(max_to_keep=100)
        global_step=tf.Variable(0,trainable=False)
        learning_rate = tf.train.exponential_decay(learning_rate, global_step,
                                         int(len(index)/batch_size), 0.95)
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        tf.global_variables_initializer().run()

        losses=[]

        for epoch in range(n_epochs):
            for start, end in zip( range(0, len(index), batch_size), range(batch_size, len(index), batch_size)):

                current_feats = feats[index[start:end]]
                current_caption_matrix = caption_matrix[index[start:end]]
                current_mask_matrix = caption_mask_matrix[index[start:end]]
                _, loss_value = sess.run([train_op, loss], feed_dict={
                    image: current_feats.astype(np.float32),
                    sentence : current_caption_matrix.astype(np.int32),
                    mask : current_mask_matrix.astype(np.float32)
                    })

            if epoch%100==0 :
                #loss_value = 0                
                print("Current Cost: ", loss_value, "\t Epoch {}/{}".format(epoch, n_epochs),\
                      "\t Iter {}/{}".format(start,len(feats)))
        saver.save(sess, os.path.join(model_path, 'model'))
        sess.close()


def img_captioning(word_to_idx, maxlen, n_words, model_path, features) :
        tf.reset_default_graph()
        sess = tf.InteractiveSession( )

        batch_size=features.shape[0]
        model = Captioning(word_to_idx, dim_in, dim_hidden, dim_embed, maxlen, n_words)

        image, generated_words = model.predict(batch_size)

        saver = tf.train.Saver()
        saved_path=tf.train.latest_checkpoint(model_path)
        saver.restore(sess, saved_path)

        pr_captions = sess.run(generated_words, feed_dict={image:features})
        sess.close()
        return pr_captions
