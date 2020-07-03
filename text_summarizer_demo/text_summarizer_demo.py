"""
DOCSTRING
"""
import collections
import keras
import Levenshtein
import matplotlib.pyplot as pyplot
import numpy
import os
import pickle
import random
import sklearn
import sys
import zipfile

# variables
BATCH_SIZE = 64
EMPTY = 0
EOS = 1
MAXLEND = 25 # 0 - if we dont want to use description at all
MAXLENH = 25
MODEL = keras.models.Sequential()
RNN_SIZE = 512 # must be same as 160330-word-gen
SEED = 42
# derivative variables
ACTIVATION_RNN_SIZE = 40 if MAXLEND else 0
MAXLEN = MAXLEND + MAXLENH

class SimpleContext(keras.layers.core.Lambda):
    """
    DOCSTRING
    """
    def __init__(self, **kwargs):
        super(SimpleContext, self).__init__(Train.simple_context, **kwargs)
        self.supports_masking = True

    def compute_mask(self, input, input_mask=None):
        return input_mask[:,MAXLEND:]
    
    def get_output_shape_for(self, input_shape):
        nb_samples = input_shape[0]
        n = 2 * (RNN_SIZE - ACTIVATION_RNN_SIZE)
        return (nb_samples, MAXLENH, n)

class Train:
    """
    DOCSTRING
    """
    def __init__(self):
        """
        DOCSTRING
        """
        rnn_layers = 3
        batch_norm = False
        # training parameters
        p_W, p_U, p_dense, p_emb, weight_decay = 0, 0, 0, 0, 0
        self.optimizer = 'adam'
        self.LR = 1e-4
        self.nflips = 10
        self.nb_train_samples = 30000
        self.nb_val_samples = 3000
        with open('data/%s.pkl'%'vocabulary-embedding', 'rb') as fp:
            embedding, self.idx2word, word2idx, self.glove_idx2idx = pickle.load(fp)
        self.vocab_size, embedding_size = embedding.shape
        with open('data/%s.data.pkl'%'vocabulary-embedding', 'rb') as fp:
            X, Y = pickle.load(fp)
        self.nb_unknown_words = 10
        print('number of examples', len(X), len(Y))
        print('dimension of embedding space for words', embedding_size)
        print('vocabulary size', self.vocab_size)
        print('last %d words can be used as place holders for unknown words' % self.nb_unknown_words)
        print('total number of different words', len(self.idx2word), len(word2idx))
        print('number of words outside vocabulary which we can substitute using glove similarity',
              len(self.glove_idx2idx))
        print('number of words that will be regarded as unknown(unk)/out-of-vocabulary(oov):',
              len(self.idx2word) - self.vocab_size - len(self.glove_idx2idx))
        for i in range(self.nb_unknown_words):
            self.idx2word[self.vocab_size-1-i] = '<%d>'%i
        self.oov0 = self.vocab_size - self.nb_unknown_words
        for i in range(self.oov0, len(self.idx2word)):
            self.idx2word[i] = self.idx2word[i] + '^'
        self.X_train, self.X_test, self.Y_train, self.Y_test = (
            sklearn.model_selection.train_test_split(
                X, Y, test_size=self.nb_val_samples, random_state=SEED))
        len(self.X_train), len(self.Y_train), len(self.X_test), len(self.Y_test)
        del X
        del Y
        self.idx2word[EMPTY] = '_'
        self.idx2word[EOS] = '~'
        i = 334
        self.prt('H', self.Y_train[i])
        self.prt('D', self.X_train[i])
        i = 334
        self.prt('H', self.Y_test[i])
        self.prt('D', self.X_test[i])
        # seed weight initialization
        random.seed(SEED)
        numpy.random.seed(SEED)
        self.regularizer = keras.regularizers.l2(weight_decay) if weight_decay else None
        MODEL.add(keras.layers.embeddings.Embedding(
            self.vocab_size, embedding_size, input_length=MAXLEN,
            W_regularizer=self.regularizer, dropout=p_emb, weights=[embedding],
            mask_zero=True, name='embedding_1'))
        for i in range(rnn_layers):
            lstm = keras.layers.recurrent.LSTM(
                RNN_SIZE, return_sequences=True, W_regularizer=self.regularizer,
                U_regularizer=self.regularizer, b_regularizer=self.regularizer,
                dropout_W=p_W, dropout_U=p_U, name='lstm_%d'%(i+1))
            MODEL.add(lstm)
            MODEL.add(keras.layers.core.Dropout(p_dense, name='dropout_%d' % (i+1)))

    def __call__(self):
        """
        DOCSTRING
        """
        if ACTIVATION_RNN_SIZE:
            MODEL.add(SimpleContext(name='simplecontext_1'))
        MODEL.add(keras.layers.wrappers.TimeDistributed(keras.layers.core.Dense(
            self.vocab_size, W_regularizer=self.regularizer,
            b_regularizer=self.regularizer, name='timedistributed_1')))
        MODEL.add(keras.layers.core.Activation('softmax', name='activation_1'))
        # opt = keras.optimizers.Adam(lr=self.LR)  # reduce learning rate
        MODEL.compile(loss='categorical_crossentropy', optimizer=self.optimizer)
        keras.backend.set_value(MODEL.optimizer.lr, numpy.float32(self.LR))
        self.inspect_model(MODEL)
        if 'train' and os.path.exists('data/%s.hdf5'%'train'):
            MODEL.load_weights('data/%s.hdf5'%'train')
        samples = [self.lpadd([3]*26)]
        # pad from right (post) so the first maxlend will be description followed by headline
        data = keras.preprocessing.sequence.pad_sequences(
            samples, maxlen=MAXLEN, value=EMPTY, padding='post', truncating='post')
        numpy.all(data[:,MAXLEND] == EOS)
        data.shape, map(len, samples)
        probs = MODEL.predict(data, verbose=0, batch_size=1)
        probs.shape
        self.gensamples(skips=2, batch_size=BATCH_SIZE, k=10, temperature=1.0)
        self.test_gen(self.gen(self.X_train, self.Y_train, batch_size=BATCH_SIZE))
        self.test_gen(self.gen(
            self.X_train, self.Y_train, nflips=6, model=MODEL, debug=False,
            batch_size=BATCH_SIZE))
        valgen = self.gen(self.X_test, self.Y_test, nb_batches=3, batch_size=BATCH_SIZE)
        for i in range(4):
            self.test_gen(valgen, 1)
        history = {}
        traingen = self.gen(
            self.X_train, self.Y_train, batch_size=BATCH_SIZE,
            nflips=self.nflips, model=MODEL)
        valgen = self.gen(
            self.X_test, self.Y_test, nb_batches=self.nb_val_samples//BATCH_SIZE,
            batch_size=BATCH_SIZE)
        r = next(traingen)
        r[0].shape, r[1].shape, len(r)
        for iteration in range(500):
            print('Iteration', iteration)
            h = MODEL.fit_generator(
                traingen, samples_per_epoch=self.nb_train_samples, nb_epoch=1,
                validation_data=valgen, nb_val_samples=self.nb_val_samples)
            for k,v in h.history.iteritems():
                history[k] = history.get(k,[]) + v
            with open('data/%s.history.pkl' % 'train', 'wb') as fp:
                pickle.dump(history, fp, -1)
            MODEL.save_weights('data/%s.hdf5' % 'train', overwrite=True)
            self.gensamples(batch_size=BATCH_SIZE)

    def beamsearch(
        self,
        predict,
        start,
        k=1,
        maxsample=MAXLEN,
        use_unk=True,
        empty=EMPTY,
        eos=EOS,
        temperature=1.0):
        """
        variation to https://github.com/ryankiros/skip-thoughts/blob/master/decoding/search.py
        
        return k samples (beams) and their NLL scores, each sample is a sequence of labels,
        all samples starts with an `empty` label and end with `eos` or truncated to length of `maxsample`.
        You need to supply `predict` which returns the label probability of each sample.
        `use_unk` allow usage of `oov` (out-of-vocabulary) label in samples
        """
        def sample(energy, n, temperature=temperature):
            """
            Sample at most n elements according to their energy.
            """
            n = min(n,len(energy))
            prb = numpy.exp(-numpy.array(energy) / temperature )
            res = []
            for _ in range(n):
                z = numpy.sum(prb)
                r = numpy.argmax(numpy.random.multinomial(1, prb/z, 1))
                res.append(r)
                prb[r] = 0.0 # make sure we select each element only once
            return res
        #dead_k = 0 # samples that reached eos
        dead_samples = []
        dead_scores = []
        live_k = 1 # samples that did not yet reached eos
        live_samples = [list(start)]
        live_scores = [0]
        while live_k:
            # for every possible live sample calc prob for every possible label 
            probs = predict(live_samples, empty=empty)
            # total score for every sample is sum of -log of word prb
            cand_scores = numpy.array(live_scores)[:,None] - numpy.log(probs)
            cand_scores[:,empty] = 1e20
            if not use_unk:
                for i in range(self.nb_unknown_words):
                    cand_scores[:,self.vocab_size-1-i] = 1e20
            live_scores = list(cand_scores.flatten())
            # find the best (lowest) scores we have from all possible dead samples and
            # all live samples and all possible new words added
            scores = dead_scores + live_scores
            ranks = sample(scores, k)
            n = len(dead_scores)
            ranks_dead = [r for r in ranks if r < n]
            ranks_live = [r - n for r in ranks if r >= n]
            dead_scores = [dead_scores[r] for r in ranks_dead]
            dead_samples = [dead_samples[r] for r in ranks_dead]
            live_scores = [live_scores[r] for r in ranks_live]
            # append the new words to their appropriate live sample
            voc_size = probs.shape[1]
            live_samples = [live_samples[r//voc_size] + [r%voc_size] for r in ranks_live]
            # live samples that should be dead
            # even if len(live_samples) == maxsample we dont want it dead because
            # we want one last prediction out of it to reach a headline of maxlenh
            zombie = [s[-1] == eos or len(s) > maxsample for s in live_samples]
            # add zombies to the dead
            dead_samples += [s for s,z in zip(live_samples, zombie) if z]
            dead_scores += [s for s,z in zip(live_scores, zombie) if z]
            #dead_k = len(dead_samples)
            # remove zombies from the living 
            live_samples = [s for s,z in zip(live_samples, zombie) if not z]
            live_scores = [s for s,z in zip(live_scores, zombie) if not z]
            live_k = len(live_samples)
        return dead_samples + live_samples, dead_scores + live_scores

    def conv_seq_labels(self, xds, xhs, nflips=None, model=None, debug=False):
        """
        Description and headlines are converted to padded input vectors.
        Headlines are one-hot to label.
        """
        BATCH_SIZE = len(xhs)
        assert len(xds) == BATCH_SIZE
        x = [self.vocab_fold(self.lpadd(xd) + xh) for xd, xh in zip(xds, xhs)] # input does not have 2nd eos
        x = keras.preprocessing.sequence.pad_sequences(
            x, maxlen=MAXLEN, value=EMPTY, padding='post', truncating='post')
        x = self.flip_headline(x, nflips=nflips, model=model, debug=debug)
        y = numpy.zeros((BATCH_SIZE, MAXLENH, self.vocab_size))
        for i, xh in enumerate(xhs):
            xh = self.vocab_fold(xh) + [EOS] + [EMPTY] * MAXLENH  # output has eos
            xh = xh[:MAXLENH]
            y[i,:,:] = keras.utils.np_utils.to_categorical(xh, self.vocab_size)
        return x, y
    
    def flip_headline(self, x, nflips=None, model=None, debug=False):
        """
        Given a vectorized input (after `pad_sequences`),
        flip some of the words in the second half (headline),
        with words predicted by the model.
        """
        if nflips is None or model is None or nflips <= 0:
            return x
        BATCH_SIZE = len(x)
        assert numpy.all(x[:,MAXLEND] == EOS)
        probs = model.predict(x, verbose=0, batch_size=BATCH_SIZE)
        x_out = x.copy()
        for b in range(BATCH_SIZE):
            # pick locations we want to flip
            # 0 ... maxlend-1 are descriptions and should be fixed
            # maxlend is eos and should be fixed
            flips = sorted(random.sample(range(MAXLEND + 1, MAXLEN), nflips))
            if debug and b < debug:
                print(b, )
            for input_idx in flips:
                if x[b,input_idx] == EMPTY or x[b,input_idx] == EOS:
                    continue
                # convert from input location to label location
                # the output at maxlend (when input is eos) is fed as input at maxlend + 1
                label_idx = input_idx - (MAXLEND + 1)
                prob = probs[b,label_idx]
                w = prob.argmax()
                if w == EMPTY: # replace accidental empty with oov
                    w = self.oov0
                if debug and b < debug:
                    print('%s => %s' % (self.idx2word[x_out[b,input_idx]], self.idx2word[w]), )
                x_out[b,input_idx] = w
            if debug and b < debug:
                print()
        return x_out

    def gen(
        self,
        Xd,
        Xh,
        batch_size=BATCH_SIZE,
        nb_batches=None,
        nflips=None,
        model=None,
        debug=False,
        seed=SEED):
        """
        Yield batch. For training use nb_batches=None.
        For validation generate deterministic results repeating every nb_batches.
        While training it is good idea to flip once in a while the values of the headlines
        From the value taken from Xh to value generated by the model.
        """
        c = nb_batches if nb_batches else 0
        while True:
            xds = []
            xhs = []
            if nb_batches and c >= nb_batches:
                c = 0
            new_seed = random.randint(0, sys.maxsize)
            random.seed(c + 123456789 + seed)
            for _ in range(BATCH_SIZE):
                t = random.randint(0, len(Xd) - 1)
                xd = Xd[t]
                s = random.randint(min(MAXLEND, len(xd)), max(MAXLEND, len(xd)))
                xds.append(xd[:s])
                xh = Xh[t]
                s = random.randint(min(MAXLENH, len(xh)), max(MAXLENH, len(xh)))
                xhs.append(xh[:s])
            # undo the seeding before we yield inorder not to affect the caller
            c += 1
            random.seed(new_seed)
            yield self.conv_seq_labels(xds, xhs, nflips=nflips, model=model, debug=debug)

    def gensamples(
        self,
        skips=2,
        k=10,
        batch_size=BATCH_SIZE,
        short=True,
        temperature=1.0,
        use_unk=True):
        """
        DOCSTRING
        """
        i = random.randint(0, len(self.X_test)-1)
        print('HEAD:', ' '.join(self.idx2word[w] for w in self.Y_test[i][:MAXLENH]))
        print('DESC:', ' '.join(self.idx2word[w] for w in self.X_test[i][:MAXLEND]))
        sys.stdout.flush()
        print('HEADS:')
        x = self.X_test[i]
        samples = []
        if MAXLEND == 0:
            skips = [0]
        else:
            skips = range(
                min(MAXLEND, len(x)), max(MAXLEND, len(x)),
                abs(MAXLEND-len(x)) // skips + 1)
        for s in skips:
            start = self.lpadd(x[:s])
            fold_start = self.vocab_fold(start)
            sample, score = self.beamsearch(
                predict=self.keras_rnn_predict, start=fold_start, k=k,
                temperature=temperature, use_unk=use_unk)
            assert all(s[MAXLEND] == EOS for s in sample)
            samples += [(s, start, scr) for s, scr in zip(sample, score)]
        samples.sort(key=lambda x:x[-1])
        codes = []
        for sample, start, score in samples:
            code = ''
            words = []
            sample = self.vocab_unfold(start, sample)[len(start):]
            for w in sample:
                if w == EOS:
                    break
                words.append(self.idx2word[w])
                code += chr(w//(256*256)) + chr((w//256)%256) + chr(w%256)
            if short:
                distance = min([100] + [-Levenshtein.jaro(code, c) for c in codes])
                if distance > -0.6:
                    print(score, ' '.join(words))
            else:
                print(score, ' '.join(words))
            codes.append(code)

    def inspect_model(self, model):
        """
        DOCSTRING
        """
        for i, l in enumerate(model.layers):
            print(i, 'cls=%s name=%s'%(type(l).__name__, l.name))
            weights = l.get_weights()
            for weight in weights:
                print(self.str_shape(weight), )
            print()
    
    def keras_rnn_predict(self, samples, empty=EMPTY, model=MODEL, maxlen=MAXLEN):
        """
        For every sample, calculate probability for every possible label.
        You need to supply your RNN model and maxlen (the length of sequences it can handle).
        """
        sample_lengths = map(len, samples)
        assert all(l > MAXLEND for l in sample_lengths)
        assert all(l[MAXLEND] == EOS for l in samples)
        # pad from right (post) so the first maxlend will be description followed by headline
        data = keras.preprocessing.sequence.pad_sequences(
            samples, maxlen=MAXLEN, value=empty, padding='post', truncating='post')
        probs = model.predict(data, verbose=0, batch_size=BATCH_SIZE)
        return numpy.array(
            [prob[sample_length-MAXLEND-1] for prob, sample_length in zip(probs, sample_lengths)])

    def lpadd(self, x, maxlend=MAXLEND, eos=EOS):
        """
        left (pre) pad a description to maxlend and then add eos.
        The eos is the input to predicting the first word in the headline.
        """
        assert maxlend >= 0
        if maxlend == 0:
            return [eos]
        n = len(x)
        if n > maxlend:
            x = x[-maxlend:]
            n = maxlend
        return [EMPTY] * (maxlend-n) + x + [eos]

    def prt(self, label, x):
        """
        DOCSTRING
        """
        print(label + ':', )
        for w in x:
            print(self.idx2word[w], )
        print()

    def simple_context(
        self,
        X,
        mask,
        n=ACTIVATION_RNN_SIZE,
        maxlend=MAXLEND,
        maxlenh=MAXLENH):
        """ 
        DOCSTRING
        """
        desc, head = X[:,:maxlend,:],X[:,maxlend:,:]
        head_activations, head_words = head[:,:,:n], head[:,:,n:]
        desc_activations, desc_words = desc[:,:,:n], desc[:,:,n:]
        # activation for every head word and every desc word
        activation_energies = keras.backend.batch_dot(
            head_activations, desc_activations, axes=(2,2))
        # NOTE: do not use description words that are masked out
        activation_energies = activation_energies + -1e20*keras.backend.expand_dims(
            1.0-keras.backend.cast(mask[:,:maxlend], 'float32'), 1)
        # for every head word compute weights for every desc word
        activation_energies = keras.backend.reshape(activation_energies, (-1, maxlend))
        activation_weights = keras.backend.softmax(activation_energies)
        activation_weights = keras.backend.reshape(activation_weights, (-1, MAXLENH, maxlend))
        # for every head word compute weighted average of desc words
        desc_avg_word = keras.backend.batch_dot(activation_weights, desc_words, axes=(2,1))
        return keras.backend.concatenate((desc_avg_word, head_words))
    
    def str_shape(self, x):
        """
        DOCSTRING
        """
        return 'x'.join(map(str, x.shape))

    def test_gen(self, gen, n=5):
        """
        DOCSTRING
        """
        Xtr, Ytr = next(gen)
        for i in range(n):
            assert Xtr[i,MAXLEND] == EOS
            x = Xtr[i,:MAXLEND]
            y = Xtr[i,MAXLEND:]
            yy = Ytr[i,:]
            yy = numpy.where(yy)[1]
            self.prt('L', yy)
            self.prt('H', y)
            if MAXLEND:
                self.prt('D', x)

    def vocab_fold(self, xs):
        """
        Convert list of word indexes that may contain words outside vocab_size to words inside.
        If a word is outside, try first to use glove_idx2idx to find a similar word inside.
        If none exist then replace all accurancies of the same unknown word with <0>, <1>, ...
        """
        xs = [x if x < self.oov0 else self.glove_idx2idx.get(x,x) for x in xs]
        # the more popular word is <0> and so on
        outside = sorted([x for x in xs if x >= self.oov0])
        # if there are more than nb_unknown_words oov words then put them all in nb_unknown_words-1
        outside = dict((x, self.vocab_size-1-min(i, self.nb_unknown_words-1)) for i,x in enumerate(outside))
        xs = [outside.get(x, x) for x in xs]
        return xs

    def vocab_unfold(self, desc, xs):
        """
        DOCSTRING
        """
        # assume desc is the unfolded version of the start of xs
        unfold = {}
        for i, unfold_idx in enumerate(desc):
            fold_idx = xs[i]
            if fold_idx >= self.oov0:
                unfold[fold_idx] = unfold_idx
        return [unfold.get(x,x) for x in xs]

class VocabularyEmbedding:
    """
    DOCSTRING
    """
    def __init__(self):
        """
        DOCSTRING
        """
        self.file_name = 'vocabulary-embedding'
        self.seed = 42
        self.vocab_size = 40000
        self.embedding_dim = 100
        self.lower = False # dont lower case the text
        file_name_0 = 'tokens' # this is the name of the data file
        with open('data/%s.pkl' % file_name_0, 'rb') as fp:
            self.heads, self.desc, keywords = pickle.load(fp)
        if self.lower:
            self.heads = [h.lower() for h in self.heads]
        if self.lower:
            self.desc = [h.lower() for h in self.desc]
        i=0
        self.heads[i]
        self.desc[i]
        keywords[i]
        len(self.heads), len(set(self.heads))
        len(self.desc), len(set(self.desc))

    def __call__(self):
        """
        DOCSTRING
        """
        vocab, vocabcount = self.get_vocab(self.heads + self.desc)
        print('...', len(vocab))
        pyplot.plot([vocabcount[w] for w in vocab])
        pyplot.gca().set_xscale("log", nonposx='clip')
        pyplot.gca().set_yscale("log", nonposy='clip')
        pyplot.title('word distribution in headlines and discription')
        pyplot.xlabel('rank')
        pyplot.ylabel('total appearances')
        self.empty = 0 # RNN mask of no data
        self.eos = 1 # end of sentence
        self.start_idx = self.eos + 1 # first real word
        word2idx, self.idx2word = self.get_idx(vocab, vocabcount)
        fname = 'glove.6B.%dd.txt' % self.embedding_dim
        datadir_base = os.path.expanduser(os.path.join('~', '.keras'))
        if not os.access(datadir_base, os.W_OK):
            datadir_base = os.path.join('/tmp', '.keras')
        datadir = os.path.join(datadir_base, 'datasets')
        glove_name = os.path.join(datadir, fname)
        if not os.path.exists(glove_name):
            path = keras.utils.data_utils.get_file(
                'glove.6B.zip', origin='http://nlp.stanford.edu/data/glove.6B.zip')
            with zipfile.ZipFile(os.path.join(datadir, path), 'r') as zip_ref:
                zip_ref.extractall(datadir)
        glove_n_symbols = sum(1 for line in open(glove_name))
        glove_n_symbols = int(glove_n_symbols[0].split()[0])
        glove_index_dict = {}
        glove_embedding_weights = numpy.empty((glove_n_symbols, self.embedding_dim))
        globale_scale=0.1
        with open(glove_name, 'r') as fp:
            i = 0
            for l in fp:
                l = l.strip().split()
                w = l[0]
                glove_index_dict[w] = i
                glove_embedding_weights[i,:] = map(float,l[1:])
                i += 1
        glove_embedding_weights *= globale_scale
        glove_embedding_weights.std()
        for w,i in glove_index_dict.items():
            w = w.lower()
            if w not in glove_index_dict:
                glove_index_dict[w] = i
        # generate random embedding with same scale as glove
        numpy.random.seed(self.seed)
        shape = (self.vocab_size, self.embedding_dim)
        scale = glove_embedding_weights.std()*numpy.sqrt(12)/2 # uniform and not normal
        embedding = numpy.random.uniform(low=-scale, high=scale, size=shape)
        print('random-embedding/glove scale', scale, 'std', embedding.std())
        # copy from glove weights of words that appear in our short vocabulary (idx2word)
        c = 0
        for i in range(self.vocab_size):
            w = self.idx2word[i]
            g = glove_index_dict.get(w, glove_index_dict.get(w.lower()))
            if g is None and w.startswith('#'): # glove has no hastags
                w = w[1:]
                g = glove_index_dict.get(w, glove_index_dict.get(w.lower()))
            if g is not None:
                embedding[i,:] = glove_embedding_weights[g,:]
                c+=1
        print('number of tokens, in small vocab, found in glove and copied to embedding',
              c, c/float(self.vocab_size))
        glove_thr = 0.5
        word2glove = {}
        for w in word2idx:
            if w in glove_index_dict:
                g = w
            elif w.lower() in glove_index_dict:
                g = w.lower()
            elif w.startswith('#') and w[1:] in glove_index_dict:
                g = w[1:]
            elif w.startswith('#') and w[1:].lower() in glove_index_dict:
                g = w[1:].lower()
            else:
                continue
            word2glove[w] = g
        normed_embedding = embedding/numpy.array(
            [numpy.sqrt(numpy.dot(gweight,gweight)) for gweight in embedding])[:,None]
        nb_unknown_words = 100
        glove_match = []
        for w,idx in word2idx.items():
            if idx >= self.vocab_size - nb_unknown_words and w.isalpha() and w in word2glove:
                gidx = glove_index_dict[word2glove[w]]
                gweight = glove_embedding_weights[gidx,:].copy()
                # find row in embedding that has the highest cos score with gweight
                gweight /= numpy.sqrt(numpy.dot(gweight,gweight))
                score = numpy.dot(normed_embedding[:self.vocab_size - nb_unknown_words], gweight)
                while True:
                    embedding_idx = score.argmax()
                    s = score[embedding_idx]
                    if s < glove_thr:
                        break
                    if self.idx2word[embedding_idx] in word2glove:
                        glove_match.append((w, embedding_idx, s)) 
                        break
                    score[embedding_idx] = -1
        glove_match.sort(key = lambda x: -x[2])
        print('# of glove substitutes found', len(glove_match))
        for orig, sub, score in glove_match[-10:]:
            print(score, orig,'=>', self.idx2word[sub])
        glove_idx2idx = dict((word2idx[w],embedding_idx) for  w, embedding_idx, _ in glove_match)
        Y = [[word2idx[token] for token in headline.split()] for headline in self.heads]
        len(Y)
        pyplot.hist(map(len, Y), bins=50)
        X = [[word2idx[token] for token in d.split()] for d in self.desc]
        len(X)
        pyplot.hist(map(len, X), bins=50)
        with open('data/%s.pkl' % self.file_name,'wb') as fp:
            pickle.dump((embedding, self.idx2word, word2idx, glove_idx2idx), fp, -1)
        with open('data/%s.data.pkl' % self.file_name,'wb') as fp:
            pickle.dump((X,Y),fp,-1)
   
    def get_idx(self, vocab, vocabcount):
        """
        DOCSTRING
        """
        word2idx = dict((word, idx + self.start_idx) for idx, word in enumerate(vocab))
        word2idx['<empty>'] = self.empty
        word2idx['<eos>'] = self.eos
        self.idx2word = dict((idx, word) for word, idx in word2idx.items())
        return word2idx, self.idx2word

    def get_vocab(self, lst):
        """
        DOCSTRING
        """
        vocabcount = collections.Counter(w for txt in lst for w in txt.split())
        vocab = map(lambda x:x[0], sorted(vocabcount.items(), key=lambda x:-x[1]))
        return vocab, vocabcount
