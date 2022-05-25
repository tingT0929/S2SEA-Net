import numpy as np
from keras.models import Model
import tensorflow.keras.backend as K
from tensorflow.keras.layers import *

class VarAttn2:
    def __init__(self, cfg):
        self.embed_dim = cfg.model.embed_dim
        self.travel_num = cfg.model.travel_num
        self.enc_steps = cfg.data.enc_steps
        self.t = cfg.model.temporature
        self.dense1 = TimeDistributed(Dense(self.embed_dim, activation='elu'))
        self.dense2 = TimeDistributed(Dense(self.travel_num, activation='tanh')) 
    
    def __call__(self, x, c=None):
        assert x.shape[-2:] == (self.enc_steps, self.travel_num)
        x_context = K.stack([self.dense1(x[Ellipsis,i:i+1]) for i in range(self.travel_num)], axis=2) #30*10*hidden
        context = Reshape((self.enc_steps, -1))(x_context) 
        if c is not None:
            x_context = TimeDistributed(Dense(self.embed_dim))(context) # 30*hidden
            x_context = BatchNormalization()(x_context)
            c_context = BatchNormalization()(c)
            c_context = RepeatVector(self.enc_steps)(c)
            context = Add()([x_context, c_context])
        embedding = self.dense2(context) #30*10
        var_attn = Activation('softmax', name='var_attn')(embedding / self.t)
        context = K.sum(var_attn * x, axis=-1, keepdims=True)
        return context, var_attn

class FactorEncoder:
    def __init__(self, cfg):
        self.embed_dim = cfg.model.embed_dim
        self.state_num = cfg.data.state_num
        #self.county_num = cfg.data.county_num
        self.embedding1 = Embedding(self.state_num, self.embed_dim)
        #self.embedding2 = Embedding(self.county_num, self.embed_dim)
        self.linear1 = Dense(self.embed_dim)
        self.linear2 = Dense(self.embed_dim)
        self.dense = Dense(self.embed_dim, activation='elu')
    
    def __call__(self, x):
        # inputs: state id, lon, lat ~(None, 3)
        assert x.shape[-1] == 3
        state_embed = K.squeeze(self.embedding1(x[Ellipsis,0:1]), axis=-2)
        #county_embed = K.squeeze(self.embedding2(x[Ellipsis,1:2]), axis=-2)
        lon_embed = self.linear1(x[Ellipsis,1:2])
        lat_embed = self.linear2(x[Ellipsis,2:3])
        static_embed = concatenate([state_embed, lon_embed, lat_embed])
        static_embed = self.dense(static_embed)
        return static_embed

class CoviSeq:
    def __init__(self, cfg):
        self.cfg = cfg
        self.encoder = LSTM(
            self.cfg.model.latent_dim, dropout = self.cfg.model.dropout, return_state=True, return_sequences=True
            )
        self.decoder = LSTM(
            self.cfg.model.latent_dim, dropout = self.cfg.model.dropout, return_state=True, return_sequences=True
            )
        #self.decoder_dense1 = Dense(self.cfg.model.hidden_size, 'relu')
        self.decoder_dense2 = Dense(self.cfg.model.decoder_dim, 'relu')
        self.build_network()

    def build_network(self):
        # Encoder input size: [Batch, enc_steps, encoder_dim]
        encoder_inputs = Input(shape=(self.cfg.data.enc_steps, self.cfg.model.encoder_dim))
        encoder_outputs = self.encoder(encoder_inputs)
        encoder_states = encoder_outputs[1:]
        # Decoder input: encoder_states as initial state, and the last predictions are fed in.
        decoder_inputs = Input(shape=(self.cfg.data.pred_steps, self.cfg.model.decoder_dim + self.cfg.data.use_month))
        decoder_outputs = self.decoder(decoder_inputs, initial_state=encoder_states)
        decoder_outputs = decoder_outputs[0]
        decoder_outputs = self.decoder_dense2(decoder_outputs)
        self.train_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        self.encoder_model = Model(encoder_inputs, encoder_states)
        # Inference
        infer_inputs = Input(shape=(1, self.cfg.model.decoder_dim+self.cfg.data.use_month))
        decoder_state_input_h = Input(shape=(self.cfg.model.latent_dim,))
        decoder_state_input_c = Input(shape=(self.cfg.model.latent_dim,))
        decoder_state_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_out = self.decoder(infer_inputs, initial_state=decoder_state_inputs)
        decoder_outputs, decoder_states = decoder_out[0], decoder_out[1:]
        decoder_outputs = self.decoder_dense2(decoder_outputs)
        self.decoder_model = Model([infer_inputs] + [decoder_state_inputs], [decoder_outputs] + decoder_states)

    def predict(self, test_data):
        test_enc, test_dec, _ = test_data
        states_value = self.encoder_model.predict(test_enc)
        decoder_inputs = np.zeros((test_enc.shape[0], 1, self.cfg.model.decoder_dim+self.cfg.data.use_month))
        decoder_inputs[:, 0, :] = test_enc[:, -1, :self.cfg.model.decoder_dim+self.cfg.data.use_month]
        test_output = np.zeros((test_enc.shape[0],self.cfg.data.pred_steps, self.cfg.model.decoder_dim))
        for i in range(self.cfg.data.pred_steps):
            outputs = self.decoder_model.predict([decoder_inputs] + [states_value])
            output, states_value = outputs[0], outputs[1:]
            test_output[: ,i,:] = output.squeeze(1).copy()
            decoder_inputs = np.zeros((test_enc.shape[0], 1, self.cfg.model.decoder_dim+self.cfg.data.use_month))
            if self.cfg.data.use_month:
                decoder_inputs[:,0,0] = test_dec[:,:,0][:,i].copy()
                decoder_inputs[:, 0, 1:] = output[:,0,:].copy()
            else:
                decoder_inputs[:, 0, :] = output[:,0,:].copy()
        return [test_output]

class CoviFacAttn2(CoviSeq):
    def __init__(self, cfg):
        self.var_attn = VarAttn2(cfg)
        #self.factor_h = FactorEncoder(cfg)  #Encoder State_h
        #self.factor_c = FactorEncoder(cfg)  #Encoder state_c
        self.factor_var = FactorEncoder(cfg) # For Variable Attention
        super(CoviFacAttn2, self).__init__(cfg)
    
    def build_network(self):
        # Encoder input size: [Batch, enc_steps, encoder_dim]
        factor_inputs = Input(shape=(3,))
        factor_embed = self.factor_var(factor_inputs)
        #h, c = self.factor_h(factor_inputs), self.factor_c(factor_inputs)
        #h, c = Dense(self.cfg.model.latent_dim)(h), Dense(self.cfg.model.latent_dim)(c)
        encoder_inputs = Input(shape=(self.cfg.data.enc_steps, self.cfg.model.encoder_dim))
        covid = encoder_inputs[:,:,self.cfg.data.use_month:self.cfg.model.decoder_dim+self.cfg.data.use_month] 
        travel = encoder_inputs[:,:,self.cfg.model.decoder_dim+self.cfg.data.use_month:]
        context, attention = self.var_attn(travel, factor_embed)
        context = concatenate([covid, context])
        encoder_outputs = self.encoder(context)#, initial_state=[h, c])
        encoder_states = encoder_outputs[1:]
        # Decoder input: encoder_states as initial state, and the last predictions are fed in.
        decoder_inputs = Input(shape=(self.cfg.data.pred_steps, self.cfg.model.decoder_dim + self.cfg.data.use_month))
        decoder_outputs = self.decoder(decoder_inputs, initial_state=encoder_states)
        decoder_outputs = decoder_outputs[0]
        decoder_outputs = self.decoder_dense2(decoder_outputs)
        self.train_model = Model([encoder_inputs, decoder_inputs, factor_inputs], decoder_outputs)
        self.encoder_model = Model([encoder_inputs, factor_inputs], [encoder_states, attention])
        #inference
        infer_inputs = Input(shape=(1, self.cfg.model.decoder_dim+self.cfg.data.use_month))
        decoder_state_input_h = Input(shape=(self.cfg.model.latent_dim,))
        decoder_state_input_c = Input(shape=(self.cfg.model.latent_dim,))
        decoder_state_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_out = self.decoder(infer_inputs, initial_state=decoder_state_inputs)
        decoder_outputs, decoder_states = decoder_out[0], decoder_out[1:]
        decoder_outputs = self.decoder_dense2(decoder_outputs)
        self.decoder_model = Model([infer_inputs] + [decoder_state_inputs], [decoder_outputs] + decoder_states)

    def predict(self, test_data):
        test_enc, test_dec, test_factor, _ = test_data
        states_value, attn_weights = self.encoder_model.predict([test_enc, test_factor])
        decoder_inputs = np.zeros((test_enc.shape[0], 1, self.cfg.model.decoder_dim+self.cfg.data.use_month))
        decoder_inputs[:, 0, :] = test_enc[:, -1, :self.cfg.model.decoder_dim+self.cfg.data.use_month]
        test_output = np.zeros((test_enc.shape[0],self.cfg.data.pred_steps, self.cfg.model.decoder_dim))
        for i in range(self.cfg.data.pred_steps):
            outputs = self.decoder_model.predict([decoder_inputs]+[states_value])
            output, states_value = outputs[0], outputs[1:]
            test_output[: ,i,:] = output.squeeze(1).copy()
            decoder_inputs = np.zeros((test_enc.shape[0], 1, self.cfg.model.decoder_dim+self.cfg.data.use_month))
            if self.cfg.data.use_month:
                decoder_inputs[:,0,0] = test_dec[:,:,0][:,i].copy()
                decoder_inputs[:, 0, 1:] = output[:,0,:].copy()
            else:
                decoder_inputs[:, 0, :] = output[:,0,:].copy()
        return test_output, attn_weights

class CoviSeq2:
    def __init__(self, cfg):
        self.cfg = cfg
        self.encoder = LSTM(
            self.cfg.model.latent_dim, dropout = self.cfg.model.dropout, return_state=True, return_sequences=True
            )
        self.decoder = LSTM(
            self.cfg.model.latent_dim, dropout = self.cfg.model.dropout, return_state=True, return_sequences=True
            )
        #self.decoder_dense1 = Dense(self.cfg.model.hidden_size, 'relu')
        self.decoder_dense2 = Dense(self.cfg.model.decoder_dim, 'relu')
        self.build_network()

    def build_network(self):
        # Encoder input size: [Batch, enc_steps, encoder_dim]
        encoder_inputs = Input(shape=(self.cfg.data.enc_steps, 2))
        encoder_outputs = self.encoder(encoder_inputs)
        encoder_states = encoder_outputs[1:]
        # Decoder input: encoder_states as initial state, and the last predictions are fed in.
        decoder_inputs = Input(shape=(self.cfg.data.pred_steps, self.cfg.model.decoder_dim + self.cfg.data.use_month))
        decoder_outputs = self.decoder(decoder_inputs, initial_state=encoder_states)
        decoder_outputs = decoder_outputs[0]
        decoder_outputs = self.decoder_dense2(decoder_outputs)
        self.train_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        self.encoder_model = Model(encoder_inputs, encoder_states)
        # Inference
        infer_inputs = Input(shape=(1, self.cfg.model.decoder_dim+self.cfg.data.use_month))
        decoder_state_input_h = Input(shape=(self.cfg.model.latent_dim,))
        decoder_state_input_c = Input(shape=(self.cfg.model.latent_dim,))
        decoder_state_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_out = self.decoder(infer_inputs, initial_state=decoder_state_inputs)
        decoder_outputs, decoder_states = decoder_out[0], decoder_out[1:]
        decoder_outputs = self.decoder_dense2(decoder_outputs)
        self.decoder_model = Model([infer_inputs] + [decoder_state_inputs], [decoder_outputs] + decoder_states)

    def predict(self, test_data):
        test_enc, test_dec, _ = test_data
        states_value = self.encoder_model.predict(test_enc)
        decoder_inputs = np.zeros((test_enc.shape[0], 1, self.cfg.model.decoder_dim+self.cfg.data.use_month))
        decoder_inputs[:, 0, :] = test_enc[:, -1, :self.cfg.model.decoder_dim+self.cfg.data.use_month]
        test_output = np.zeros((test_enc.shape[0],self.cfg.data.pred_steps, self.cfg.model.decoder_dim))
        for i in range(self.cfg.data.pred_steps):
            outputs = self.decoder_model.predict([decoder_inputs] + [states_value])
            output, states_value = outputs[0], outputs[1:]
            test_output[: ,i,:] = output.squeeze(1).copy()
            decoder_inputs = np.zeros((test_enc.shape[0], 1, self.cfg.model.decoder_dim+self.cfg.data.use_month))
            if self.cfg.data.use_month:
                decoder_inputs[:,0,0] = test_dec[:,:,0][:,i].copy()
                decoder_inputs[:, 0, 1:] = output[:,0,:].copy()
            else:
                decoder_inputs[:, 0, :] = output[:,0,:].copy()
        return [test_output]