import keras
from keras.layers import *
import numpy as np
import keras.backend as K
from keras.engine import *
from keras import *
from Capsule import Capsule
from keras.applications import *
from keras_multi_head import MultiHeadAttention # pip install keras_multi_head
from keras.utils.vis_utils import plot_model

EMBEDDING_DIM = 300
POSITION_EMBEDDING_DIM = 50
MAX_LEN = 36
MAX_LEN1 = 8

def reduce_dimension(x, length, mask):
    res = K.reshape(x, [-1, length])  # n*30   x=n*30*1
    res = K.softmax(res)
    res = res * K.cast(mask, dtype='float32')  # n*30
    temp = K.sum(res, axis=1, keepdims=True)  # n*1
    temp = K.repeat_elements(temp, rep=length, axis=1)  #n*30
    return res / temp

def reduce_dimension_output_shape(input_shape):
    shape = list(input_shape)
    assert len(shape) == 3  # only valid for 3D tensors
    return [shape[0], shape[1]]

def attention(x, dim):
    res = K.batch_dot(x[0], x[1], axes=[1, 1])
    #x = [[1, 2], [3, 4]] 和 y = [[5, 6], [7, 8]]， batch_dot(x, y, axes=1) = [[17], [53]]
    return K.reshape(res, [-1, dim])
def attention_output_shape(input_shape):
    shape = list(input_shape[1])
    assert len(shape) == 3
    return [shape[0], shape[2]]

def no_change(input_shape):
    return input_shape

def liter(x, length):
    res = K.repeat(x, length)  # (?, 82, 300)
    return res
def liter_output_shape(input_shape):
    shape = list(input_shape)
    return [shape[0], MAX_LEN, shape[1]]

def liter1(x, length):
    res = K.repeat(x, length)  # (?, 82, 300)
    return res
def liter_output_shape1(input_shape):
    shape = list(input_shape)
    return [shape[0], MAX_LEN1, shape[1]]

def build_model(max_len,aspect_max_len,embedding_matrix=[],position_embedding_matrix=[],class_num=3,num_words=12877):
    # image / input and embedding
    image_input = Input(shape=(224, 224, 3),name='image_input')
    base_model = VGG19(include_top=False, weights='imagenet', input_tensor=image_input)
    image_embedding = base_model.output  #(?, 7, 7, 2048)
    image_embedding = Flatten(name="flatten")(image_embedding)
    image_embedding = Dense(36*128, activation="relu")(image_embedding)
    image_embedding = Dropout(0.3)(image_embedding)
    image_embedding = Reshape((36,128))(image_embedding)  # (?, 36, 128)
    image_avg = GlobalAveragePooling1D()(image_embedding)  # (?, 128)

    for layer in base_model.layers[:30]:
        layer.trainable = False
    for layer in base_model.layers[30:]:
        layer.trainable = True

    # sentence / input and embedding
    sentence_input = Input(shape=(max_len,), dtype='int32', name='sentence_input')  # n*36
    sentence_embedding_layer = Embedding(num_words + 1, EMBEDDING_DIM, weights=[embedding_matrix],input_length=max_len,
                                         trainable=False, mask_zero=False)
    sentence_embedding = sentence_embedding_layer(sentence_input)  # n*36*300

    # position / input embedding
    position_input = Input(shape=(max_len,), dtype='int32', name='position_input')  # n*36
    position_embedding = Embedding(max_len * 2, POSITION_EMBEDDING_DIM, weights=[position_embedding_matrix],
                                   input_length=max_len, trainable=True, mask_zero=False)(position_input)  # n*36*50

    # aspect / input and  embedding
    aspect_input = Input(shape=(aspect_max_len,), dtype='int32', name='aspect_input')  # n*8
    aspect_embedding_layer = Embedding(num_words + 1, EMBEDDING_DIM, weights=[embedding_matrix],input_length=aspect_max_len,
                                       trainable=False, mask_zero=False)
    aspect_embedding = aspect_embedding_layer(aspect_input)  # n*8*300
    aspect_embedding = Bidirectional(GRU(128, activation="relu", return_sequences=True,
                                         recurrent_dropout=0.5, dropout=0.5))(aspect_embedding) # n*8*256

    ## mid=aspect pool
    aspect_attention = MultiHeadAttention(head_num=4)(aspect_embedding) # n*8*128
    aspect_attention = Dropout(0.3)(aspect_attention)

    aspect_avg = GlobalAveragePooling1D()(aspect_attention) # n*128
    aspect_liter = Lambda(liter,output_shape=liter_output_shape,arguments={'length': max_len})(aspect_avg)  # n*36*128

    ## left: con=embd+position+aspect, avg; senlf attention, capsule
    sentence_con = keras.layers.concatenate([sentence_embedding,position_embedding]) # ?*36*350
    sentence_attention = MultiHeadAttention(head_num=7)(sentence_con)  #(?, ?, 384)
    sentence_attention = Dropout(0.3)(sentence_attention)
    sentence_avg = GlobalAveragePooling1D()(sentence_attention) # (?, 384)

    sentence_merge = keras.layers.concatenate([sentence_attention, aspect_liter])  # (?, 36, 606)
    sentence_capsule = Capsule(num_capsule=8, dim_capsule=300, routings=3, share_weights=True)(sentence_merge) #(?, 8, 64)
    sentence_capsule = Dropout(0.3)(sentence_capsule)

    ## right: con=iamge+aspect,capsule
    image_con = keras.layers.concatenate([image_embedding, position_embedding,aspect_liter]) #(?, 36, 512)
    image_capsule = Capsule(num_capsule=8, dim_capsule=300, routings=3, share_weights=True)(image_con) #(?, 8, 150)
    image_capsule = Dropout(0.3)(image_capsule)

    ### final=mix attention
    final = MultiHeadAttention(head_num=8)([sentence_capsule, image_capsule, aspect_attention]) #(?, ?, 192)
    final = Dropout(0.3)(final)
    final_avg = GlobalAveragePooling1D()(final)

    final_con = keras.layers.concatenate([final_avg,sentence_avg,aspect_avg,image_avg])
    final_dense = Dense(256, activation='sigmoid')(final_con)   # (?, 9 256)
    final_dense = Dropout(0.3)(final_dense)
    predictions = Dense(class_num, activation='softmax')(final_dense)  # n*3

    model = Model(inputs=[image_input, sentence_input, position_input, aspect_input], outputs=predictions)

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy']) #optimizer='rmsprop' optimizers.SGD(lr=1e-4, momentum=0.9)
    print(model.summary())
    plot_model(model, show_shapes=True, to_file='multichannel.png')
    return model
