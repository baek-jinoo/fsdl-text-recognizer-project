from boltons.cacheutils import cachedproperty
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.python.client import device_lib
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D, Permute, RepeatVector, Reshape, TimeDistributed, Lambda, LSTM, GRU, CuDNNLSTM, Bidirectional, Add
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model as KerasModel
    
#from tensorflow.keras.layers.recurrent import LSTM
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import BatchNormalization

from text_recognizer.models.line_model import LineModel
from text_recognizer.networks.lenet import lenet
from text_recognizer.networks.misc import slide_window
from text_recognizer.networks.ctc import ctc_decode


def line_lstm_ctc(input_shape, output_shape, window_width=28, window_stride=14, conv_dim=128, lstm_dim=256, num_lstm_layers=3, dropout=0.3):
    image_height, image_width = input_shape
    output_length, num_classes = output_shape

    num_windows = int((image_width - window_width) / window_stride) + 1
    if num_windows < output_length:
        raise ValueError(f'Window width/stride need to generate at least {output_length} windows (currently {num_windows})')

    image_input = Input(shape=input_shape, name='image')
    y_true = Input(shape=(output_length,), name='y_true')
    input_length = Input(shape=(1,), name='input_length')
    label_length = Input(shape=(1,), name='label_length')

    gpu_present = len(device_lib.list_local_devices()) > 1
    lstm_fn = CuDNNLSTM if gpu_present else LSTM

    # Your code should use slide_window and extract image patches from image_input.
    # Pass a convolutional model over each image patch to generate a feature vector per window.
    # Pass these features through one or more LSTM layers.
    # Convert the lstm outputs to softmax outputs.
    # Note that lstms expect a input of shape (num_batch_size, num_timesteps, feature_length).

    ##### Your code below (Lab 3)
    image_reshaped = Reshape((image_height, image_width, 1))(image_input)
    # (image_height, image_width, 1)

    image_patches = Lambda(
        slide_window,
        arguments={'window_width': window_width, 'window_stride': window_stride}
    )(image_reshaped)
    # (num_windows, image_height, window_width, 1)

    conv = Conv2D(conv_dim, (image_height, window_width), (1, window_stride), activation='relu')(image_reshaped)
    conv = BatchNormalization()(conv)
    conv = Dropout(dropout)(conv)

    conv_squeezed = Lambda(lambda x: K.squeeze(x, 1))(conv)

#model.add(Bidirectional(LSTM(128, activation=None), input_shape=(256,10)))
#model.add(BatchNormalization())
    lstm_output1 = Bidirectional(lstm_fn(lstm_dim, return_sequences=True), input_shape=(num_windows, conv_dim))(conv_squeezed)
    print("dropout", dropout)
    #lstm_output1 = lstm_fn(lstm_dim, return_sequences=True)(conv_squeezed)
    # (num_windows, 128)
    lstm_output2 = lstm_fn(lstm_dim, return_sequences=True)(lstm_output1)
    lstm_output2 = BatchNormalization()(lstm_output2)
    lstm_output2 = Dropout(dropout)(lstm_output2)
    lstm_input3 = lstm_output2
    
    lstm_output3 = lstm_fn(lstm_dim, return_sequences=True)(lstm_input3)
    lstm_output3 = BatchNormalization()(lstm_output3)
    lstm_output3 = Dropout(dropout)(lstm_output3)
    
    lstm_input4 = Add()([lstm_output3, lstm_input3])
    lstm_output4 = lstm_fn(lstm_dim, return_sequences=True)(lstm_input4)
    lstm_output4 = BatchNormalization()(lstm_output4)
    lstm_output4 = Dropout(dropout)(lstm_output4)
    
    #lstm_input5 = Add()([lstm_output4, lstm_input4])
    #lstm_output5 = lstm_fn(lstm_dim, return_sequences=True)(lstm_input5)
    #lstm_output5 = BatchNormalization()(lstm_output5)
    #lstm_output5 = Dropout(dropout)(lstm_output5)

    #lstm_input6 = Add()([lstm_output5, lstm_input5])
    #lstm_output6 = lstm_fn(lstm_dim, return_sequences=True)(lstm_input6)
    #lstm_output6 = BatchNormalization()(lstm_output6)
    #lstm_output6 = Dropout(dropout)(lstm_output6)
    
    softmax_output = Dense(num_classes, activation='softmax', name='softmax_output')(lstm_output4)
    # (num_windows, num_classes)
    ##### Your code above (Lab 3)

    input_length_processed = Lambda(
        lambda x, num_windows=None: x * num_windows,
        arguments={'num_windows': num_windows}
    )(input_length)

    ctc_loss_output = Lambda(
        lambda x: K.ctc_batch_cost(x[0], x[1], x[2], x[3]),
        name='ctc_loss'
    )([y_true, softmax_output, input_length_processed, label_length])

    ctc_decoded_output = Lambda(
        lambda x: ctc_decode(x[0], x[1], output_length),
        name='ctc_decoded'
    )([softmax_output, input_length_processed])

    model = KerasModel(
        inputs=[image_input, y_true, input_length, label_length],
        outputs=[ctc_loss_output, ctc_decoded_output]
    )
    return model

