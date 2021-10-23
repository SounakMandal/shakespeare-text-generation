from utils import *

filename = input('Enter file name : ')
stateful = False
with open(f"datasets\\{filename}.txt") as file:
    text = file.read()
print("Total character count : ", len(text))
dataset, batch_size, word_count = make_dataset(text)

mode = input('\nEnter mode : ')
if mode == "train":
    base_model = int(time.time())
    log_dir = logging_path(filename, base_model)  # Logging path
    save_filepath = saving_path(
        filename, mode, base_model, base_model
    )  # Saving path

elif mode == "retrain":
    model_name = int(time.time())
    base_model = find_base(filename)
    log_dir = logging_path(filename, base_model)  # Logging path
    load_filepath = loading_path(filename, base_model)  # Loading path
    save_filepath = saving_path(
        filename, mode, base_model, model_name
    )  # Saving path

else:
    base_model = find_base(filename)
    load_filepath = loading_path(filename, base_model)  # Loading path

callbacks = [
    keras.callbacks.ModelCheckpoint(
        save_filepath, monitor='loss',
        verbose=1, save_best_only=False
    ),
    keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1,
        update_freq=10, write_graph=True
    ),
]

epochs = int(input('\nEnter the number of epochs : '))
model = keras.models.Sequential([
    keras.layers.InputLayer(
        input_shape=(None, len(word_count))
    ),
    keras.layers.GRU(
        1024, return_sequences=True,
        dropout=0.2, recurrent_dropout=0.2
    ),
    keras.layers.GRU(
        512, return_sequences=True,
        dropout=0.2, recurrent_dropout=0.2
    ),
    keras.layers.GRU(
        256, return_sequences=True,
        dropout=0.2, recurrent_dropout=0.2
    ),
    keras.layers.GRU(
        128, return_sequences=True,
        dropout=0.1, recurrent_dropout=0.1
    ),
    keras.layers.GRU(
        64, return_sequences=True,
        dropout=0.1, recurrent_dropout=0.1
    ),
    keras.layers.GRU(
        32, return_sequences=True,
        dropout=0.1, recurrent_dropout=0.1
    ),
    keras.layers.TimeDistributed(
        keras.layers.Dense(len(word_count), activation='softmax')
    )
]) if mode == 'train' else keras.models.load_model(load_filepath, compile=False)

model.summary()

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

history = model.fit(
    dataset, epochs=epochs,
    callbacks=callbacks
)
