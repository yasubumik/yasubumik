import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt

df = pd.read_csv('test.csv' )#,parse_dates=True)
df.columns = ['times','x']
df.set_index('times', inplace = True)
fig, ax = plt.subplots()
df.plot(legend=False, ax=ax)
plt.show()

x = df[0:1500]
x_ano = df[1500:3000]

x.plot(legend=False)
plt.ylim([-40000, 45000])

x_ano.plot(legend=False)
plt.ylim([-40000, 45000])

#標準化
x_norm = (x - x.mean())/x.std()
x_ano_norm = (x_ano - x.mean())/x.std()

t_steps = 50
def create_sequences(df):
    x = []
    for i in range(0, len(df) - t_steps + 1):
        x.append(df[i:i + t_steps].to_numpy())
    x_out = np.array(x)
    return x_out

x_train = create_sequences(x_norm)#_normalized)
x_train.shape

model =keras.initializers.Initializer()
model = keras.Sequential(
    [
        layers.Input(shape=(x_train.shape[1], x_train.shape[2])),
        layers.Conv1D(
            filters=32, kernel_size=7, padding="same", strides=1, activation="relu"
        ),
        layers.Dropout(rate=0.2),
        layers.Conv1D(
            filters=16, kernel_size=7, padding="same", strides=1, activation="relu"
        ),
        layers.Conv1DTranspose(
            filters=16, kernel_size=7, padding="same", strides=1, activation="relu"
        ),
        layers.Dropout(rate=0.2),
        layers.Conv1DTranspose(
            filters=32, kernel_size=7, padding="same", strides=1, activation="relu"
        ),
        layers.Conv1DTranspose(filters=1, kernel_size=7, padding="same"),
    ]
)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
model.summary()

history = model.fit(
    x_train,
    x_train,
    epochs=200,
    batch_size=100,
    validation_split=0.1,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min")
    ],
)

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()
plt.show()

x_train_pred = model.predict(x_train)
train_mae_loss = np.mean(np.abs(x_train_pred - x_train), axis=1)
threshold = np.max(train_mae_loss)

x_test = create_sequences(x_ano_norm)
x_test_pred = model.predict(x_test)
test_mae_loss = np.mean(np.abs(x_test_pred - x_test), axis=1)
test_mae_loss = test_mae_loss.reshape((-1))
anomalies = test_mae_loss > threshold

anomalous_data_indices = []
for data_idx in range(t_steps - 1, len(x_ano_norm) - t_steps + 1):
    if np.all(anomalies[data_idx - t_steps + 1 : data_idx]):
        anomalous_data_indices.append(data_idx)
        
df_subset = x_ano.iloc[anomalous_data_indices]
fig, ax = plt.subplots()
df.plot(legend=False, ax=ax)
df_subset.plot(legend=False, ax=ax, color="r")
plt.show()