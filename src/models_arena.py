import tensorflow as tf
import torch
import torch.nn as nn
from sklearn.neural_network import MLPRegressor, MLPClassifier

def train_sklearn(X_tr, y_tr, X_te, task='reg'):
    model = MLPRegressor(hidden_layer_sizes=(128,64), max_iter=20) if task=='reg' else MLPClassifier(hidden_layer_sizes=(128,64), max_iter=20)
    model.fit(X_tr, y_tr)
    return model.predict(X_te)

def train_tf(X_tr, y_tr, X_te, task='reg', num_classes=3):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_tr.shape[1],)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1 if task=='reg' else num_classes, activation=None if task=='reg' else 'softmax')
    ])
    model.compile(optimizer='adam', loss='mse' if task=='reg' else 'sparse_categorical_crossentropy')
    model.fit(X_tr, y_tr, epochs=15, verbose=0)
    preds = model.predict(X_te, verbose=0)
    return preds.flatten() if task=='reg' else preds.argmax(axis=1)

def train_pytorch(X_tr, y_tr, X_te, task='reg', num_classes=3):
    X_t, y_t = torch.FloatTensor(X_tr), torch.FloatTensor(y_tr) if task=='reg' else torch.LongTensor(y_tr)
    model = nn.Sequential(nn.Linear(X_tr.shape[1], 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1 if task=='reg' else num_classes))
    opt, crit = torch.optim.Adam(model.parameters(), lr=0.01), nn.MSELoss() if task=='reg' else nn.CrossEntropyLoss()
    for _ in range(15):
        opt.zero_grad(); loss = crit(model(X_t).squeeze(), y_t); loss.backward(); opt.step()
    with torch.no_grad():
        preds = model(torch.FloatTensor(X_te))
        return preds.numpy().flatten() if task=='reg' else preds.argmax(dim=1).numpy()