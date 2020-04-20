import pandas as pd
from keras.engine.saving import model_from_json
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv("az_1_100.csv")
print(data.shape)

data.rename(columns={'0': 'label'}, inplace=True)
X = data.drop('label', axis=1)
y = data['label']
(X_train, X_test, Y_train, Y_test) = train_test_split(X, y)

standard_scaler = MinMaxScaler()
standard_scaler.fit(X_train)

X_train = standard_scaler.transform(X_train)
X_test = standard_scaler.transform(X_test)

X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype('float32')
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype('float32')

print(X_train.shape) # (1, 249542, 784, 1)

Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)

print(Y_test.shape) # (122909, 26)


model = model_from_json(open('model.json', 'r').read())
model.summary()
model.load_weights('weights.model')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
scores = model.evaluate(X_test, Y_test, verbose=0)
print('scores:', scores)
