import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, TextVectorization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

print("👉 Bắt đầu load dữ liệu...")
df = pd.read_csv('emails.csv')
print(f"Dữ liệu có {df.shape[0]} dòng và {df.shape[1]} cột.")

print("👉 Chuẩn bị dữ liệu...")

keywords = [col for col in df.columns if col not in ['Email No.', 'Prediction']]

with open('keywords.json', 'w', encoding='utf-8') as f:
    json.dump(keywords, f, ensure_ascii=False, indent=4)

X = df.drop(columns=['Email No.', 'Prediction'])
y = df['Prediction']

print("👉 Xây dựng mô hình với TextVectorization...")

text_input = Input(shape=(1,), dtype=tf.string, name='text_input')

print(f"Số lượng từ khóa: {len(keywords)}")
vectorizer = TextVectorization(
    standardize="lower_and_strip_punctuation",
    split="whitespace",
    output_mode="count",
    vocabulary=keywords,
    name='text_vectorization'
)

x = vectorizer(text_input)
print(f"Kích thước đầu ra TextVectorization: {x.shape}")

def prepare_data_for_model(df, column_names):
    texts = []
    for _, row in df.iterrows():
        text_parts = []
        for col in column_names:
            count = int(row[col])
            if count > 0:
                text_parts.extend([col.lower()] * count)
        text = ' '.join(text_parts)
        texts.append(text)
    return tf.constant(texts, dtype=tf.string)

class MinMaxScalerLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(MinMaxScalerLayer, self).__init__(**kwargs)
        self.min_vals = None
        self.max_vals = None
    
    def adapt(self, data):
        self.min_vals = tf.reduce_min(data, axis=0)
        self.max_vals = tf.reduce_max(data, axis=0)
        if not isinstance(self.min_vals, tf.Variable):
            self.min_vals = tf.Variable(self.min_vals, trainable=False, dtype=tf.float32)
            self.max_vals = tf.Variable(self.max_vals, trainable=False, dtype=tf.float32)
        self.input_shape = data.shape[1]
        print(f"MinMaxScalerLayer adapted to shape: {self.input_shape}")
    
    def call(self, inputs):
        # Kiểm tra kích thước đầu vào
        input_shape = tf.shape(inputs)[1]
        print(f"Input shape to scaler: {input_shape}")
        inputs = tf.cast(inputs, tf.float32)
        min_vals = self.min_vals
        max_vals = self.max_vals
        return (inputs - min_vals) / (max_vals - min_vals + tf.keras.backend.epsilon())
    
    def get_config(self):
        config = super(MinMaxScalerLayer, self).get_config()
        return config

scaler_layer = MinMaxScalerLayer(name='min_max_scaler')

sample_text = tf.constant(["word1 word2"])
sample_output = vectorizer(sample_text)
vec_output_dim = sample_output.shape[1]
print(f"TextVectorization output dimension: {vec_output_dim}")

raw_values = np.zeros((len(X), vec_output_dim), dtype=np.float32)

sample_texts = prepare_data_for_model(X.head(5), keywords)
for i, text in enumerate(sample_texts):
    text_batch = tf.expand_dims(text, 0)
    vec_output = vectorizer(text_batch).numpy()
    raw_values[i] = vec_output[0]

for i, row in X.iterrows():
    if i >= 5:
        text_parts = []
        for col in keywords:
            count = int(row[col])
            if count > 0:
                text_parts.extend([col.lower()] * count)
        text = ' '.join(text_parts)
        text_tensor = tf.constant([text], dtype=tf.string)
        vec_output = vectorizer(text_tensor).numpy()
        raw_values[i] = vec_output[0]

scaler_layer.adapt(tf.convert_to_tensor(raw_values, dtype=tf.float32))

x = scaler_layer(x)

x = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
x = tf.keras.layers.Dropout(0.3)(x)
x = Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = Dense(1, activation='sigmoid')(x)

model = Model(inputs=text_input, outputs=outputs)

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("👉 Chia dữ liệu train/test...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_texts = prepare_data_for_model(X_train, keywords)
X_test_texts = prepare_data_for_model(X_test, keywords)

print("👉 Huấn luyện mô hình...")
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.2, 
    patience=3, 
    min_lr=0.00001,
    verbose=1
)

history = model.fit(
    X_train_texts, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)
print("👉 Đánh giá mô hình...")
loss, accuracy = model.evaluate(X_test_texts, y_test, verbose=0)
print(f"✅ Test Accuracy: {accuracy:.4f}")
print("👉 Lưu mô hình...")
model.save('spam_classifier_model.keras', save_format='tf')
print("🎉 Huấn luyện và lưu mô hình thành công!")
