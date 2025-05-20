import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer


class MinMaxScalerLayer(Layer):
    def __init__(self, **kwargs):
        super(MinMaxScalerLayer, self).__init__(**kwargs)
        self.min_vals = tf.Variable(tf.zeros(3001), trainable=False, dtype=tf.float32)
        self.max_vals = tf.Variable(tf.ones(3001), trainable=False, dtype=tf.float32)
    
    def build(self, input_shape):
        if self.min_vals.shape[0] != input_shape[1]:
            self.min_vals = tf.Variable(tf.zeros(input_shape[1]), trainable=False, dtype=tf.float32)
            self.max_vals = tf.Variable(tf.ones(input_shape[1]), trainable=False, dtype=tf.float32)
    
    def call(self, inputs):
        
        inputs = tf.cast(inputs, tf.float32)
        
        return inputs
    
    def get_config(self):
        config = super(MinMaxScalerLayer, self).get_config()
        return config

custom_objects = {"MinMaxScalerLayer": MinMaxScalerLayer}

print("👉 Bắt đầu load model...")

model = load_model("spam_classifier_model.keras", custom_objects=custom_objects)

print("👉 Load danh sách từ khóa...")
with open("keywords.json", "r", encoding="utf-8") as f:
    keywords = json.load(f)

print("👉 Load email_test.json...")
with open("email_test.json", "r", encoding="utf-8") as f:
    emails = json.load(f)

def email_to_text_features(email_text, keywords_list):
    """
    Chuyển đổi email thô thành text có định dạng:
    - Phân tích text thành từng từ
    - Chỉ giữ lại những từ trong danh sách keywords
    - Đếm số lần xuất hiện của mỗi từ khóa
    """
    email_text = email_text.lower()
    for ch in "!@#$%^&*()[]{};:,./<>?\\|`~-=+\"'":
        email_text = email_text.replace(ch, " ")
    words = email_text.split()
    text_features = []
    for keyword in keywords_list:
        keyword = keyword.lower()
        count = words.count(keyword)
        if count > 0:
            text_features.extend([keyword] * count)
    
    return " ".join(text_features)

print("👉 Bắt đầu dự đoán các email:")

for idx, email in enumerate(emails, start=1):
    email_text = email['email']
    text_features = email_to_text_features(email_text, keywords)
    input_text = tf.constant([text_features])
    prediction = model.predict(input_text, verbose=0)
    label = "Spam" if prediction[0][0] >= 0.5 else "Ham"
    confidence = prediction[0][0] if prediction[0][0] >= 0.5 else 1 - prediction[0][0]
    print(f"Email {idx}: {label} (Độ tin cậy: {confidence:.2f})")
print("🎉 Hoàn thành quá trình dự đoán!")