import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === Ham luu va hien thi anh ===
def save_and_show_plot(plt, filename, folder="eda_images"):
    """Luu va hien thi bieu do."""
    if not os.path.exists(folder):
        os.makedirs(folder)
    save_path = os.path.join(folder, filename)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"\u0110a~ luu anh vao: {save_path}")

# === Doc du lieu ===
df = pd.read_csv("emails.csv")  # Doi lai ten file cho dung
print("So luong dong:", df.shape[0])
print("So luong cot:", df.shape[1])

values = df['Prediction'].value_counts()
total = values.sum()

percentage_0 = (values[0] /total) * 100
percentage_1 = (values[1]/ total) *100

colors = ['#FF5733', '#33FF57']

# Define the explode parameter to create a gap between slices
explode = (0, 0.1)  # Explode the second slice (spam) by 10%

# Create a figure with a white background
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_facecolor('white')

# Create the pie chart with custom colors, labels, explode parameter, and shadow
wedges, texts, autotexts = ax.pie(
    values, labels=['ham', 'spam'],
    autopct='%0.2f%%',
    startangle=90,
    colors=colors,
    wedgeprops={'linewidth': 2, 'edgecolor': 'white'},
    explode=explode,  # Apply the explode parameter
    shadow=True  # Add shadow
)

# Customize text properties
for text, autotext in zip(texts, autotexts):
    text.set(size=14, weight='bold')
    autotext.set(size=14, weight='bold')

# Add a title
ax.set_title('Email Classification', fontsize=16, fontweight='bold')

# Equal aspect ratio ensures that pie is drawn as a circle
ax.axis('equal')

# Show the pie chart
plt.show()

# === Phan phoi nha~n Spam va Ham ===
plt.figure(figsize=(5, 4))
df["Prediction"].value_counts().plot(kind="bar", color=["skyblue", "salmon"])
plt.title("So luong email: Ham (0) vs Spam (1)")
plt.xticks(ticks=[0, 1], labels=["Ham", "Spam"], rotation=0)
plt.ylabel("So luong")
save_and_show_plot(plt, "label_distribution.png")

# === Trung binh tan suat xuat hien tu trong spam va ham ===
feature_cols = df.columns.difference(["Email No.", "Prediction"])

mean_spam = df[df["Prediction"] == 1][feature_cols].mean()
mean_ham = df[df["Prediction"] == 0][feature_cols].mean()

# === Top 20 tu pho bien trong Spam ===
top_spam = mean_spam.sort_values(ascending=False).head(20)
plt.figure(figsize=(9, 5))
top_spam.plot(kind="bar", color="salmon")
plt.title("Top 20 tu xuat hien nhieu nhat trong email Spam")
plt.ylabel("Tan suat trung binh")
plt.xticks(rotation=90, ha="right")
save_and_show_plot(plt, "top_words_spam.png")

# === Top 20 tu pho bien trong Ham ===
top_ham = mean_ham.sort_values(ascending=False).head(20)
plt.figure(figsize=(9, 5))
top_ham.plot(kind="bar", color="skyblue")
plt.title("Top 20 tu xuat hien nhieu nhat trong email Ham")
plt.ylabel("Tan suat trung binh")
plt.xticks(rotation=90, ha="right")
save_and_show_plot(plt, "top_words_ham.png")

# === (Tuy chon) Tim tu tuong quan cao nhat voi Prediction ===
# Warning: Mat thoi gian voi file lon!
corr = df[feature_cols.tolist() + ["Prediction"]].corr()
top_corr = corr["Prediction"].abs().sort_values(ascending=False).head(20)
print("\nTop 20 dac trung tuong quan cao voi nha~n:")
print(top_corr)