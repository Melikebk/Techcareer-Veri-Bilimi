### Veri Görselleştirme Ödevi
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv(r"C:\Users\EXCALIBUR\OneDrive\Masaüstü\Programlama\Proje\50_Startups.csv")


print(df.head())
## 1. Veri Görselleştirme

plt.figure(figsize=(8,6))
plt.scatter(df["R&D Spend"], df["Profit"], alpha=0.7, color="blue")

plt.xlabel("R&D Spend (Ar-Ge Harcaması)")
plt.ylabel("Profit (Kâr)")
plt.title("R&D Harcaması ile Kâr Arasındaki İlişki")

plt.grid(True)
plt.show()

## 2. Yönetim harcaması ile kâr arasındaki ilişkiyi scatter plot ile gösterin.
plt.figure(figsize=(8,6))
plt.scatter(df["Administration"], df["Profit"], alpha=0.7, color="green")
plt.xlabel("Administration (Yönetim Harcaması)")
plt.ylabel("Profit (Kâr)")
plt.title("Yönetim Harcaması ile Kâr Arasındaki İlişki")
plt.grid(True)
plt.show()

## 3. Eyaletlere göre ortalama kârları bar chart ile karşılaştırın.
plt.figure(figsize=(8,6))
state_means = df.groupby("State")["Profit"].mean().reset_index()
sns.barplot(x="State", y="Profit", data=state_means, palette="Set2")
plt.title("Eyaletlere Göre Ortalama Kâr")
plt.ylabel("Ortalama Kâr")
plt.xlabel("Eyalet")
plt.show()

## 4. R&D, yönetim ve pazarlama harcamalarının dağılımını boxplot ile karşılaştırın.
plt.figure(figsize=(8,6))
sns.boxplot(data=df[["R&D Spend", "Administration", "Marketing Spend"]])
plt.title("Harcama Türlerinin Dağılımı (Boxplot)")
plt.ylabel("Harcama Tutarı")
plt.show()

### DECİSİON TREE ÖDEVİ 

## 1. Eksik veya aykırı değerleri kontrol et

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\EXCALIBUR\OneDrive\Masaüstü\Programlama\Proje\dava_sonuclari.csv")

print(df.isnull().sum())

df = df.dropna()
numeric_cols = df.select_dtypes(include='number').columns
print(df[numeric_cols].describe())

## 2. Eğitim ve test seti

X = df.drop("karar", axis=1)   # karar = sınıf sütunu
y = df["karar"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

## 3. Decision Tree Modeli

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

## 4. Metrik Hesaplama

y_pred = dt_model.predict(X_test)


print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average="weighted"))
print("Recall:", recall_score(y_test, y_pred, average="weighted"))
print("F1-Score:", f1_score(y_test, y_pred, average="weighted"))

## 5. Karar Ağacı görselleştirme

plt.figure(figsize=(20,10))
plot_tree(dt_model, feature_names=X.columns, class_names=dt_model.classes_, filled=True, rounded=True)
plt.show()


### K-Means Kümeleme Ödevi 

## 1. Uygun özellikleri seç

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns

df2 = pd.read_csv(r"C:\Users\EXCALIBUR\OneDrive\Masaüstü\Programlama\Proje\dava.csv")

X2 = df2.select_dtypes(include='number')

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X2)

##  2. Elbow yöntemi ile küme sayısı

inertia = []
K_range = range(1, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_scaled)
    inertia.append(km.inertia_)


plt.figure(figsize=(8,5))
plt.plot(K_range, inertia, marker='o')
plt.xlabel("Küme Sayısı")
plt.ylabel("Inertia")
plt.title("Elbow Yöntemi ile Optimal Küme Sayısı")
plt.show()

## 3. K-Means ile kümeleme

optimal_k = 3 
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

df2['Cluster'] = clusters
##  4. Sonuçların Görselleştirilmesi
plt.figure(figsize=(8,6))
sns.scatterplot(x=X2.iloc[:,0], y=X2.iloc[:,1], hue=df2['Cluster'], palette='Set1')
plt.title("Kümeleme Sonuçları (İlk 2 Özellik)")
plt.show()