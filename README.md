# Is the mushroom edible?
Roy Andhika Satria, [satriaroy70@gmail.com](mailto:satriaroy70@gmail.com).

## Classification
Sebuah proyek machine learning yang menggunakan algoritma klasifikasi untuk memprediksi apakah sebuah jamur beracun atau tidak.

## Dataset
Data bersumber dari [Kaggle](https://www.kaggle.com/datasets/uciml/mushroom-classification).  
Data berisi informasi mengenai daftar jenis jamur yang aslinya berasal dari **UCI Machine Learning repository** hampir 30 tahun yang lalu. *Shrooming* (mushroom hunting) sempat menjadi populer pada masa itu. Dataset ini memiliki sampel dari 23 spesies jamur, dan tiap spesies diidentifikasi sebagai *edible* atau *poisonous*.

## Tujuan
Harapannya saya bisa menentukan apakah sebuah jamur bisa dimakan atau tidak, berdasarkan dari ciri - cirinya dengan menggunakan data yang telah ada sebagai data *training* untuk machine learning.

### Problem Statement :
1. Model apa yang paling tepat untuk klasifikasi jamur?
1. Fitur apa yang paling menentukan sebuah jamur beracun?
1. Matriks dan scoring apa akan digunakan untuk pengukuran kualitas machine learning?

## Penjelasan Dataset 
Dataset yang ada meliputi `8124` pengamatan dengan `22` fitur dan 1 label (`Class`)

| No | __Nama Fitur__ | __Penjelasan__ |
| - | - | - |
| 1 | class | edible=e, poisonous=p | 
| 2 | cap-shape | bell=b,conical=c,convex=x,flat=f, knobbed=k,sunken=s |
| 3 | cap-surface | fibrous=f,grooves=g,scaly=y,smooth=s |
| 4 | cap-color  | brown=n,buff=b,cinnamon=c,gray=g,green=r,pink=p,purple=u,red=e,white=w,yellow=y |
| 5 | bruises | bruises=t,no=f | 
| 6 | odor | almond=a,anise=l,creosote=c,fishy=y,foul=f,musty=m,none=n,pungent=p,spicy=s |
| 7 | gill-attachment | attached=a, descending=d, free=f, notched=n |
| 8 | gill-spacing | close=c,crowded=w,distant=d |
| 9 | gill-size | broad=b,narrow=n |
| 10 | gill-color | black=k,brown=n,buff=b,chocolate=h,gray=g, green=r,orange=o,pink=p,purple=u,red=e,white=w,yellow=y | 
| 11 | stalk-shape | enlarging=e,tapering=t | 
| 12 | stalk-root | bulbous=b,club=c,cup=u,equal=e,rhizomorphs=z,rooted=r,missing=? |
| 13 | stalk-surface-above-ring | fibrous=f,scaly=y,silky=k,smooth=s | 
| 14 | stalk-surface-below-ring | fibrous=f,scaly=y,silky=k,smooth=s |
| 15 | stalk-color-above-ring | brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y |
| 16 | stalk-color-below-ring | brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y |
| 17 | veil-type | partial=p,universal=u |
| 18 | veil-color | brown=n,orange=o,white=w,yellow=y |
| 19 | ring-number | none=n,one=o,two=t |
| 20 | ring-type | cobwebby=c,evanescent=e,flaring=f,large=l,none=n,pendant=p,sheathing=s,zone=z |
| 21 | spore-print-color | black=k,brown=n,buff=b,chocolate=h,green=r,orange=o,purple=u,white=w,yellow=y |
| 22 | population | abundant=a,clustered=c,numerous=n,scattered=s,several=v,solitary=y |
| 23 | habitat | grasses=g,leaves=l,meadows=m,paths=p,urban=u,waste=w,woods=d |

## Steps
1. Data Understanding
2. Exploratory Data Analysis (EDA)
3. Preprocessing 
4. Modeling & Conclusion 

### 1. Data Understanding
1. Terdapat 22 fitur dan 1 label pada dataset
2. Tidak ada missing value atau null value
3. Data semuanya sudah dalam bentuk kategorikal
4. Perbandingan data target e & p sudah seimbang

### 2. Ringkasan Exploratory Data Analysis (EDA)
1. Jamur yang bisa dimakan, 80% memiliki **odor = n** atau tidak berbau.
![graph2](https://raw.githubusercontent.com/royandhika/classification-mushroom/main/assets/odor.png)
2. Hampir keseluruhan data memiliki fitur **gill-attachment = f** baik edible maupun poisonous, fitur kurang relevan dan dipertimbangkan untuk menghapus fitur ini.
![graph1](https://raw.githubusercontent.com/royandhika/classification-mushroom/main/assets/gill-attachment.png)
3. Jamur beracun **97%** memiliki **gill-spacing = c**.
![graph3](https://raw.githubusercontent.com/royandhika/classification-mushroom/main/assets/gill-spacing.png)
4. Jamur beracun **44%** memiliki **gill-color = b** atau kuning ke-krem-an, sedangkan jamur yang bisa dimakan tidak memiliki warna ini sehingga dapat dipastikan jika menemukan jamur dengan **gill-color = b pasti beracun**.
![graph4](https://raw.githubusercontent.com/royandhika/classification-mushroom/main/assets/gill-color.png)
5. **veil-color** mayoritas data memiliki warna putih, fitur kurang relevan dan dipertimbangkan untuk menghapus fitur ini.
![graph5](https://raw.githubusercontent.com/royandhika/classification-mushroom/main/assets/veil-color.png)

### 3. Preprocessing
| Target |	Persentase	| 
| - | - | 
| Edible	| 51.79% |	
| Poisonous	| 48.21%	| 

Data target sudah cukup seimbang, tidak perlu dilakukan balancing  
Kemudian data kategorikal semuanya diubah ke dalam bentuk 1 dan 0 menggunakan One-hot-encoding

### 4. Base Model / Benchmarking
Menggunakan single testing dan kemudian divalidasi ulang menggunakan stratifiedKFold dengan k = 5. Scoring yang digunakan adalah 'f1-score'.

| Model | F1 score | 5-fold validation (mean) | Waktu |
|:-:|:-:|:-:|:-:|
| Logistic Regression | 1.00 | 0.999 | 0.5s |
| Categorical Naive Bayes | 0.93 | 0.937 | 0.4s |
| Decision Tree | 1.00 | 1.00 | 0.4s |
| K Nearest Neighbor | 1.00 | 1.00 | 1.3s |
| Random Forest | 1.00 | 1.00 | 1.6s |
| Support Vector Classification | 1.00 | 1.00 | 4.8s |

Berdasarkan hasil confusion matrix dan waktu processing, kita dapati bahwa performa terbaik terdapat dalam model Decision Tree Classifier. 
