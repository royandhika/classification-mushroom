# Is the mushroom edible?
Roy Andhika Satria, [email](mailto:satriaroy70@gmail.com).

## Classification
Sebuah proyek machine learning yang menggunakan algoritma klasifikasi untuk memprediksi apakah sebuah jamur beracun atau tidak.

## Dataset
Data bersumber dari [Kaggle](https://www.kaggle.com/datasets/uciml/mushroom-classification).

Data berisi informasi mengenai daftar jenis jamur yang aslinya berasal dari **UCI Machine Learning repository** hampir 30 tahun yang lalu. *Shrooming* (mushroom hunting) sempat menjadi populer pada masa itu. Dataset ini memiliki sampel dari 23 spesies jamur, dan tiap spesies diidentifikasi sebagai *edible* atau *poisonous*.

## Tujuan
Harapannya saya bisa menentukan apakah sebuah jamur bisa dimakan atau tidak, berdasarkan dari ciri - cirinya dengan menggunakan data yang telah ada sebagai data *training* untuk machine learning.
---
### Problem Statement for Machine Learning:

1. Berapa banyak keuntungan yang hilang karena pelanggan churn? 
1. Model apa yang paling tepat untuk mendeteksi pelanggan yang akan churn?
1. Bagaimana memprediksi pelanggan yang churn dengan baik, sehingga dapat meminimalisasi prediksi yang berupa false negatif?
1. Matriks apa yang akan digunakan untuk pengukuran kualitas machine learning?

### Problem Statement for Analytics:

1. Ciri - ciri apa yang paling sering ditemukan dalam jamur beracun?
1. Apa yang membuat customer bertahan menggunakan provider? Produk unggulan nya apa?
1. Layanan apa yang paling tidak diminati oleh customer?

## Penjelasan Dataset 
Dataset yang ada meliputi `7043` pengamatan dengan `20` fitur dan 1 label (`Churn`)

| No | __Nama Fitur__ | __Penjelasan__ | __Data Type__ |
| - | - | - | - | 
| 1 | class | edible=e, poisonous=p | categorical | 
| 2 | cap-shape | bell=b,conical=c,convex=x,flat=f, knobbed=k,sunken=s | categorical |
| 3 | cap-surface | fibrous=f,grooves=g,scaly=y,smooth=s | categorical |
| 4 | cap-color  | brown=n,buff=b,cinnamon=c,gray=g,green=r,pink=p,purple=u,red=e,white=w,yellow=y | categorical |
| 5 | bruises | bruises=t,no=f | categorical | 
| 6 | odor | almond=a,anise=l,creosote=c,fishy=y,foul=f,musty=m,none=n,pungent=p,spicy=s | categorical |
| 7 | gill-attachment | attached=a, descending=d, free=f, notched=n | categorical |
| 8 | gill-spacing | close=c,crowded=w,distant=d | categorical |
| 9 | gill-size | broad=b,narrow=n | categorical |
| 10 | gill-color | black=k,brown=n,buff=b,chocolate=h,gray=g, green=r,orange=o,pink=p,purple=u,red=e,white=w,yellow=y | categorical | 
| 11 | stalk-shape | enlarging=e,tapering=t | categorical | 
| 12 | stalk-root | bulbous=b,club=c,cup=u,equal=e,rhizomorphs=z,rooted=r,missing=? | categorical |
| 13 | stalk-surface-above-ring | fibrous=f,scaly=y,silky=k,smooth=s | categorical | 
| 14 | stalk-surface-below-ring | fibrous=f,scaly=y,silky=k,smooth=s | categorical |
| 15 | stalk-color-above-ring | brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y | categorical |
| 16 | stalk-color-below-ring | brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y | categorical |
| 17 | veil-type | partial=p,universal=u | categorical |
| 18 | veil-color | brown=n,orange=o,white=w,yellow=y | categorical |
| 19 | ring-number | none=n,one=o,two=t | categorical |
| 20 | ring-type | cobwebby=c,evanescent=e,flaring=f,large=l,none=n,pendant=p,sheathing=s,zone=z | categorical |
| 21 | spore-print-color | black=k,brown=n,buff=b,chocolate=h,green=r,orange=o,purple=u,white=w,yellow=y | categorical |
| 22 | population | abundant=a,clustered=c,numerous=n,scattered=s,several=v,solitary=y | categorical |
| 23 | habitat | grasses=g,leaves=l,meadows=m,paths=p,urban=u,waste=w,woods=d | categorical |

## Steps

1. Data Understanding
2. Exploratory Data Analysis (EDA)
3. Modeling & Conclusion 

### Data Understanding

1. Terdapat 22 fitur dan 1 label pada dataset
2. Tidak ada missing value atau null value
3. Data semuanya sudah dalam bentuk categorical

### Ringkasan Exploratory Data Analysis (EDA)


1. Terdapat pemasukan kotor sebesar $2.862.926,9 dollar AS yang hilang akibat pelanggan yang churn.
2. Gender: Hampir tidak ada perbedaan perbedaan persentase churn antara pria dan wanita.
3. Senior Citizen: Persentase churn dalam kelompok senior citizen adalah 42%, ada indikasi churn yang tinggi pada kelompok ini.
4. Partner dan Dependents: Kedua fitur memiliki korelasi dan berkontribusi besar dalam kecenderungan pelanggan untuk churn.
5. Phone Service dan Internet Service: Ada sebagian pelanggan yang tidak memiliki internet service dan bahkan sejumlah kecil pelanggan tidak memiliki phone service. Sebagian besar feature lainnya yang tersedia berhubungan dengan internet service.
6. Internet Service: Pelanggan yang menggunakan DSL membayar biaya bulanan yang lebih murah daripada pengguna fiber optic. Pengguna fiber optic juga terlihat cenderung lebih banyak untuk churn.
7. Online Security, Online Backup, Device Protection, Tech Support: Keempat layanan tersebut hanya bisa diakses jika pengguna memiliki layanan internet. Pengguna keempat layanan tersebut cenderung untuk tidak churn. Terutama pengguna Tech Support dan Online Security.
8. Streaming TV dan Movies: Menunjukkan data yang sangat mirip. Baik pengguna maupun tidak, perbedaan pelanggan yang churn hanya sedikit.
9. Contract: Pelanggan yang membayar bulanan lebih banyak yang churn.
10. Paperless Billing: Pelanggan yang menggunakan paperless billing lebih banyak yang churn daripada yang tidak. Hal ini dikarenakan banyak pelanggan yang membayar bulanan juga menggunakan electronic check (bagian dari paperles billing) sebagai metode pembayaran.
11. Tenure: Churn paling banyak terjadi di bulan-bulan awal.
12. Monthly Charges dan Total Charges: Churn paling banyak terjadi di biaya bulanan yang tinggi. Rata rata biaya bulanan untuk pelanggan yang churn lebih rendah daripada yang tidak (karena hanya bayar beberapa bulan saja).

### Modeling

| Target |	Persentase	| 
| - | - | 
| Churn	| 26.54% |	
| Tidak	| 73.46%	| 

Ada 2 cara yang dapat digunakan untuk mengatasi data target (y) yang imbalance:

  1. Setting parameter "(class_weight='balanced')" untuk model yang memilikinya dan masing-masing "random_state" parameter.
  1. Menggunakan metode SMOTE oversampling dan setting "random_state" parameter untuk model yang tidak memiliki "class_weight" parameter.

### Base, Tune and Hyperparameter Tuning Model

RepeatedStratifiedKFold dan GridSearchCV digunakan untuk mencari nilai paramater yang terbaik. Scoring yang digunakan adalah 'recall'.

| Model |  Sebelum  | Setelah |
|:-:|:-:|:-:|
| Logistic Regression | 0.787 | 0.799 |
| Decision Tree Classifier | 0.531 | 0.753 |
| Random Forest Classifier | 0.538 | 0.713 |
| XGBoost Classifier | 0.559 | 0.643 |
| K Nearest Classifier | 0.764 | 0.632 |

Berdasarkan hasil confusion matrix dan waktu perhitungan, kita dapati bahwa performa terbaik terdapat dalam model Decision Tree Classifier. Setelah dilakukan hyperparameter tuning pada model, 'recall' score bertambah drastis, yaitu dari 0.48 menjadi 0.85

### Decision Tree Classifier Confusion Matrix

#### Sebelum Hyperparameter Tuning
![dtc](https://user-images.githubusercontent.com/88280579/139032798-7452b98e-1834-4eec-90f7-79e0d1167f08.png)

#### Setelah Hyperparameter Tuning
![dtc_tuned](https://user-images.githubusercontent.com/88280579/139032956-27751966-992b-4578-a7c5-5319ef086668.png)


### Kesimpulan

Model machine learning yang dikembangkan dapat membantu perusahaan untuk menurunkan biaya dan waktu dalam memprediksi pelanggan mana yang akan berhenti. Sebagaimana disebutkan pada bagian problem stetment, fokus kita adalah false negative rate. Dimana kita ketahui sebelumnya bahwa, biaya yang dikeuarkan untuk memperoleh pelanggan 5 kali lebih besar dari biaya mempertahankan pelanggan yang sudah ada.

Dapat kita simpulkan, dari 100 pelanggan yang diprediksi, machine learning hanya menghasilkan kesalahan berjumlah 9 pelanggan atau 8,55%. Artinya 9 pelanggan tersebut tidak akan mendapat perlakuan khusus dari perusahaan dan akan berhenti berlangganan. Terdapat peningkatan dari model sebelumnya sebelum dituning, yakni dari 100 pelanggan yang diteliti, kesalahan prediksi mencapai 19 orang atau 18,64%.

Namun, yang perlu diingat juga bahwa terdapat konsekuensi lain. Yakni dari 100 pelanggan yang diprediksi akan berhenti berlangganan, ternyata 58,36% nya, atau 59 orang akan mendapatkan perlakuan khusus, tapi ternyata mereka tidak ada kecenderungan untuk churn dan akan tetap berlangganan. Jika dibandingkan model sebelumnya juga terdapat peningkatan dari 50.69% atau hanya 51 orang.

### Business Insight

1. Terdapat pemasukan kotor sebesar $2.862.926,9 dollar AS yang hilang akibat pelanggan yang churn.
2. Perusahaan harus lebih memperhatikan:
    - Senior citizen, karena lebih berpotensi untuk churn.
    - Fiber optic, baik dari segi harga maupun kualitas. Karena ada indikasi dapat meningkatkan potensi pelanggan untuk churn.
    - Tech support dan online security sebagai fitur unggulan. Dapat dijadikan sebagai sarana promosi kepada pelanggan yang berpotensi untuk churn.
    - 12 bulan pertama pelanggan mulai menggunakan layanan adalah masa yang penting untuk membuat pelanggan merasa nyaman dengan pelayanan yang diberikan. 
3. Tidak ada perbedaan signifikan bagi pria maupun wanita dalam kecenderungan untuk churn. Strategi apapun yang akan dilakukan dapat berjalan pada kedua gender.
