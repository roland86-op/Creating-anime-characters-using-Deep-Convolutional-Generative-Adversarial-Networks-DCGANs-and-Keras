# Creating-anime-characters-using-Deep-Convolutional-Generative-Adversarial-Networks-DCGANs-and-Keras
<div align="center">

<img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54">
<img src="https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white">
<img src="https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white">
<img src="https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white">
<img src="https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black">
<img src="https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white">
<img src="https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white">
<img src="https://user-images.githubusercontent.com/315810/92159303-30d41100-edfb-11ea-8107-1c5352202571.png" width="30" height="30">

</div>

<h2 align="center"> Dokumentasi </h2>
<h3 align="left"> Setup </h3>
Untuk lab ini, kita akan menggunakan library berikut:

- pandas untuk mengelola data.
- numpy untuk operasi matematika.
- sklearn untuk fungsi terkait pembelajaran mesin dan pipeline pembelajaran mesin.
- seaborn untuk visualisasi data.
- matplotlib untuk alat plotting tambahan.
- keras untuk memuat dataset.
- tensorflow untuk fungsi pembelajaran mesin dan jaringan saraf.

Steps:
- Instal library, restart kernel
- Import library
- Membuat fungsi pembantu
  
<h2 align="center"> Basic: Generative Adversarial Networks (GANs) </h2>
<div>
  Generative Adversarial Networks (GANs) adalah model generatif yang mengubah sampel acak dari satu distribusi ke distribusi lain.

  Dalam bagian Lab GANs ini, kita akan menggunakan contoh sederhana untuk membantu memahami prinsip teoretis dasar di balik GANs. Bentuk asli dari GANs terdiri dari sebuah diskriminator dan generator; mari gunakan analogi pemalsu uang dan polisi.

  Generator adalah pemalsu uang, dan outputnya adalah uang palsu, misalnya, uang kertas 100 dolar. Diskriminator dianalogikan dengan polisi yang memeriksa uang palsu tersebut dan mencoba menentukan apakah uang tersebut asli dengan membandingkannya dengan uang kertas $100 yang asli. Dalam kehidupan nyata, jika uang palsu mudah dideteksi, pemalsu akan beradaptasi; sebaliknya, polisi juga akan meningkat; GANs meniru permainan kucing dan tikus ini.

  Yang membuat GANs menarik adalah bahwa diskriminator dan generator terus meningkatkan satu sama lain melalui fungsi biaya yang dirumuskan dengan baik yang melakukan backpropagation dari kesalahan. GANs adalah keluarga algoritma yang menggunakan pembelajaran melalui perbandingan. Di lab ini, kita akan meninjau formulasi asli dan menggunakan dataset simulasi. Kami juga akan menunjukkan beberapa metode yang lebih maju dan masalah yang mungkin Anda temui dengan dataset nyata untuk lab berikutnya.
</div>

<h3 align="left">The Generator</h3>
<div>
  Ada dua jaringan yang terlibat dalam GAN, yaitu Generator dan Discriminator. Mari pahami jaringan Generator terlebih dahulu.

Generator adalah jaringan saraf yang dilambangkan dengan G; idenya adalah bahwa jaringan saraf dapat mendekati fungsi apa pun (berdasarkan Teorema Pendekatan Universal), sehingga Anda dapat menghasilkan sampel data dari jenis distribusi apa pun.
</div>

<h3 align="left">The Discriminator</h3>
<div>
 Discriminator D(x) adalah jaringan saraf yang belajar membedakan antara sampel asli dan sampel yang dihasilkan. Discriminator paling sederhana adalah fungsi regresi logistik sederhana. Mari buat sebuah diskriminator di Keras dengan satu lapisan Dense; fungsi logistik tidak disertakan karena akan dimasukkan dalam fungsi biaya, yang merupakan konvensi di Keras.

Diskriminator dan generator diinisialisasi secara acak, tetapi kita dapat memplot output masing-masing dan membandingkannya dengan distribusi data asli, dengan data yang dihasilkan berwarna merah dan data asli berwarna hijau, serta fungsi logistik sebagai fungsi dari sumbu x. Kami juga menyertakan ambang batasnya. Jika output dari fungsi logistik kurang dari 0,5, sampel diklasifikasikan sebagai data yang dihasilkan; sebaliknya, jika outputnya lebih besar dari 0,5, sampel akan diklasifikasikan sebagai data yang berasal dari distribusi asli.
</div>

<h2 align="center"> Deep Convolutional Generative Adversarial Networks (DCGANs) </h2>

<h3 align="left">Dataset</h3>
<div>Kita akan menggunakan dataset Anime Face dari Kaggle. Dataset aslinya memiliki 63.632 wajah anime "berkualitas tinggi", tetapi untuk mempercepat pelatihan model dalam lab ini, kami secara acak mengambil sampel 20.000 gambar dan menyiapkan dataset yang disebut <a href="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML311-Coursera/labs/Module6/cartoon_20000.zip">cartoon_20000</a>.</div>

<h3 align="left">Building The Generator</h3>
Generator terdiri dari beberapa lapisan transposed convolution, yang merupakan kebalikan dari operasi konvolusi.
Setiap lapisan Conv2DTranspose (kecuali lapisan terakhir) diikuti oleh lapisan Batch Normalization dan aktivasi Relu; untuk detail implementasi lebih lanjut.
Lapisan konvolusi transpos akhir memiliki tiga saluran output karena output harus berupa gambar berwarna. Kami menggunakan aktivasi Tanh di lapisan terakhir.
<br>

<h3 align="left">Building The Discriminator</h3>
<div>
Discriminator memiliki lima lapisan konvolusi.
Semua kecuali lapisan Conv2D pertama dan terakhir memiliki Batch Normalization, karena menerapkan batchnorm secara langsung ke semua lapisan dapat menyebabkan osilasi sampel dan ketidakstabilan model.
Empat lapisan Conv2D pertama menggunakan aktivasi Leaky-Relu dengan kemiringan 0,2.
Terakhir, lapisan output memiliki lapisan konvolusi dengan fungsi aktivasi Sigmoid.
</div>

<h3 align="left">Defining Loss Functions</h3>
<div>
  Seperti yang dibahas di bagian sebelumnya, masalah optimasi min-max dapat diformulasikan dengan meminimalkan cross entropy loss untuk Generator dan Discriminator.
Objek cross_entropy adalah Binary Cross Entropy loss yang akan digunakan untuk memodelkan tujuan dari kedua jaringan.
</div>

<h3 align="left">Create Train Step Function</h3>
Kita membuat dua optimizer Adam untuk diskriminator dan generator, masing-masing. Kami memberikan argumen berikut ke optimizer:learning rate sebesar 0,0002. dan koefisien beta β1=0,5 dan β2=0,999, yang bertanggung jawab untuk menghitung rata-rata berjalan dari gradien selama backpropagation.
<br>

<h3 align="left">Creating Train Step Function</h3>
  Ringkasan tugas train step:
    <ul>Mengambil sampel z, sekumpulan vektor kebisingan dari distribusi normal ( μ=1,σ=1 ) dan memasukkannya ke Generator.</ul>
    <ul>Generator menghasilkan gambar yang dihasilkan atau "palsu".</ul>
    <ul>Kami memasukkan gambar asli X dan gambar palsu xhat ke Diskriminator dan mendapatkan real_output dan fake_output masing-masing sebagai skor.</ul>
    <ul>Kami menghitung kehilangan Generator gen_loss menggunakan output_palsu dari Diskriminator karena kami ingin gambar palsu tersebut menipu Diskriminator sebanyak mungkin.</ul>
    <ul>Kami menghitung kerugian Diskriminator disc_loss menggunakan real_output dan fake_output karena kami ingin Diskriminator membedakan keduanya sebanyak mungkin.</ul>
    <ul>Kami menghitung gradien_of_generator dan gradien_of_diskriminator berdasarkan kerugian yang diperoleh.</ul>
    <ul>Terakhir, kami memperbarui Generator dan Diskriminator dengan membiarkan pengoptimal masing-masing menerapkan gradien yang diproses pada parameter model yang dapat dilatih.</ul>

<h3 align="left">Training DCGANs</h3>
<div>Melatih GAN dengan hanya satu epoch memakan waktu yang cukup lama. Jika kita ingin mengevaluasi kinerja GAN yang sepenuhnya dilatih dan dioptimalkan, kita perlu menambah jumlah epoch. Oleh karena itu, untuk membantu Anda menghindari waktu pelatihan yang sangat lama di lab ini, kita hanya akan mengunduh parameter jaringan Generator yang telah dilatih sebelumnya dan kemudian menggunakan fungsi load_model Keras untuk memperoleh Generator yang telah dilatih, yang akan kita gunakan untuk menghasilkan gambar secara langsung.</div>
