# NOTES — Multivariate MNAR Imputation of Well Logs (konteks proyek & keputusan)

> Catatan ini merangkum diskusi di claude.ai (Project "Lithology Multivariate Imputation").
> Untuk sesi Claude Code: baca file ini + README.md dulu sebelum mengubah kode.
> Status keputusan: [FINAL] = sudah disetujui, [OPEN] = belum diputuskan.

## 1. Konteks
- Data: well logs Poseidon Field (sumber NOPIMS), 6 well, 6 kelas litologi, 6.811 baris (sama dengan RINENG).
- Paper sebelumnya: Litho-Class untuk RINENG, sudah disubmit (under review). Reviewer 1 (poin 6.2) meminta imputasi multivariate.
- Motivasi empiris: co-missing DTCO–DTSM = 99,92% di data mentah. Di RINENG DTSM dipakai sebagai prediktor, padahal di lapangan hilang bersamaan dengan DTCO (hanya ditulis sebagai limitation).
- Supervisor: nilai dulu hasil conference sebelum memutuskan ekstensi ke journal; sarankan variasi seed masking (Monte Carlo).

## 2. Rencana publikasi
- [FINAL] Paper berikutnya (conference, IEEE, 2 kolom, target 6–8 hal; batas 6–10 hal): imputasi multivariate di bawah MNAR, imputation-only (tanpa klasifikasi downstream).
- Journal (lanjutan): ekstensi ke klasifikasi (6 well, 6 kelas) + multivariate, dan opsi tail-robust (QRILC, GSimp, kNN, robust linear models). Skenario "sonic juga hilang (DTCO+DTSM)" disimpan untuk journal.
- Rencana lama (conference = tail-robust, journal = multivariate) digantikan usulan teman: conference = multivariate, journal = ekstensi ke klasifikasi.

## 3. Keputusan desain [FINAL]
- Target imputasi (yang hilang): RS, NPHI, RHOB. (Skenario awal DTCO+RHOB+NPHI dan skenario A DTCO+DTSM tidak dipakai di conference.)
- Prediktor selalu tersedia: GR, DTCO, DTSM, RD. Ini asumsi dan limitation (tidak realistis bila sonic ikut hilang) -> diarahkan ke journal.
- Data: 6 well, 6 kelas (imputasi tidak peduli balancing kelas).
- Validasi: LOWO (leave-one-well-out), 6 fold.
- Masking MNAR componentwise (self-masking) independen per log:
  - Per litologi dan per log: kandidat = nilai <= kuantil q atau >= kuantil 1-q, q = 0,20 (kapasitas tail 40%).
  - Pilih acak 30% baris (dari baris litologi tsb) di antara kandidat. SATU level saja (30%).
  - Diterapkan ke train well dan test well, dengan RNG independen.
  - Alasan independen per log: kalau masking dikondisikan pada log lain, jadi MAR dan pembeda utama paper hilang.
  - 3 seed masking (pada 30% pemilihan baris di dalam tail masih acak -> seed bermakna; menjawab saran Monte Carlo supervisor).
  - k = jumlah log hilang per baris (0–3); laporkan proporsinya.
- Metode: Mean, MICE (IterativeImputer), LGBM Regressor (satu model per target; fitur = 4 prediktor + 2 target lain; NaN native, single pass; early stopping, konfigurasi samakan dengan RINENG).
  - Listwise deletion hanya dinilai lewat jumlah baris tersisa (opsional).
  - KNN/missForest dan deep learning TIDAK dievaluasi (versi dipangkas); tulis di limitation.
- Pipeline tanpa leakage: split LOWO -> log10(RS) -> MinMax di-fit pada train (sebelum masking, seperti RINENG) -> masking -> fit imputer hanya pada baris train teramati -> imputasi test -> inverse MinMax -> 10**RS -> simpan hanya sel test yang di-mask.
- Evaluasi: RMSE, MAE (satuan asli), nRMSE (dibagi range dataset), R^2; dari residual mentah yang di-pool lintas fold (bukan rata-rata per fold); rata-rata ± sd antar seed. Breakdown per k (1/2/3) dan per well. Tanpa uji signifikansi (deskriptif).

## 4. Rumusan masalah (versi dipangkas, 3 RQ)
- RQ1: Seberapa akurat Mean/MICE/LGBM merekonstruksi RS, NPHI, RHOB pada well yang tidak dipakai training dengan masking MNAR 30%?
- RQ2: Bagaimana akurasi berubah menurut jumlah log yang hilang bersamaan dalam satu baris (k = 1, 2, 3)?
- RQ3: Seberapa stabil hasil antar well (LOWO) dan antar realisasi masking (seed)?
- (Versi 4 RQ sebelumnya memuat galat tail vs tengah distribusi; dipangkas. Di Discussion tulis error tail "tidak dievaluasi langsung".)

## 5. Outline paper (IEEE conference)
Tidak perlu Nomenclature, Practical Implications, Data Availability terpisah (dilebur ke Discussion/Conclusion).
- I. Introduction (~0,8 hal): P1 log sumur untuk interpretasi formasi, missing data umum; P2 mekanisme MCAR/MAR/MNAR, missingness lapangan sering serentak (co-missing 99,92%); P3 gap: studi imputasi umumnya univariat/random split, belum ada evaluasi multi-log di bawah MNAR dengan validasi well-level; P4 tujuan, 3 RQ, 2 poin kontribusi.
- II. Related Works (~0,6 hal): A. imputasi data log sumur (statistik, ML, tree-based); B. mekanisme missingness dan masking MNAR, positioning terhadap Gama et al. 2025 dan Hallam et al. 2022, self-masking componentwise (Muzellec et al. 2020). Opsional tabel kecil perbandingan studi terdahulu.
- III. Methodology (~2 hal): A. Dataset (satu paragraf pendek, rujuk RINENG under review + NOPIMS, jangan salin teks RINENG); B. LOWO + urutan pipeline tanpa leakage; C. masking MNAR multivariate (tail per litologi, q=0,20, 30%, independen per log, 3 seed, definisi k); D. metode imputasi; E. metrik.
- IV. Results (~2 hal): A. hasil masking (proporsi k=1/2/3); B. akurasi per log × metode (RQ1); C. pengaruh k (RQ2); D. stabilitas antar well dan seed (RQ3).
- V. Discussion (~0,6 hal): kapan LGBM unggul/sebanding MICE; error naik di tail (tree tidak ekstrapolasi, tulis "tidak dievaluasi langsung"); implikasi: rekonstruksi log sebelum interpretasi litologi, tahan klaim (tidak ada evaluasi downstream); limitations: 6 well satu basin, DTCO/DTSM/GR/RD selalu tersedia, satu level masking (30%), mekanisme simulasi, KNN/missForest/deep learning tidak dievaluasi.
- VI. Conclusion (~0,3 hal): apa yang dievaluasi, 2–3 temuan kunci dengan angka, batas bukti (tested wells), future work: sonic juga hilang, ekstensi ke klasifikasi (journal).
- References (~0,6 hal, 25–30 ref, IEEE bernomor).
- Gambar/tabel: Gambar 1 diagram pipeline; Tabel I ringkasan dataset per well; Tabel II log (RS, NPHI, RHOB) × metode (RMSE, nRMSE, R^2, mean ± sd antar seed); Tabel III k × metode (nRMSE); Gambar 2 parity plot LGBM per log (3 panel). Opsional: sd per well, cukup satu kalimat.

## 6. Risiko / hal yang dijaga
1. Overlap dengan RINENG: RQ harus jelas beda ("akurasi rekonstruksi saat beberapa log hilang bersamaan"); cantumkan RINENG sebagai "under review"; deskripsi dataset/metode dipersingkat dan ditulis ulang.
2. Jaga mekanisme MNAR (self-masking independen per log); jangan dikondisikan pada log lain.
3. R^2 rendah/jatuh tidak masalah; itu temuan jujur.
4. Cek template dan batas halaman conference IEEE yang dituju (sering 6 hal, ada biaya halaman tambahan).
5. Checklist bug: index mask tidak bergeser setelah reset_index (bug lama RINENG); inverse_transform sebelum 10**RS dan metrik RS di satuan asli; pooling dari residual mentah.

## 7. Status kode (folder multivariate_imputation/)
- File: litho_setup.py (config), part1_data.py, part2_masking.py (mnar_mask, positional indexing), part3_imputation.py (Mean/MICE/LGBM), part4_evaluation.py (tabel + parity plot), run_experiment.py (loop seed × well × metode).
- Sudah diuji end-to-end dengan data SINTETIS (bukan hasil riset). Belum dijalankan pada data asli.
- MICE: imputasi di-clip ke rentang training per kolom, karena tanpa itu RS meledak setelah 10** (ekstrapolasi di ruang log). Sebutkan di bagian metode.
- TODO:
  - [x] Set DATA_PATH dan nama kolom di litho_setup.py; jalankan pada data asli (preprocessing RINENG part1 sudah masuk part1_data.py).
  - [x] Samakan LGBM_PARAMS/early stopping dengan RINENG [FINAL]: n_estimators=5000, lr=0.05, num_leaves=31, min_child_samples=20, subsample=0.8 (tanpa subsample_freq -> bagging tidak aktif, sama seperti RINENG), colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1, max_depth=-1, early stopping 50, validation = train_test_split acak 0.2 (bukan per well). MICE max_iter=15.
  - [x] RD tidak masuk LOG10_COLS (pakai config lama) [FINAL].
  - [x] Cek mask_stats.csv (proporsi k) dan sanity check hasil per log -> part5_paper_assets.py (results/paper/).
  - [ ] Draft Introduction (EN), lalu Methodology, Results, Discussion.

## 8. Konvensi kerja
- Diskusi dalam Bahasa Indonesia; naskah paper dalam bahasa Inggris akademik.
- Edit seminimal dan sepresisi mungkin; output ringkas dan actionable; verifikasi dari first principles, bukan sekadar mengandalkan sitasi.
