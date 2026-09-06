"""
Jaya Jaya Institut — Sistem Peringatan Dini Dua Titik Pantau
=============================================================
Prototype machine learning untuk memperkirakan risiko dropout mahasiswa pada dua
titik pantau: akhir semester 1 dan akhir semester 2.

Menjalankan secara lokal:
    streamlit run app.py
"""

import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Jaya Jaya Institut — Peringatan Dini Dropout",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------------------
# Gaya
# --------------------------------------------------------------------------------------
st.markdown(
    """
    <style>
      .block-container { padding-top: 1.8rem; max-width: 1300px; }
      .topbar {
          border-left: 6px solid #C1443C; background: rgba(128,128,128,.08);
          padding: 1rem 1.4rem; border-radius: 6px; margin-bottom: 1.4rem;
      }
      .topbar h1 { font-size: 1.5rem; margin: 0 0 .25rem; }
      .topbar p  { margin: 0; font-size: .9rem; opacity: .82; }
      .verdict {
          border-radius: 8px; padding: 1.4rem 1.6rem; margin: .4rem 0 1rem;
          border: 2px solid; background: rgba(128,128,128,.06);
      }
      .verdict .score { font-size: 2.6rem; font-weight: 800; line-height: 1; }
      .verdict .label { font-size: 1.05rem; font-weight: 700; letter-spacing: .04em; }
      .verdict .note  { font-size: .9rem; margin-top: .6rem; line-height: 1.55; }
      .v-high { border-color: #C1443C; }
      .v-mid  { border-color: #D89B2C; }
      .v-low  { border-color: #2E7D5B; }
      .tag {
          display: inline-block; padding: .22rem .6rem; margin: .15rem .25rem .15rem 0;
          border-radius: 4px; font-size: .78rem; border: 1px solid;
      }
      .tag-bad  { border-color: rgba(193,68,60,.5);  background: rgba(193,68,60,.12); }
      .tag-good { border-color: rgba(46,125,91,.5);  background: rgba(46,125,91,.12); }
      hr { margin: 1.2rem 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

BERKAS_MODEL = "model/model_dropout.joblib"

RISK, WARN, SAFE = "#C1443C", "#D89B2C", "#2E7D5B"


# --------------------------------------------------------------------------------------
# Kamus label (tampilan  ->  kode dataset)
# --------------------------------------------------------------------------------------
PRODI = {
    "Nursing": 9500, "Management": 9147, "Social Service": 9238,
    "Veterinary Nursing": 9085, "Journalism & Communication": 9773,
    "Advertising & Marketing": 9670, "Management (kelas malam)": 9991,
    "Tourism": 9254, "Communication Design": 9070, "Animation & Multimedia Design": 171,
    "Social Service (kelas malam)": 8014, "Agronomy": 9003, "Basic Education": 9853,
    "Informatics Engineering": 9119, "Equinculture": 9130, "Oral Hygiene": 9556,
    "Biofuel Production Technologies": 33,
}
PERNIKAHAN = {"Belum Menikah": 1, "Menikah": 2, "Duda/Janda": 3, "Bercerai": 4,
              "Facto Union": 5, "Pisah Ranjang": 6}
JALUR = {"Gelombang 1 — Umum": 1, "Gelombang 2 — Umum": 17, "Gelombang 3 — Umum": 18,
         "Usia di atas 23 tahun": 39, "Pindahan": 42, "Pindah Program Studi": 43,
         "Pindah Institusi": 51, "Pemegang Ijazah PT Lain": 7, "Mahasiswa Internasional": 15}
PENDIDIKAN = {"SMA / Sederajat": 1, "Sarjana (S1)": 2, "Diploma": 3, "Magister (S2)": 4,
              "Kelas 12 — Tidak Selesai": 9, "Kelas 11 — Tidak Selesai": 10,
              "Pendidikan Dasar": 19}
KELAS = {"Pagi (Daytime)": 1, "Malam (Evening)": 0}
JK = {"Perempuan": 0, "Laki-laki": 1}
YA_TIDAK = {"Ya": 1, "Tidak": 0}
SPP = {"Lunas / tepat waktu": 1, "Menunggak": 0}


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Memuat kedua model checkpoint...")
def muat_artefak():
    if not os.path.exists(BERKAS_MODEL):
        return None
    return joblib.load(BERKAS_MODEL)


artefak = muat_artefak()
if artefak is None:
    st.error(
        f"Berkas model tidak ditemukan di `{BERKAS_MODEL}`. "
        "Jalankan seluruh sel `notebook.ipynb` terlebih dahulu untuk melatih dan menyimpan model."
    )
    st.stop()

CP = artefak["checkpoints"]
NOMINAL = artefak["fitur_nominal"]
BIAYA_DEFAULT = artefak["asumsi_biaya"]


def rasio(lulus, diambil):
    return lulus / diambil if diambil > 0 else 0.0


def bangun_baris(nilai: dict, checkpoint: int) -> pd.DataFrame:
    """Menyusun satu baris fitur lengkap, termasuk fitur turunan.
    Harus identik dengan fungsi buat_fitur() pada notebook."""
    kolom = CP[checkpoint]["kolom"]
    baris = {k: CP[checkpoint]["default"][k] for k in kolom}
    baris.update({k: v for k, v in nilai.items() if k in baris})

    a1 = baris["Curricular_units_1st_sem_approved"]
    e1 = baris["Curricular_units_1st_sem_enrolled"]
    baris["rasio_sem1"] = rasio(a1, e1)
    baris["tekanan_finansial"] = ((1 - baris["Tuition_fees_up_to_date"])
                                  + baris["Debtor"] - baris["Scholarship_holder"])
    baris["selisih_evaluasi"] = baris["Curricular_units_1st_sem_evaluations"] - a1

    if checkpoint == 2:
        a2 = baris["Curricular_units_2nd_sem_approved"]
        e2 = baris["Curricular_units_2nd_sem_enrolled"]
        baris["rasio_sem2"] = rasio(a2, e2)
        baris["delta_rasio"] = baris["rasio_sem2"] - baris["rasio_sem1"]
        baris["delta_nilai"] = (baris["Curricular_units_2nd_sem_grade"]
                                - baris["Curricular_units_1st_sem_grade"])
        baris["total_lulus"] = a1 + a2

    return pd.DataFrame([baris])[kolom]


def skor(nilai: dict, checkpoint: int) -> float:
    return float(CP[checkpoint]["pipeline"].predict_proba(
        bangun_baris(nilai, checkpoint))[0, 1])


def bangun_batch(data: pd.DataFrame, checkpoint: int) -> pd.DataFrame:
    """Versi vektor dari bangun_baris untuk prediksi massal."""
    kolom = CP[checkpoint]["kolom"]
    d = data.copy()
    for k in kolom:
        if k not in d.columns:
            d[k] = CP[checkpoint]["default"][k]
    d = d[kolom].apply(pd.to_numeric, errors="coerce")
    d = d.fillna(pd.Series(CP[checkpoint]["default"]))

    a1, e1 = d["Curricular_units_1st_sem_approved"], d["Curricular_units_1st_sem_enrolled"]
    d["rasio_sem1"] = np.where(e1 > 0, a1 / e1, 0.0)
    d["tekanan_finansial"] = ((1 - d["Tuition_fees_up_to_date"])
                              + d["Debtor"] - d["Scholarship_holder"])
    d["selisih_evaluasi"] = d["Curricular_units_1st_sem_evaluations"] - a1
    if checkpoint == 2:
        a2, e2 = d["Curricular_units_2nd_sem_approved"], d["Curricular_units_2nd_sem_enrolled"]
        d["rasio_sem2"] = np.where(e2 > 0, a2 / e2, 0.0)
        d["delta_rasio"] = d["rasio_sem2"] - d["rasio_sem1"]
        d["delta_nilai"] = (d["Curricular_units_2nd_sem_grade"]
                            - d["Curricular_units_1st_sem_grade"])
        d["total_lulus"] = a1 + a2
    return d[kolom]


def rupiah(x) -> str:
    return f"Rp{x:,.0f}".replace(",", ".")


# --------------------------------------------------------------------------------------
# Sidebar — pengaturan sistem
# --------------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📡 Pengaturan Sistem")

    titik = st.radio(
        "Titik pantau",
        [1, 2],
        format_func=lambda c: f"Akhir Semester {c}",
        help="Semester 1 memakai data yang sudah tersedia setelah satu semester. "
             "Semester 2 lebih akurat, tetapi menyisakan waktu lebih sedikit untuk bertindak.",
    )

    m = CP[titik]["metrik"]
    st.caption(f"**Model semester {titik}** — {len(CP[titik]['kolom'])} fitur")
    k1, k2 = st.columns(2)
    k1.metric("Recall", f"{m['Recall']:.1%}",
              help="Bagian mahasiswa yang benar-benar dropout dan berhasil terdeteksi.")
    k2.metric("Precision", f"{m['Precision']:.1%}",
              help="Bagian mahasiswa bertanda risiko yang memang benar berisiko.")
    k3, k4 = st.columns(2)
    k3.metric("ROC-AUC", f"{m['ROC-AUC']:.3f}")
    k4.metric("Accuracy", f"{m['Accuracy']:.1%}")

    st.divider()
    st.markdown("#### 💰 Asumsi biaya")
    st.caption("Ambang batas peringatan dihitung ulang otomatis dari ketiga angka ini.")

    rugi = st.number_input("Kerugian per mahasiswa dropout (Rp)",
                           5_000_000, 200_000_000,
                           int(BIAYA_DEFAULT["rugi_dropout"]), step=5_000_000)
    biaya = st.number_input("Biaya satu paket intervensi (Rp)",
                            100_000, 20_000_000,
                            int(BIAYA_DEFAULT["biaya_intervensi"]), step=250_000)
    peluang = st.slider("Peluang intervensi berhasil", 0.05, 0.95,
                        float(BIAYA_DEFAULT["peluang_berhasil"]), 0.05)

    ambang_hitung = biaya / (peluang * rugi)
    ambang_model = CP[titik]["ambang"]
    ambang = float(np.clip(ambang_hitung, 0.02, 0.95))

    st.metric("Ambang batas peringatan", f"{ambang:.3f}",
              delta=f"{ambang - ambang_model:+.3f} vs kalibrasi awal ({ambang_model})")
    st.caption(
        "Rumus: biaya intervensi ÷ (peluang berhasil × kerugian). Menandai seorang "
        "mahasiswa layak dilakukan bila risikonya melampaui angka ini."
    )

st.markdown(
    """
    <div class="topbar">
      <h1>📡 Sistem Peringatan Dini Dropout — Jaya Jaya Institut</h1>
      <p>Dua titik pantau, satu keputusan: siapa yang perlu dihubungi lebih dulu, dan seberapa
      mendesak.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

mode = st.radio("Mode", ["🎯 Penilaian Individu", "📑 Pemindaian Angkatan", "📘 Cara Kerja Sistem"],
                horizontal=True, label_visibility="collapsed")

# --------------------------------------------------------------------------------------
# MODE 1 — Penilaian individu
# --------------------------------------------------------------------------------------
if mode.startswith("🎯"):
    kiri, kanan = st.columns([1.05, 1], gap="large")

    with kiri:
        st.markdown("##### Data mahasiswa")

        with st.expander("👤 Profil & pendaftaran", expanded=True):
            c1, c2 = st.columns(2)
            v_prodi = c1.selectbox("Program studi", list(PRODI))
            v_kelas = c2.selectbox("Waktu kuliah", list(KELAS))
            v_jk = c1.selectbox("Jenis kelamin", list(JK))
            v_usia = c2.number_input("Usia saat mendaftar", 16, 75, 20)
            v_nikah = c1.selectbox("Status pernikahan", list(PERNIKAHAN))
            v_jalur = c2.selectbox("Jalur pendaftaran", list(JALUR))
            v_pend = c1.selectbox("Pendidikan sebelumnya", list(PENDIDIKAN))
            v_nilai_masuk = c2.slider("Nilai masuk", 0.0, 200.0, 127.0, 0.5)
            v_rantau = c1.selectbox("Perantau", list(YA_TIDAK), index=1)
            v_intl = c2.selectbox("Mahasiswa internasional", list(YA_TIDAK), index=1)

        with st.expander("💰 Kondisi finansial", expanded=True):
            c1, c2, c3 = st.columns(3)
            v_spp = c1.selectbox("Status SPP", list(SPP))
            v_beasiswa = c2.selectbox("Penerima beasiswa", list(YA_TIDAK), index=1)
            v_debitur = c3.selectbox("Berstatus debitur", list(YA_TIDAK), index=1)

        with st.expander("📚 Semester 1", expanded=True):
            c1, c2 = st.columns(2)
            v_e1 = c1.number_input("SKS diambil", 0, 26, 6, key="e1")
            v_a1 = c2.number_input("SKS lulus", 0, 26, 5, key="a1")
            v_g1 = c1.slider("Nilai rata-rata (0–20)", 0.0, 20.0, 12.0, 0.1, key="g1")
            v_ev1 = c2.number_input("Jumlah evaluasi/ujian", 0, 45, 8, key="ev1")
            if v_e1:
                st.progress(min(v_a1 / v_e1, 1.0),
                            text=f"Rasio kelulusan semester 1: {v_a1 / v_e1:.0%}")

        if titik == 2:
            with st.expander("📗 Semester 2", expanded=True):
                c1, c2 = st.columns(2)
                v_e2 = c1.number_input("SKS diambil", 0, 26, 6, key="e2")
                v_a2 = c2.number_input("SKS lulus", 0, 26, 5, key="a2")
                v_g2 = c1.slider("Nilai rata-rata (0–20)", 0.0, 20.0, 12.0, 0.1, key="g2")
                v_ev2 = c2.number_input("Jumlah evaluasi/ujian", 0, 45, 8, key="ev2")
                if v_e2:
                    st.progress(min(v_a2 / v_e2, 1.0),
                                text=f"Rasio kelulusan semester 2: {v_a2 / v_e2:.0%}")
        else:
            v_e2 = v_a2 = v_ev2 = 0
            v_g2 = 0.0
            st.info(
                "Titik pantau **Semester 1** dipilih, sehingga data semester 2 tidak diminta — "
                "model ini sengaja dilatih tanpa informasi tersebut agar benar-benar dapat "
                "dipakai di akhir semester pertama."
            )

        if v_a1 > v_e1 or (titik == 2 and v_a2 > v_e2):
            st.warning("⚠️ SKS lulus melebihi SKS yang diambil — periksa kembali datanya.")

    nilai = {
        "Course": PRODI[v_prodi], "Daytime_evening_attendance": KELAS[v_kelas],
        "Gender": JK[v_jk], "Age_at_enrollment": v_usia,
        "Marital_status": PERNIKAHAN[v_nikah], "Application_mode": JALUR[v_jalur],
        "Previous_qualification": PENDIDIKAN[v_pend], "Admission_grade": v_nilai_masuk,
        "Displaced": YA_TIDAK[v_rantau], "International": YA_TIDAK[v_intl],
        "Tuition_fees_up_to_date": SPP[v_spp], "Scholarship_holder": YA_TIDAK[v_beasiswa],
        "Debtor": YA_TIDAK[v_debitur],
        "Curricular_units_1st_sem_enrolled": v_e1,
        "Curricular_units_1st_sem_approved": v_a1,
        "Curricular_units_1st_sem_grade": v_g1,
        "Curricular_units_1st_sem_evaluations": v_ev1,
        "Curricular_units_2nd_sem_enrolled": v_e2,
        "Curricular_units_2nd_sem_approved": v_a2,
        "Curricular_units_2nd_sem_grade": v_g2,
        "Curricular_units_2nd_sem_evaluations": v_ev2,
    }

    with kanan:
        st.markdown("##### Hasil penilaian")
        risiko = skor(nilai, titik)

        if risiko >= max(ambang * 3, 0.60):
            gaya, label, warna = "v-high", "RISIKO TINGGI", RISK
            saran = ("Jadwalkan <b>konseling akademik wajib dalam 7 hari</b>. Bila terdapat "
                     "tunggakan SPP, tawarkan skema cicilan pada sesi yang sama.")
        elif risiko >= ambang:
            gaya, label, warna = "v-mid", "PERLU DIPANTAU", WARN
            saran = ("Masukkan ke <b>daftar pantauan dosen wali</b>, kirim undangan tutoring, "
                     "dan nilai ulang pada akhir semester berjalan.")
        else:
            gaya, label, warna = "v-low", "RISIKO RENDAH", SAFE
            saran = "Tidak memerlukan penanganan khusus. Cukup <b>monitoring rutin per semester</b>."

        st.markdown(
            f"""
            <div class="verdict {gaya}">
              <div class="score" style="color:{warna}">{risiko:.1%}</div>
              <div class="label" style="color:{warna}">{label}</div>
              <div class="note">Ambang batas peringatan saat ini <b>{ambang:.1%}</b>.<br>{saran}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        rugi_harapan = risiko * rugi
        manfaat = risiko * peluang * rugi
        d1, d2 = st.columns(2)
        d1.metric("Ekspektasi kerugian", rupiah(rugi_harapan),
                  help="Probabilitas dropout dikalikan kerugian per mahasiswa.")
        d2.metric("Manfaat bersih intervensi", rupiah(manfaat - biaya),
                  delta="layak" if manfaat > biaya else "tidak layak",
                  delta_color="normal" if manfaat > biaya else "inverse",
                  help="Manfaat yang diharapkan dikurangi biaya intervensi.")

        # ---------------- Simulator what-if ----------------
        st.markdown("---")
        st.markdown("##### 🔧 Simulator: apa yang paling menurunkan risiko?")
        st.caption("Setiap baris menghitung ulang risiko bila satu kondisi diperbaiki.")

        skenario = []
        if nilai["Tuition_fees_up_to_date"] == 0:
            v = dict(nilai); v["Tuition_fees_up_to_date"] = 1
            skenario.append(("Tunggakan SPP dilunasi", v))
        if nilai["Debtor"] == 1:
            v = dict(nilai); v["Debtor"] = 0
            skenario.append(("Status debitur diselesaikan", v))
        if nilai["Scholarship_holder"] == 0:
            v = dict(nilai); v["Scholarship_holder"] = 1
            skenario.append(("Diberikan beasiswa", v))
        if v_e1 and v_a1 / v_e1 < 0.8:
            v = dict(nilai); v["Curricular_units_1st_sem_approved"] = int(np.ceil(v_e1 * 0.8))
            skenario.append(("Kelulusan SKS semester 1 naik ke 80%", v))
        if titik == 2 and v_e2 and v_a2 / v_e2 < 0.8:
            v = dict(nilai); v["Curricular_units_2nd_sem_approved"] = int(np.ceil(v_e2 * 0.8))
            skenario.append(("Kelulusan SKS semester 2 naik ke 80%", v))

        if skenario:
            hasil = []
            for nama, v in skenario:
                baru = skor(v, titik)
                hasil.append({"Perbaikan": nama, "Risiko baru": baru,
                              "Penurunan": risiko - baru})
            sim = pd.DataFrame(hasil).sort_values("Penurunan", ascending=False)
            st.dataframe(
                sim, hide_index=True, width="stretch",
                column_config={
                    "Risiko baru": st.column_config.ProgressColumn(
                        "Risiko setelah perbaikan", min_value=0.0, max_value=1.0, format="%.1%%"),
                    "Penurunan": st.column_config.NumberColumn("Penurunan risiko", format="%.1%%"),
                },
            )
            teratas = sim.iloc[0]
            if teratas["Penurunan"] > 0.01:
                st.success(
                    f"Tindakan paling berdampak: **{teratas['Perbaikan']}** — "
                    f"menurunkan risiko sebesar **{teratas['Penurunan']:.1%}**."
                )
        else:
            st.caption(
                "Tidak ada kondisi yang bisa disimulasikan: status finansial mahasiswa ini sudah "
                "baik dan rasio kelulusannya sudah di atas 80%."
            )

        with st.expander("Bandingkan dengan titik pantau yang lain"):
            lain = 2 if titik == 1 else 1
            try:
                skor_lain = skor(nilai, lain)
                b1, b2 = st.columns(2)
                b1.metric(f"Model Semester {titik}", f"{risiko:.1%}")
                b2.metric(f"Model Semester {lain}", f"{skor_lain:.1%}",
                          delta=f"{skor_lain - risiko:+.1%}")
                st.caption(
                    "Model semester 2 memakai data yang lebih lengkap sehingga skornya bisa "
                    "berbeda. Perbedaan besar menandakan kondisi mahasiswa berubah antar semester — "
                    "justru sinyal yang paling perlu ditindaklanjuti."
                    if titik == 1 else
                    "Skor semester 1 memperlihatkan apa yang sudah bisa diketahui institusi "
                    "satu semester lebih awal."
                )
            except Exception as exc:  # noqa: BLE001
                st.caption(f"Perbandingan tidak tersedia: {exc}")

# --------------------------------------------------------------------------------------
# MODE 2 — Pemindaian angkatan
# --------------------------------------------------------------------------------------
elif mode.startswith("📑"):
    st.markdown("##### Pemindaian satu angkatan dari berkas CSV")
    st.caption(
        f"Menggunakan model **titik pantau semester {titik}** dengan ambang batas "
        f"**{ambang:.3f}**. Ubah keduanya lewat panel di sebelah kiri."
    )

    with st.expander("Kolom yang dibutuhkan"):
        st.code(", ".join(c for c in CP[titik]["kolom"]
                          if c not in ("rasio_sem1", "rasio_sem2", "delta_rasio", "delta_nilai",
                                       "total_lulus", "tekanan_finansial", "selisih_evaluasi")),
                language=None)
        st.caption(
            "Fitur turunan dihitung otomatis oleh sistem. Kolom yang tidak ada akan diisi nilai "
            "umum populasi, dan kolom tambahan seperti `Status` diabaikan saat prediksi namun "
            "tetap tampil pada hasil."
        )

    berkas = st.file_uploader("Unggah berkas CSV", type=["csv"])

    if berkas is not None:
        try:
            baris_awal = berkas.getvalue().decode("utf-8-sig").split("\n")[0]
            pemisah = ";" if baris_awal.count(";") > baris_awal.count(",") else ","
            data = pd.read_csv(berkas, sep=pemisah)
            st.success(f"Terbaca **{len(data):,} baris** (pemisah `{pemisah}`).")

            if st.button("🔎 Jalankan pemindaian", type="primary", width="stretch"):
                with st.spinner("Menghitung skor risiko..."):
                    prob = CP[titik]["pipeline"].predict_proba(
                        bangun_batch(data, titik))[:, 1]
                    out = data.copy()
                    out["Skor_Risiko"] = prob.round(4)
                    out["Tindakan"] = np.where(
                        prob >= max(ambang * 3, 0.60), "Konseling wajib (7 hari)",
                        np.where(prob >= ambang, "Pantau + tawarkan tutoring", "Monitoring rutin"))
                    out = out.sort_values("Skor_Risiko", ascending=False)

                mendesak = int((prob >= max(ambang * 3, 0.60)).sum())
                pantau = int(((prob >= ambang) & (prob < max(ambang * 3, 0.60))).sum())
                aman = int((prob < ambang).sum())

                a, b, c, d = st.columns(4)
                a.metric("Total mahasiswa", f"{len(out):,}")
                b.metric("🔴 Konseling wajib", f"{mendesak:,}",
                         f"{mendesak / len(out):.1%} dari total")
                c.metric("🟠 Perlu dipantau", f"{pantau:,}")
                d.metric("🟢 Aman", f"{aman:,}")

                total_risiko = float(prob.sum() * rugi)
                e, f = st.columns(2)
                e.metric("Ekspektasi pendapatan berisiko", rupiah(total_risiko))
                f.metric("Perkiraan biaya intervensi",
                         rupiah((mendesak + pantau) * biaya),
                         help="Jumlah mahasiswa yang ditandai dikalikan biaya satu paket intervensi.")

                st.markdown("##### Daftar prioritas (urut dari risiko tertinggi)")
                st.dataframe(
                    out.head(200), width="stretch",
                    column_config={"Skor_Risiko": st.column_config.ProgressColumn(
                        "Skor risiko", min_value=0.0, max_value=1.0, format="%.2f")},
                )
                if len(out) > 200:
                    st.caption(f"Menampilkan 200 dari {len(out):,} baris — unduh untuk hasil lengkap.")

                st.download_button(
                    "⬇️ Unduh hasil pemindaian (CSV)",
                    out.to_csv(index=False).encode("utf-8"),
                    file_name=f"pemindaian_risiko_semester{titik}.csv",
                    mime="text/csv", width="stretch",
                )
        except Exception as exc:  # noqa: BLE001
            st.error(f"Gagal memproses berkas: {exc}")

# --------------------------------------------------------------------------------------
# MODE 3 — Dokumentasi
# --------------------------------------------------------------------------------------
else:
    st.markdown("##### Cara kerja sistem")

    st.markdown(
        f"""
Sistem ini memakai **dua model terpisah**, bukan satu model untuk semua keadaan. Alasannya
sederhana: institusi perlu bertindak sedini mungkin, tetapi data yang tersedia di akhir semester 1
berbeda dengan yang tersedia di akhir semester 2.

| | Model Semester 1 | Model Semester 2 |
|---|---|---|
| Jumlah fitur | {len(CP[1]['kolom'])} | {len(CP[2]['kolom'])} |
| Recall | **{CP[1]['metrik']['Recall']:.1%}** | **{CP[2]['metrik']['Recall']:.1%}** |
| Precision | {CP[1]['metrik']['Precision']:.1%} | {CP[2]['metrik']['Precision']:.1%} |
| Accuracy | {CP[1]['metrik']['Accuracy']:.1%} | {CP[2]['metrik']['Accuracy']:.1%} |
| ROC-AUC | {CP[1]['metrik']['ROC-AUC']:.3f} | {CP[2]['metrik']['ROC-AUC']:.3f} |
| Ambang kalibrasi awal | {CP[1]['ambang']} | {CP[2]['ambang']} |

Model semester 1 **sengaja tidak pernah melihat data semester 2** selama pelatihan, sehingga
angkanya jujur mencerminkan apa yang benar-benar bisa diketahui institusi pada saat itu.

**Kemampuan mendeteksi keduanya nyaris sama** ({CP[1]['metrik']['Recall']:.1%} vs
{CP[2]['metrik']['Recall']:.1%}). Yang berbeda adalah ketepatannya
({CP[1]['metrik']['Precision']:.1%} vs {CP[2]['metrik']['Precision']:.1%}). Karena itu keduanya
dipakai dengan peran berbeda:

- **Semester 1 — penyaringan luas.** Daftarnya lebih panjang dan mengandung lebih banyak alarm
  palsu, jadi tindak lanjutnya harus murah: undangan tutoring, pengecekan kondisi finansial,
  kontak singkat dari dosen wali.
- **Semester 2 — penanganan intensif.** Daftarnya jauh lebih bersih, sehingga layak diikuti
  tindakan yang mahal: konseling wajib, restrukturisasi SPP, kontrak akademik.

---

##### Dari mana ambang batas berasal

Ambang 0,50 hanya tepat bila kedua jenis kesalahan berbiaya sama — dan di sini jelas tidak.
Melewatkan mahasiswa yang akan *dropout* berarti kehilangan seluruh sisa pendapatannya, sedangkan
salah menandai mahasiswa aman hanya menghabiskan satu paket intervensi.

Ambang batas karenanya dihitung, bukan ditebak:

$$\\text{{ambang}} = \\frac{{\\text{{biaya intervensi}}}}{{\\text{{peluang berhasil}} \\times \\text{{kerugian per dropout}}}}$$

Dengan angka yang sedang aktif di panel kiri: {rupiah(biaya)} ÷ ({peluang:.0%} × {rupiah(rugi)})
= **{ambang_hitung:.3f}**.

Ubah ketiga angka tersebut sesuai kondisi keuangan institusi, dan seluruh sistem akan menyesuaikan
diri secara otomatis.

---

##### Algoritma dan data

- **Algoritma:** {artefak['algoritma']}
- **Data latih:** 3.630 mahasiswa berstatus final (*Dropout* dan *Graduate*). Mahasiswa berstatus
  *Enrolled* dikeluarkan dari pelatihan karena hasil akhirnya belum diketahui — merekalah justru
  sasaran sistem ini.
- **Fitur turunan:** rasio kelulusan SKS, selisih performa antar-semester, dan indeks tekanan
  finansial. Fitur rasio menempati peringkat pertama pada kedua model.
- **Preprocessing:** fitur kategorikal ditangani secara native oleh model tanpa One-Hot Encoding
  maupun penskalaan, seluruhnya menyatu dalam satu `Pipeline`.

##### Fitur paling berpengaruh — model semester {titik}
"""
    )
    for i, f in enumerate(CP[titik]["fitur_penting"][:8], 1):
        st.markdown(f"{i}. `{f}`")

    st.warning(
        "**Batasan penggunaan.** Model memberi peringkat prioritas, bukan vonis. Jangan "
        "menjadikannya dasar sanksi akademik atau keputusan administratif yang merugikan "
        "mahasiswa. Nilai kerugian, biaya intervensi, dan peluang keberhasilan bersifat asumsi "
        "dan wajib diganti dengan angka riil institusi. Latih ulang kedua model setiap tahun "
        "ajaran dengan data angkatan terbaru."
    )

st.markdown("---")
st.caption(
    "Proyek Akhir Dicoding — Menyelesaikan Permasalahan Perusahaan Edutech · Abyan Hisyam · "
    "Dashboard monitoring tersedia terpisah pada Metabase (lihat README.md)."
)
