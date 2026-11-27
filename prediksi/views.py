
import joblib
import pandas as pd

from pathlib import Path
from django.conf import settings
from django.shortcuts import render


# Memuat model Random Forest
MODEL_PATH = Path(settings.BASE_DIR) / "model_random_forest.pkl"
model_data = joblib.load(MODEL_PATH)

model = model_data["model"]
features = model_data["features"]
medians = model_data["medians"]


def home(request):
    hasil = None
    probabilitas = None
    error = None

    # Mengisi nilai awal menggunakan median
    data_input = [
        {
            "name": fitur,
            "value": medians[fitur]
        }
        for fitur in features
    ]

    if request.method == "POST":
        try:
            nilai = {}

            for fitur in features:
                nilai[fitur] = float(request.POST[fitur])

            # Menyusun data sesuai urutan fitur model
            data_prediksi = pd.DataFrame(
                [nilai],
                columns=features
            )

            # Melakukan prediksi
            prediksi = int(model.predict(data_prediksi)[0])

            # Mengubah hasil menjadi keterangan
            hasil = "Benign (Jinak)" if prediksi == 0 else "Malignant (Ganas)"

            # Mengambil probabilitas kelas
            peluang = model.predict_proba(data_prediksi)[0]
            indeks = list(model.classes_).index(prediksi)
            probabilitas = round(float(peluang[indeks]) * 100, 2)

            # Menampilkan kembali nilai yang dimasukkan
            data_input = [
                {
                    "name": fitur,
                    "value": nilai[fitur]
                }
                for fitur in features
            ]

        except (ValueError, KeyError, TypeError):
            error = "Masukkan semua nilai fitur dengan angka yang valid."

    return render(request, "prediksi/home.html", {
        "data_input": data_input,
        "hasil": hasil,
        "probabilitas": probabilitas,
        "error": error
    })
