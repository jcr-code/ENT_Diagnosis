from random import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from RandomForest import RandomForest
from sklearn.preprocessing import LabelEncoder
from functions import *
import matplotlib.pyplot as plt
import pickle

# ----------------------------------------------------------------------------
# Fungsi Pelatihan melalui Handling Data
# ----------------------------------------------------------------------------

def handle_form_data(form_data):
    suhu = int(form_data.get('suhu'))
    hidung_tersumbat = int(form_data.get('hidung_tersumbat'))
    pilek = int(form_data.get('hidung_pilek'))
    suara_serak = int(form_data.get('tenggorokan_suara_serak'))
    nyeri_membuka_mulut = int(form_data.get('tenggorokan_membuka_mulut'))
    nyeri_kepala = int(form_data.get('kepala_nyeri'))
    vertigo = int(form_data.get('vertigo'))
    hidung_nyeri = int(form_data.get('hidung_nyeri'))
    belakang_hidung_ganjal = int(form_data.get('hidung_belakang_ganjal'))
    nyeri_tenggorokan = int(form_data.get('tenggorokan_nyeri'))
    batuk = int(form_data.get('tenggorok_batuk'))
    nyeri_telinga = int(form_data.get('telinga_nyeri'))
    gangguan_dengar = int(form_data.get('telinga_gangguan_pendengaran'))
    cairan_telinga = int(form_data.get('cairan_telinga'))
    leher_bengkak = int(form_data.get('tenggorokan_bengkak'))
    mata_gatal = int(form_data.get('mata_gatal'))
    telinga_kemerahan = int(form_data.get('telinga_merah'))
    hidung_kemerahan = int(form_data.get('hidung_merah'))
    telinga_bengkak = int(form_data.get('telinga_bengkak'))
    hidung_bengkak = int(form_data.get('hidung_bengkak'))
    telinga_mendengung = int(form_data.get('telinga_mendengung'))
    telinga_gatal = int(form_data.get('telinga_gatal'))
    keringat_dingin = int(form_data.get('keringat_dingin'))
    tenggorokan_kering = int(form_data.get('tenggorokan_kering'))
    tenggorokan_gatal = int(form_data.get('tenggorokan_gatal'))
    kepala_berat = int(form_data.get('kepala_berat'))
    telinga_berat = int(form_data.get('telinga_berat'))
    bersin = int(form_data.get('hidung_bersin'))
    gendang_telinga_lubang = int(form_data.get('telinga_gendang_lubang'))
    telinga_berair_kemasukan_air = int(form_data.get('telinga_berair'))
    telinga_penuh_tertutup_tersumbat = int(form_data.get('telinga_penuh'))
    pusing = int(form_data.get('pusing'))
    mimisan = int(form_data.get('hidung_mimisan'))
    tenggorokan_ganjal = int(form_data.get('tenggorokan_ganjal'))
    tenggorokan_panas = int(form_data.get('tenggorokan_panas'))
    hidung_keluar_ingus = int(form_data.get('hidung_ingusan'))
    kekentalan_ingus = int(form_data.get('hidung_kekentalan_ingus'))
    sesak_nafas = int(form_data.get('sesak_nafas'))
    bersendawa = int(form_data.get('bersendawa'))
    berdehem = int(form_data.get('tenggorokan_berdehem'))
    kembung = int(form_data.get('kembung'))
    mulut_pahit = int(form_data.get('tenggorokan_mulut_pahit'))
    mulut_bau = int(form_data.get('tenggorokan_mulut_bau'))
    mulut_kering = int(form_data.get('tenggorokan_mulut_kering'))
    pandangan = int(form_data.get('pandangan'))
    mata_juling = int(form_data.get('mata_juling'))
    nafsu_makan = int(form_data.get('nafsu_makan'))
    pipi_bengkak = int(form_data.get('tenggorokan_pipi_bengkak'))
    pipi_nyeri = int(form_data.get('tenggorokan_pipi_nyeri'))
    badan_lemas = int(form_data.get('lemas'))
    berdat_badan = int(form_data.get('berat'))
    usia = int(form_data.get('usia'))
    mual = int(form_data.get('mual'))
    muntah = int(form_data.get('muntah'))
    telinga = int(form_data.get('pemeriksaan_telinga'))
    hidung = int(form_data.get('pemeriksaan_hidung'))
    tenggorokan = int(form_data.get('pemeriksaan_tenggorokan'))
    leher = int(form_data.get('pemeriksaan_leher'))
    input = np.array([suhu, hidung_tersumbat, pilek, suara_serak, nyeri_membuka_mulut, nyeri_kepala, vertigo, hidung_nyeri, 
                    belakang_hidung_ganjal, nyeri_tenggorokan, batuk, nyeri_telinga, gangguan_dengar, cairan_telinga, leher_bengkak, 
                    mata_gatal, telinga_kemerahan, hidung_kemerahan, telinga_bengkak, hidung_bengkak, telinga_mendengung, telinga_gatal, 
                    keringat_dingin, tenggorokan_kering, tenggorokan_gatal, kepala_berat, telinga_berat, bersin, gendang_telinga_lubang,
                    telinga_berair_kemasukan_air, telinga_penuh_tertutup_tersumbat, pusing, mimisan, tenggorokan_ganjal, tenggorokan_panas,
                    hidung_keluar_ingus, kekentalan_ingus, sesak_nafas, bersendawa, berdehem, kembung, mulut_pahit, mulut_bau, mulut_kering,
                    pandangan, mata_juling, nafsu_makan, pipi_bengkak, pipi_nyeri, badan_lemas, berdat_badan, usia, mual, muntah, telinga, 
                    hidung, tenggorokan, leher])

    print(input)
    # if np.all(input == 0):
    #     voteBox = [{'key': "No Prediction", 'value': 0.0}]
    #     print(voteBox)
    #     return voteBox
    # Load your dataset from the Excel file
    # file_path = "../Progress_Proposal_THT/data_THT_transform_tanpa_mimisanFrek.xlsx" #non manipulative
    file_path = "../Progress_Proposal_THT/manipulasi_data_tht_transform.xlsx" #manipulative
    # file_path = "../Progress_Proposal_THT/tht_transform_featureranking_sfs.xlsx" #SFS
    # file_path = "../Progress_Proposal_THT/tht_transform_featureranking_sbs.xlsx" #SBS
    # file_path = "../Progress_Proposal_THT/tht_transform_featureranking_sklearn.xlsx" #feat importance
    # file_path = "../Progress_Proposal_THT/tanpa_korpus_alenium.xlsx" #non korpus alenium
    df = pd.read_excel(file_path)

    # Drop rows with missing values
    df.dropna(axis=0, inplace=True)

    # Initialize LabelEncoder
    label_encoder = LabelEncoder()

    # Encode the target labels into numeric values
    df['hasil_diagn_encoded'] = label_encoder.fit_transform(df['hasil_diagn'])

    # Check the mapping between original labels and encoded values
    label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

    # hasil reverse value dengan key terhadap label_mapping
    reverse_label_mapping = {v: k for k, v in label_mapping.items()}

    # print("Label Mapping:", label_mapping)

    # Now you have a new column 'hasil_diagn_encoded' containing numeric representations of the labels
    # You can drop the original 'hasil_diagn' column if you don't need it anymore
    df.drop(columns=['hasil_diagn'], inplace=True)
    totalTree = 1200
    # Load the trained model
    with open('random_forest_model_baseline.pkl', 'rb') as f:
        loaded_model = pickle.load(f)

    X_test_filt = np.array([input])
    predictions = loaded_model.predict(X_test_filt)
    
    # ----------------------------------------------------------------------------
    # hasl voting prediksi diubah ke label asli
    # ----------------------------------------------------------------------------
    predictedLabel = label_encoder.inverse_transform(predictions)
    print(predictedLabel)
    
    
    # ----------------------------------------------------------------------------
    # hasil seluruh vote
    # ----------------------------------------------------------------------------
    vote = loaded_model.voteResult(X_test_filt)
    # Flatten the array
    flattened_array = [item for sublist in vote for item in sublist]
    # Count occurrences of each number
    number_counts = {}
    for num in flattened_array:
        if num in number_counts:
            number_counts[num] += 1
        else:
            number_counts[num] = 1

    voteBox = []
    for key, value in number_counts.items():
        percentages = []
        voteLabels = label_encoder.inverse_transform([int(key)]).tolist()[0]
        percentages.append(voteLabels)
        percent = (value / totalTree) * 100
        percentages.append(percent)
        voteBox.append(percentages)
    
    sorted_voteBox = sorted(voteBox, key=lambda x: x[1], reverse=True)
    print(sorted_voteBox)
    return sorted_voteBox
    


