import joblib
import numpy as np
import pandas as pd
import argparse
import os

# ==================================================
# Load Model Terbaik
# ==================================================
MODEL_PATH = "models/rf_ga_best_model.joblib"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model tidak ditemukan di: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

# Urutan fitur harus sama persis dengan saat training
FEATURE_ORDER = [
    "Age", "Gls", "Ast", "xG", "xA", "SCA", "PrgP",
    "Pos_DF,FW", "Pos_DF,MF", "Pos_FW", "Pos_FW,DF",
    "Pos_FW,MF", "Pos_MF", "Pos_MF,DF", "Pos_MF,FW",
    "Comp_eng Premier League", "Comp_es La Liga",
    "Comp_fr Ligue 1", "Comp_it Serie A"
]


def predict_market_value(data: dict) -> float:
    """
    Memprediksi nilai pasar pemain dalam Euro.
    
    Parameters
    ----------
    data : dict
        Dictionary berisi nilai fitur pemain.
        
    Returns
    -------
    float
        Estimasi nilai pasar dalam Euro.
    """
    # Buat DataFrame dengan urutan kolom yang benar
    df = pd.DataFrame([data], columns=FEATURE_ORDER)
    
    # Prediksi dalam skala log
    pred_log = model.predict(df)[0]
    
    # Inverse transform ke skala Euro
    pred_euro = np.expm1(pred_log)
    
    return pred_euro


def main():
    parser = argparse.ArgumentParser(
        description="Prediksi Nilai Pasar Pemain Sepak Bola menggunakan Random Forest + GA"
    )
    
    # Fitur numerik
    parser.add_argument("--age", type=float, required=True, help="Usia pemain")
    parser.add_argument("--gls", type=float, default=0, help="Jumlah gol")
    parser.add_argument("--ast", type=float, default=0, help="Jumlah assist")
    parser.add_argument("--xg", type=float, default=0, help="Expected Goals")
    parser.add_argument("--xa", type=float, default=0, help="Expected Assists")
    parser.add_argument("--sca", type=float, default=0, help="Shot-Creating Actions")
    parser.add_argument("--prgp", type=float, default=0, help="Progressive Passes")
    
    # Posisi (pilih salah satu)
    parser.add_argument("--pos", type=str, default="MF",
                        choices=["DF", "MF", "FW", "DF,MF", "MF,DF", "FW,MF", "MF,FW", "DF,FW", "FW,DF"],
                        help="Posisi pemain")
    
    # Liga
    parser.add_argument("--liga", type=str, default="other",
                        choices=["Premier League", "La Liga", "Serie A", "Ligue 1", "other"],
                        help="Liga tempat pemain bermain")
    
    args = parser.parse_args()
    
    # One-hot encoding posisi
    pos_features = {
        "Pos_DF,FW": 0, "Pos_DF,MF": 0, "Pos_FW": 0, "Pos_FW,DF": 0,
        "Pos_FW,MF": 0, "Pos_MF": 0, "Pos_MF,DF": 0, "Pos_MF,FW": 0
    }
    pos_map = {
        "DF,FW": "Pos_DF,FW", "DF,MF": "Pos_DF,MF", "FW": "Pos_FW",
        "FW,DF": "Pos_FW,DF", "FW,MF": "Pos_FW,MF", "MF": "Pos_MF",
        "MF,DF": "Pos_MF,DF", "MF,FW": "Pos_MF,FW"
    }
    if args.pos in pos_map:
        pos_features[pos_map[args.pos]] = 1
    
    # One-hot encoding liga
    liga_features = {
        "Comp_eng Premier League": 0,
        "Comp_es La Liga": 0,
        "Comp_fr Ligue 1": 0,
        "Comp_it Serie A": 0
    }
    liga_map = {
        "Premier League": "Comp_eng Premier League",
        "La Liga": "Comp_es La Liga",
        "Ligue 1": "Comp_fr Ligue 1",
        "Serie A": "Comp_it Serie A"
    }
    if args.liga in liga_map:
        liga_features[liga_map[args.liga]] = 1
    
    # Gabungkan semua fitur
    input_data = {
        "Age": args.age,
        "Gls": args.gls,
        "Ast": args.ast,
        "xG": args.xg,
        "xA": args.xa,
        "SCA": args.sca,
        "PrgP": args.prgp,
        **pos_features,
        **liga_features
    }
    
    # Prediksi
    hasil = predict_market_value(input_data)
    
    print("\n" + "="*50)
    print("       HASIL PREDIKSI NILAI PASAR PEMAIN")
    print("="*50)
    print(f"Usia              : {args.age}")
    print(f"Posisi            : {args.pos}")
    print(f"Liga              : {args.liga}")
    print(f"SCA               : {args.sca}")
    print(f"Progressive Pass  : {args.prgp}")
    print("-"*50)
    print(f"Estimasi Nilai Pasar : € {hasil:,.0f}")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()