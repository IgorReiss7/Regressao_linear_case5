from flask import Flask, jsonify, render_template
import numpy as np

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/modelo")
def modelo():

    # ===================== DATASET (40 registros) =====================
    # Variáveis: itens, tipo (1=simples,2=médio,3=complexo), cozinheiros, horario (1=vazio,2=normal,3=pico)
    qtdeItens      = [1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13,14,14,15,15,16,16,17,17,18,18,19,19,20,20]
    qtdeTipo       = [1,2,1,3,2,1,3,2,1,3,2,1,3,2,1,3,2,1,3,2,1,3,2,1,3,2,1,3,2,1,3,2,1,3,2,1,3,2,1,3]
    qtdeCozinheiros= [3,2,4,1,3,2,4,1,3,2,4,1,3,2,4,1,3,2,4,1,3,2,4,1,3,2,4,1,3,2,4,1,3,2,4,1,3,2,4,1]
    qtdeHorario    = [1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1]
    qtdeTempo      = [
        11.5, 14.5, 15.5, 20.5, 19.5, 16.5, 25.5, 20.5, 17.5, 28.5,
        23.5, 19.5, 31.5, 26.5, 21.5, 34.5, 29.5, 23.5, 37.5, 30.5,
        31.5, 38.5, 35.5, 32.5, 43.5, 38.5, 33.5, 46.5, 41.5, 35.5,
        49.5, 44.5, 37.5, 52.5, 47.5, 39.5, 55.5, 50.5, 41.5, 44.5
    ]

    n = len(qtdeTempo)

    # ===================== REGRESSÃO LINEAR SIMPLES (1 variável) =====================
    mediaX = np.mean(qtdeItens)
    mediaY = np.mean(qtdeTempo)

    numerador   = sum((qtdeItens[i] - mediaX) * (qtdeTempo[i] - mediaY) for i in range(n))
    denominador = sum((qtdeItens[i] - mediaX) ** 2 for i in range(n))

    B1_simples = numerador / denominador
    B0_simples = mediaY - B1_simples * mediaX

    # Comparação polyfit (simples)
    poly = np.polyfit(qtdeItens, qtdeTempo, 1)

    # ===================== REGRESSÃO LINEAR MÚLTIPLA MANUAL =====================
    # Modelo: Y = B0 + B1*itens + B2*tipo + B3*cozinheiros + B4*horario
    # Forma matricial: Y = X * B  =>  B = (X^T X)^-1 X^T Y

    # Monta matriz X com coluna de 1s (intercepto) + 4 features
    X = np.column_stack([
        np.ones(n),        # B0 (intercepto)
        qtdeItens,         # B1
        qtdeTipo,          # B2
        qtdeCozinheiros,   # B3
        qtdeHorario        # B4
    ])
    Y = np.array(qtdeTempo)

    # Cálculo manual: B = (X^T * X)^-1 * X^T * Y
    XtX    = X.T @ X
    XtY    = X.T @ Y
    B_manual = np.linalg.inv(XtX) @ XtY

    # ===================== COMPARAÇÃO COM np.linalg.lstsq =====================
    B_lstsq, residuals, rank, sv = np.linalg.lstsq(X, Y, rcond=None)

    return jsonify({
        # --- Regressão simples ---
        "simples_manual_B0": round(B0_simples, 4),
        "simples_manual_B1": round(B1_simples, 4),
        "simples_polyfit_B0": round(float(poly[1]), 4),
        "simples_polyfit_B1": round(float(poly[0]), 4),


        #PARTE IGOR
        # --- Regressão múltipla manual ---
        "multipla_manual_B0": round(float(B_manual[0]), 4),  #aqui é o tempo base
        "multipla_manual_B1_itens": round(float(B_manual[1]), 4),
        "multipla_manual_B2_tipo": round(float(B_manual[2]), 4),
        "multipla_manual_B3_cozinheiros": round(float(B_manual[3]), 4),
        "multipla_manual_B4_horario": round(float(B_manual[4]), 4),
            #formula B = (XᵀX)⁻¹ Xᵀy

        # --- Comparação com lstsq ---
        "lstsq_B0": round(float(B_lstsq[0]), 4),
        "lstsq_B1_itens": round(float(B_lstsq[1]), 4),
        "lstsq_B2_tipo": round(float(B_lstsq[2]), 4),
        "lstsq_B3_cozinheiros": round(float(B_lstsq[3]), 4),
        "lstsq_B4_horario": round(float(B_lstsq[4]), 4),

        #Residuals (erro quadrático total do lstsq)
        "residuals": round(float(residuals[0]), 4) if len(residuals) > 0 else 0,

        #Coeficientes usados pelo frontend (múltipla manual)
        "tempoBase": round(float(B_manual[0]), 4),
        "valorItem": round(float(B_manual[1]), 4),
        "valorTipo": round(float(B_manual[2]), 4),
        "qtdeCozinheiros": round(float(B_manual[3]), 4),
        "Horario": round(float(B_manual[4]), 4),
    })

if __name__ == "__main__":
    app.run(debug=True)
