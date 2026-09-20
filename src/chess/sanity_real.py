import os
import sys

# Aseguramos que busque la data en el SSD
os.environ["FLYPOKE_DATA"] = "/run/media/dolfius/Juegos/mosca_ajedrez_data/flypoke"

try:
    from flypoke.data import build_network
    from flypoke.sim import run, Stimulus, Params
except ImportError:
    print("flypoke no está instalado en el entorno.")
    sys.exit(1)

def run_real_feeding():
    print("Cargando conectoma FlyWire 783 (matriz dispersa firmada)...")
    net = build_network(min_syn=5)
    print(f"Red cargada: {net.n} neuronas.")
    
    sugar_idx = net.select("cell_sub_class=sugar/water")
    bitter_idx = net.select("cell_sub_class=bitter")
    mn9_l = net.select("cell_type=CB0700,side=left")
    mn9_r = net.select("cell_type=CB0700,side=right")
    
    print(f"Neuronas identificadas: {len(sugar_idx)} sugar, {len(bitter_idx)} bitter.")
    
    # Experimento 1: Azúcar
    print("1) Simulando Azúcar...")
    rec_sugar = run(net, [Stimulus(sugar_idx, 150.0)], Params(), n_trials=1)
    rate_mn9_sugar = rec_sugar.rates(net.n)[mn9_l[0]] if len(mn9_l) else 0
    
    # Experimento 2: Azúcar + Amargo
    print("2) Simulando Azúcar + Amargo...")
    rec_both = run(net, [Stimulus(sugar_idx, 150.0), Stimulus(bitter_idx, 150.0)], Params(), n_trials=1)
    rate_mn9_both = rec_both.rates(net.n)[mn9_l[0]] if len(mn9_l) else 0
    
    print("\n==================================")
    print("RESULTADOS SANIDAD (Conectoma Real)")
    print("==================================")
    print(f"MN9 Frecuencia (Solo Azúcar) : {rate_mn9_sugar:.2f} Hz")
    print(f"MN9 Frecuencia (Azúcar+Amargo): {rate_mn9_both:.2f} Hz")
    
    if rate_mn9_sugar > 100 and rate_mn9_both < rate_mn9_sugar * 0.5:
        print("¡Sanidad SUPERADA! El efecto inhibitorio biológico funciona correctamente.")
    else:
        print("FALLO: La inhibición no funcionó o las frecuencias están mal.")

if __name__ == "__main__":
    run_real_feeding()
