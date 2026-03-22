# Calcular características para detección de caídas usando datos de radar FMCW

import numpy as np

def fall_features_from_md(md_pow, f_doppler, t_vec):
    # md_pow: [T, F], potencia vs tiempo vs Doppler
    # f_doppler: [F], Doppler (Hz) o m/s
    # t_vec: [T], vector de marcas de tiempo en segundos

    # 1) Seleccionar una banda alta de Doppler de interés (e.g. 25–50 Hz, o equivalente en m/s)
    fmin, fmax = 25.0, 50.0   # ejemplo en Hz; ajustar según la configuración del radar y la velocidad de caída esperada
    band = (np.abs(f_doppler) >= fmin) & (np.abs(f_doppler) <= fmax)
    md_band = md_pow[:, band]          # [T, F_band]

    # Curva de brote de energía EB(t) = suma de Doppler en esa banda 
    EB = md_band.sum(axis=1)           # [T]

    EB_max = EB.max()
    t_max = t_vec[EB.argmax()]         # tiempo del brote máximo

    # Agudeza de brote normalizada: pico / media
    EB_mean = EB.mean() + 1e-12
    burst_sharpness = EB_max / EB_mean

    # Duración por encima del umbral (e.g. 50% del máximo) alrededor de t_max
    thr = 0.5 * EB_max
    above = EB >= thr
    # región contigua alrededor de t_max
    idx_max = EB.argmax()
    left = idx_max
    while left > 0 and above[left-1]:
        left -= 1
    right = idx_max
    while right < len(EB)-1 and above[right+1]:
        right += 1
    burst_duration = t_vec[right] - t_vec[left]

    # 2) Quietud post-brote: energía en banda baja de Doppler (cerca de 0 Hz) después de t_max
    low_band = (np.abs(f_doppler) <= fmin)      # Doppler bajo, cerca de 0 Hz
    md_low = md_pow[:, low_band]

    # Definir ventana post-brote: desde t_max a t_max + 1.0 s (ajustar según la duración típica de la caída)
    post_end = t_max + 1.0
    post_mask = (t_vec >= t_max) & (t_vec <= post_end)
    if post_mask.sum() > 0:
        post_energy = md_low[post_mask].mean()
    else:
        post_energy = md_low[-1].mean()

    # Energía pre-evento en la misma banda de bajo Doppler para comparación
    pre_start = max(0.0, t_max - 1.0)
    pre_mask = (t_vec >= pre_start) & (t_vec < t_max)
    if pre_mask.sum() > 0:
        pre_energy = md_low[pre_mask].mean()
    else:
        pre_energy = md_low[0].mean()

    # Tasa de quietud: energía post-brote en Doppler bajo vs energía pre-brote
    stillness_ratio = post_energy / (pre_energy + 1e-12)

    # 3) Dispersión Doppler en el pico: ancho de banda de Doppler con energía significativa en el tiempo del brote
    md_peak = md_pow[idx_max]           # [F]
    md_norm = md_peak / (md_peak.max() + 1e-12)
    # indices encima de -10 dB (~0.316)
    mask_spread = md_norm >= 0.316
    if mask_spread.any():
        f_spread = np.max(np.abs(f_doppler[mask_spread]))  # max |f| con energía significativa
    else:
        f_spread = 0.0

    features = {
        "EB_max": float(EB_max),
        "burst_sharpness": float(burst_sharpness),
        "burst_duration": float(burst_duration),
        "t_max": float(t_max),
        "stillness_ratio": float(stillness_ratio),
        "doppler_spread_at_peak": float(f_spread),
    }
    return features, EB