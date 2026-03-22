# Graficar datos del radar

import numpy as np
from scipy.signal import stft, windows
import matplotlib.pyplot as plt
import os

# -----------------------------
# 0) Configuración de usuario
# -----------------------------

# Ruta al archivo de datos (cambiar a tu ruta real)
DATA_FILE = '../datos/1P01A01R01.dat'   # poner el nombre real del archivo

# Parámetros del radar (ajustar según tu configuración)
Ns = 128          # muestras por chirp
Tc = 1e-3         # duración del chirp (s)
B = 400e6         # ancho de banda del chirp (Hz)
fc = 5.8e9        # frecuencia de la señal portadora (Hz)

# parámetros STFT para micro-Doppler (en chirps)
MD_NPERSEG = 128   # longitud de la ventana STFT 
MD_NOVERLAP = 96   # solapamiento entre ventanas 

# -----------------------------
# 1) Leer el archivo completo
# -----------------------------

def read_ascii_complex(path):
    """Leer líneas con formato '2200+1869i' y ponerlas en un arreglo numpy de números complejos."""
    vals = []
    with open(path, 'r') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            s = s.replace('i', 'j')  # Python usa 'j' para representar sqrt(-1)
            try:
                vals.append(complex(s))
            except ValueError:
                # Ignorar las líneas malformadas
                continue
    return np.array(vals, dtype=np.complex64)

# -----------------------------
# 2) Cargar y reformatear los datos en forma de [chirps, muestras]
# -----------------------------

iq = read_ascii_complex(DATA_FILE)

num_chirps = iq.size // Ns
if num_chirps == 0:
    raise RuntimeError('No hay suficientes muestras para un chirp; verificar Ns o el archivo.')

iq = iq[: num_chirps * Ns]
data = iq.reshape(num_chirps, Ns)  # [chirps, muestras por chirp]

print(f'Se cargaron {iq.size} muestras -> {num_chirps} chirps de {Ns} muestras cada uno')

# -----------------------------
# 3) FFT de rango para cada chirp
# -----------------------------

c = 3e8
S = B / Tc  # pendiente del chirp (Hz/s)

win_fast = windows.hann(Ns, sym=False)
range_fft = np.fft.fft(data * win_fast[None, :], axis=1)

# Solo tomamos la mitad positiva del espectro de rango (frecuencias positivas)
Nb = Ns // 2
range_fft = range_fft[:, :Nb]   # [chirps, Nb]

# Ejes de rango correspondientes a las frecuencias de la FFT
fb = np.fft.fftfreq(Ns, d=Tc / Ns)[:Nb]
R = c * fb / (2 * S)  # metros

# -----------------------------
# 4) Construir un mapa de rango-Doppler
# -----------------------------

# Número de chirps por frame para rango-Doppler
Nc = 256
if num_chirps < Nc:
    Nc = num_chirps

num_frames = num_chirps // Nc
range_fft_f = range_fft[: num_frames * Nc].reshape(num_frames, Nc, Nb)

win_slow = windows.hann(Nc, sym=False)

# FFT de Doppler a lo largo de los chirps
rdm = np.fft.fft(range_fft_f * win_slow[None, :, None], axis=1)
rdm = np.fft.fftshift(rdm, axes=1)  # centrar el eje de Doppler en cero

# Ejes de Doppler
lam = c / fc
fD = np.fft.fftfreq(Nc, d=Tc)
fD = np.fft.fftshift(fD)
vel = fD * lam / 2  # m/s

# Tomar el primer frame para visualizar el mapa de rango-Doppler
rdm0 = rdm[0]
rdm0_pow = 20 * np.log10(np.abs(rdm0) + 1e-6)

# -----------------------------
# 5) Espectrograma de micro-Doppler para un rango específico
# -----------------------------

# Escoger el bin de rango con máxima potencia promedio (torso aproximado)
range_power_mean = np.mean(np.abs(range_fft)**2, axis=0)
range_bin = int(np.argmax(range_power_mean))
print(f'Usando el bin de rango {range_bin} a aproximadamente R = {R[range_bin]:.2f} m para micro-Doppler')

slow_time_sig = range_fft[:, range_bin]

# STFT a lo largo de los chirps (eje slow-time para micro-Doppler)
win_md = windows.hann(MD_NPERSEG, sym=False)
f_md, t_md, Z_md = stft(
    slow_time_sig,
    fs=1.0 / Tc,            # chirps por segundo
    window=win_md,
    nperseg=MD_NPERSEG,
    noverlap=MD_NOVERLAP,
    return_onesided=False,
)

# Mapear frecuencia STFT a velocidad radial
v_md = f_md * lam / 2
md_pow = 20 * np.log10(np.abs(Z_md) + 1e-6)

# -----------------------------
# 6) Guardar los resultados en archivos de imagen
# -----------------------------

os.makedirs('output', exist_ok=True)

# Range-Doppler map
plt.figure(figsize=(6, 4))
plt.imshow(
    rdm0_pow.T,
    extent=[vel[0], vel[-1], R[0], R[-1]],
    aspect='auto',
    origin='lower',
    cmap='jet',
)
plt.colorbar(label='Potencia (dB)')
plt.xlabel('Velocidad radial (m/s)')
plt.ylabel('Rango (m)')
plt.title('Mapa Range-Doppler (primer frame)')
plt.tight_layout()
plt.savefig('output/range_doppler_map.png', dpi=200)
plt.close()

# Espectrograma de micro-Doppler
plt.figure(figsize=(6, 4))
plt.imshow(
    md_pow,
    extent=[t_md[0], t_md[-1], v_md[0], v_md[-1]],
    aspect='auto',
    origin='lower',
    cmap='jet',
)
plt.colorbar(label='Potencia (dB)')
plt.xlabel('Tiempo (s)')
plt.ylabel('Velocidad radial (m/s)')
plt.title(f'Micro-Doppler (contenedor para distancia {range_bin})')
plt.tight_layout()
plt.savefig('output/micro_doppler_spectrogram.png', dpi=200)
plt.close()

print('Guardados los plots a output/range_doppler_map.png and output/micro_doppler_spectrogram.png')