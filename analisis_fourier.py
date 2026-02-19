import numpy as np
import matplotlib.pyplot as plt

# ===============================
# 1. Definición del dominio del tiempo
# ===============================

fs = 1000  # Frecuencia de muestreo (Hz)
t = np.linspace(-1, 1, fs)

# ===============================
# 2. Definición de señales
# ===============================

# Pulso rectangular
pulso = np.where(np.abs(t) <= 0.2, 1, 0)

# Función escalón
escalon = np.where(t >= 0, 1, 0)

# Señal senoidal
f = 5  # Frecuencia de la señal senoidal (Hz)
seno = np.sin(2 * np.pi * f * t)

# ===============================
# 3. Función para calcular la FFT
# ===============================

def calcular_fft(signal):
    fft_signal = np.fft.fft(signal)
    fft_shift = np.fft.fftshift(fft_signal)
    freqs = np.fft.fftshift(np.fft.fftfreq(len(signal), d=1/fs))
    return freqs, fft_shift

# Calcular FFT de cada señal
freq_p, fft_p = calcular_fft(pulso)
freq_e, fft_e = calcular_fft(escalon)
freq_s, fft_s = calcular_fft(seno)

# ===============================
# 4. Función para graficar
# ===============================

def graficar(signal, fft_signal, freqs, titulo):
    plt.figure(figsize=(12,6))
    
    # Dominio del tiempo
    plt.subplot(2,1,1)
    plt.plot(t, signal)
    plt.title("Dominio del tiempo - " + titulo)
    plt.xlabel("Tiempo")
    plt.ylabel("Amplitud")
    
    # Dominio de la frecuencia (magnitud)
    plt.subplot(2,1,2)
    plt.plot(freqs, np.abs(fft_signal))
    plt.title("Magnitud del espectro - " + titulo)
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Magnitud")
    
    plt.tight_layout()
    plt.show()

# ===============================
# 5. Graficar señales
# ===============================

graficar(pulso, fft_p, freq_p, "Pulso Rectangular")
graficar(escalon, fft_e, freq_e, "Función Escalón")
graficar(seno, fft_s, freq_s, "Señal Senoidal")

# ===============================
# 6. Verificación de linealidad
# ===============================

senal_lineal = pulso + seno
freq_l, fft_l = calcular_fft(senal_lineal)

graficar(senal_lineal, fft_l, freq_l, "Combinación Lineal (Pulso + Seno)")
