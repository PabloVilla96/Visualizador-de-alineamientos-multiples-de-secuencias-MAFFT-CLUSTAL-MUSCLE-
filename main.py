# Este código fue creado con ayuda de la IA Claude de Anthropic.
# El autor que usó dicha es Juan Pablo Villa Núñez, el 02 de noviembre de 2026
# Este código es un visualizador de alineamientos múltiples de secuencias de DNA/RNA.
# La función del algoritmo es graficar el alineamiento múltiple utilizando una secuencia como referencia para señalar
# sus regiones génicas por arriba del alineamiento. Y el alineamiento es graficado como un heatmap.
# Además, debajo del heatmap se genera una gráfica de identidad del alineamiento, calculada con una ventana de 50 columnas.

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator
from Bio import SeqIO

# ==========================================
# 0. CONFIGURACIÓN
# ==========================================
output_dir = r"D:\Maestría_DIC2025\Figuras_Tesis"
os.makedirs(output_dir, exist_ok=True)

gen_objetivo = 'CgiPLT4'  # <-- CAMBIA ESTO PARA CADA GEN
archivo_alineamiento = os.path.expanduser("~/Downloads/MAFFT_PLT4_CarGig.aln-fasta")  # <-- Y ESTO

orden_deseado_especies = ['Cgi', 'Ppr', 'Lsc', 'Sth', 'Hun']

# Ventana (en columnas del alineamiento) para suavizar la curva inferior.
# El heatmap NO se suaviza. Pon 1 para ver el valor crudo de cada columna.
ventana_curva = 50

# Mínimo de especies que deben compartir una base para contarla como
# coincidencia. Con 2, una base que ninguna otra especie comparte vale 0 %.
min_coinciden = 2

# ==========================================
# 1. DICCIONARIO CENTRAL DE COORDENADAS Cgi
#    (mismas longitudes y colores que el script de mVISTA)
# ==========================================
c_prom, c_utr5, c_exon, c_intron, c_utr3 = '#ff9999', '#99ff99', '#80bfff', '#d9d9d9', '#df80ff'

estructuras_cgi = {
    'CgiPLT1/2a': [
        ('Promotor', 2000, c_prom), ("5' UTR", 442, c_utr5),
        ('Exón', 149, c_exon), ('Intrón', 267, c_intron),
        ('Exón', 441, c_exon), ('Intrón', 102, c_intron),
        ('Exón', 84, c_exon), ('Intrón', 147, c_intron),
        ('Exón', 10, c_exon), ('Intrón', 110, c_intron),
        ('Exón', 90, c_exon), ('Intrón', 117, c_intron),
        ('Exón', 72, c_exon), ('Intrón', 85, c_intron),
        ('Exón', 130, c_exon), ('Intrón', 500, c_intron),
        ('Exón', 785, c_exon), ("3' UTR", 87, c_utr3)
    ],
    'CgiPLT1/2b': [
        ('Promotor', 2000, c_prom), ("5' UTR", 395, c_utr5),
        ('Exón', 158, c_exon), ('Intrón', 174, c_intron),
        ('Exón', 502, c_exon), ('Intrón', 115, c_intron),
        ('Exón', 82, c_exon), ('Intrón', 254, c_intron),
        ('Exón', 11, c_exon), ('Intrón', 128, c_intron),
        ('Exón', 90, c_exon), ('Intrón', 101, c_intron),
        ('Exón', 72, c_exon), ('Intrón', 102, c_intron),
        ('Exón', 130, c_exon), ('Intrón', 117, c_intron),
        ('Exón', 713, c_exon), ("3' UTR", 433, c_utr3)
    ],
    'CgiPLT3/7a': [
        ('Promotor', 2000, c_prom), ("5' UTR", 424, c_utr5),
        ('Exón', 180, c_exon), ('Intrón', 96, c_intron),
        ('Exón', 512, c_exon), ('Intrón', 106, c_intron),
        ('Exón', 84, c_exon), ('Intrón', 1390, c_intron),
        ('Exón', 10, c_exon), ('Intrón', 119, c_intron),
        ('Exón', 90, c_exon), ('Intrón', 119, c_intron),
        ('Exón', 72, c_exon), ('Intrón', 88, c_intron),
        ('Exón', 51, c_exon), ('Intrón', 110, c_intron),
        ('Exón', 78, c_exon), ('Intrón', 211, c_intron),
        ('Exón', 459, c_exon), ("3' UTR", 282, c_utr3)
    ],
    'CgiPLT3/7b': [
        ('Promotor', 2000, c_prom), ("5' UTR", 492, c_utr5),
        ('Exón', 177, c_exon), ('Intrón', 107, c_intron),
        ('Exón', 549, c_exon), ('Intrón', 136, c_intron),
        ('Exón', 85, c_exon), ('Intrón', 1683, c_intron),
        ('Exón', 7, c_exon), ('Intrón', 273, c_intron),
        ('Exón', 91, c_exon), ('Intrón', 123, c_intron),
        ('Exón', 72, c_exon), ('Intrón', 89, c_intron),
        ('Exón', 51, c_exon), ('Intrón', 155, c_intron),
        ('Exón', 78, c_exon), ('Intrón', 108, c_intron),
        ('Exón', 441, c_exon), ("3' UTR", 251, c_utr3)
    ],
    'CgiPLT4': [
        ('Promotor', 2000, c_prom), ("5' UTR", 229, c_utr5),
        ('Exón', 207, c_exon), ('Intrón', 253, c_intron),
        ('Exón', 630, c_exon), ('Intrón', 182, c_intron),
        ('Exón', 84, c_exon), ('Intrón', 88, c_intron),
        ('Exón', 9, c_exon), ('Intrón', 325, c_intron),
        ('Exón', 89, c_exon), ('Intrón', 745, c_intron),
        ('Exón', 73, c_exon), ('Intrón', 95, c_intron),
        ('Exón', 51, c_exon), ('Intrón', 1475, c_intron),
        ('Exón', 79, c_exon), ('Intrón', 96, c_intron),
        ('Exón', 842, c_exon), ("3' UTR", 303, c_utr3)
    ]
}

# ==========================================
# 2. LECTURA DEL ALINEAMIENTO (Cgi en primera posición)
# ==========================================
try:
    records_raw = list(SeqIO.parse(archivo_alineamiento, "fasta"))
except FileNotFoundError:
    print(f"Error: No se encontró el archivo:\n{archivo_alineamiento}")
    raise

alineamiento = []
for especie in orden_deseado_especies:
    record = next((r for r in records_raw if especie in r.id), None)
    if record:
        alineamiento.append(record)
    else:
        print(f"Advertencia: No se encontró la especie {especie} en el archivo FASTA.")

if not alineamiento or 'Cgi' not in alineamiento[0].id:
    raise ValueError("Error crítico: 'Cgi' no se encontró o no quedó en primera posición.")

seqs = [str(r.seq).upper() for r in alineamiento]
etiquetas = [r.id for r in alineamiento]          # nombres tal cual vienen en el FASTA
n_esp = len(seqs)
L = len(seqs[0])
if any(len(s) != L for s in seqs):
    raise ValueError("Las secuencias no tienen la misma longitud: el archivo no parece un alineamiento.")

print(f"Alineamiento cargado: {n_esp} secuencias, {L} columnas.")

# ==========================================
# 3. REGIONES GÉNICAS SOBRE EL ALINEAMIENTO
#    Se cuentan las bases de Cgi a lo largo del alineamiento: la base número k
#    de Cgi (sin gaps) está en la columna cols_cgi[k].
# ==========================================
cols_cgi = np.array([i for i, b in enumerate(seqs[0]) if b != '-'])
n_cgi = len(cols_cgi)

estructura_gen = estructuras_cgi[gen_objetivo]
longitud_total_estructura = sum(l for _, l, _ in estructura_gen)
if longitud_total_estructura != n_cgi:
    print(f"Advertencia: la estructura suma {longitud_total_estructura} pb, "
          f"pero la secuencia de Cgi tiene {n_cgi} pb sin gaps.")

# Cada región se dibuja solo en las columnas donde Cgi tiene base; donde Cgi
# tiene gap, el rectángulo se corta y continúa después.
segmentos_regiones = []   # (nombre, color, [(inicio_columna, ancho), ...])
pos = 0                   # posición en la secuencia de Cgi (sin gaps)
for nombre, longitud, color in estructura_gen:
    ini, fin = pos, min(pos + longitud, n_cgi)
    pos += longitud
    if ini >= n_cgi:
        break
    cols = cols_cgi[ini:fin]
    cortes = np.where(np.diff(cols) != 1)[0] + 1
    tramos = []
    for grupo in np.split(cols, cortes):
        if len(grupo) > 0:
            tramos.append((int(grupo[0]), len(grupo)))
    segmentos_regiones.append((nombre, color, tramos))

# ==========================================
# 4. CONSERVACIÓN POR COLUMNA (sin ventanas, sin comparaciones por pares)
#    Para cada columna: fracción de las especies (de todas, n_esp) que
#    comparten la base más común. Los gaps y las N no cuentan como base.
# ==========================================
matriz = np.array([list(s) for s in seqs])                  # (n_esp, L)
conteos = np.stack([(matriz == b).sum(axis=0) for b in "ACGT"])   # (4, L)
max_comparten = conteos.max(axis=0)                         # (L,)
max_comparten = np.where(max_comparten >= min_coinciden, max_comparten, 0)
pct_columna = max_comparten / n_esp * 100                   # (L,)

def promedio_movil(valores, ventana):
    """Promedio de cada columna con sus vecinas (ventana centrada).
    En los extremos la ventana se trunca y se promedia solo lo disponible."""
    if ventana <= 1:
        return valores.astype(float)
    mitad = ventana // 2
    acumulado = np.concatenate(([0.0], np.cumsum(valores, dtype=float)))
    idx = np.arange(len(valores))
    ini = np.clip(idx - mitad, 0, len(valores))
    fin = np.clip(idx + mitad + 1, 0, len(valores))
    return (acumulado[fin] - acumulado[ini]) / (fin - ini)

pct_curva = promedio_movil(pct_columna, ventana_curva)      # solo para la curva inferior

tiene_base = np.isin(matriz, list("ACGT"))                  # (n_esp, L)
datos_heatmap = np.where(tiene_base, pct_columna[None, :], np.nan)
datos_heatmap = np.ma.masked_invalid(datos_heatmap)         # gaps/N -> blanco

# ==========================================
# 5. FIGURA
# ==========================================
cmap_calido = LinearSegmentedColormap.from_list(
    'calido', plt.cm.YlOrRd(np.linspace(0.2, 1, 256)))
cmap_calido.set_bad('white')
vmin, vmax = 0, 100

fig = plt.figure(figsize=(16, 7.5))

izq, ancho = 0.10, 0.80
ax_reg = fig.add_axes([izq, 0.800, ancho, 0.035])
ax_heat = fig.add_axes([izq, 0.425, ancho, 0.350], sharex=ax_reg)
ax_curva = fig.add_axes([izq, 0.090, ancho, 0.200], sharex=ax_reg)
ax_cbar = fig.add_axes([izq + ancho + 0.015, 0.425, 0.012, 0.350])

# --- Barra de regiones génicas de Cgi ---
for nombre, color, tramos in segmentos_regiones:
    ax_reg.broken_barh(tramos, (0, 1), facecolors=color, edgecolor='none')
ax_reg.set_ylim(0, 1)
ax_reg.set_xlim(0, L)
ax_reg.set_yticks([])
ax_reg.tick_params(axis='x', bottom=False, labelbottom=False)
for s in ax_reg.spines.values():
    s.set_visible(False)
ax_reg.text(-0.008, 0.5, 'Regiones de Cgi', transform=ax_reg.transAxes,
            ha='right', va='center', fontsize=12, weight='bold')

# --- Heatmap ---
im = ax_heat.imshow(datos_heatmap, aspect='auto', cmap=cmap_calido,
                    vmin=vmin, vmax=vmax, extent=(0, L, n_esp, 0),
                    interpolation='nearest')
for y in range(1, n_esp):
    ax_heat.axhline(y, color='black', linewidth=1.5)
ax_heat.set_yticks(np.arange(n_esp) + 0.5)
ax_heat.set_yticklabels(etiquetas, fontsize=14, weight='bold')
ax_heat.xaxis.set_major_locator(MultipleLocator(500))
ax_heat.tick_params(axis='x', labelsize=12)
ax_heat.set_xlabel('Posición en el alineamiento múltiple (columnas)',
                   fontsize=14, weight='bold', labelpad=6)
for s in ax_heat.spines.values():
    s.set_linewidth(1.5)

# --- Barra de color ---
cbar = fig.colorbar(im, cax=ax_cbar)
ticks_cbar = np.linspace(0, 100, 6)
cbar.set_ticks(ticks_cbar)
cbar.ax.set_yticklabels([f"{int(round(t))}%" for t in ticks_cbar], fontsize=11)
cbar.set_label('Identidad de la columna (%)', fontsize=12, weight='bold')

# --- Curva de porcentaje por columna (sin relleno) ---
ax_curva.plot(np.arange(L) + 0.5, pct_curva, color='black', linewidth=1.2)
ax_curva.set_ylim(0, 105)
ax_curva.set_yticks([0, 50, 100])
ax_curva.set_yticklabels(['0%', '50%', '100%'], fontsize=12)
ax_curva.xaxis.set_major_locator(MultipleLocator(500))
ax_curva.tick_params(axis='x', labelbottom=False)
ax_curva.grid(True, linestyle=':', alpha=0.4)
ax_curva.set_ylabel(f'Identidad\n(ventana de\n{ventana_curva} columnas)', fontsize=12, weight='bold',
                    rotation=0, ha='right', va='center', labelpad=10)

# --- Título y leyenda ---
fig.suptitle(
    f'Alineamiento múltiple de los genes {gen_objetivo.replace("Cgi", "")} '
    f'de las especies de cactus analizadas',
    fontsize=16, weight='bold', y=0.975)

orden_etiquetas = ['Promotor', "5' UTR", 'Exón', 'Intrón', "3' UTR"]
colores_por_nombre = {n: c for n, c, _ in segmentos_regiones}
handles = [Patch(facecolor=colores_por_nombre[n], edgecolor='none', label=n)
           for n in orden_etiquetas if n in colores_por_nombre]
handles.append(Patch(facecolor='white', edgecolor='black', label='Gap / N'))
fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.935),
           ncol=len(handles), frameon=False, fontsize=14)

# --- Guardar ---
file_path = os.path.join(
    output_dir, f"MSA_visualizador(005)_{gen_objetivo.replace('/', '')}.jpg")
plt.savefig(file_path, dpi=300, bbox_inches="tight")
print(f"Figura guardada en: {file_path}")

plt.show()