import uproot
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Load ROOT file and TTree
file = uproot.open("/home/mileshar/dream/cosmics_analysis/run316/run316_250517140056.root")
tree = file["EventTree;539"]
board_number = 5
# Load HG and LG energy arrays for board 5
hg_data = tree[f"FERS_Board{board_number}_energyHG"].array(library="np")
lg_data = tree[f"FERS_Board{board_number}_energyLG"].array(library="np")

# Channel remap (FERS to layout)
channel_remap = [4,2,3,1,8,6,7,5,11,9,12,10,15,13,16,14,
                 20,18,19,17,24,22,23,21,27,25,28,26,
                 31,29,32,30,36,34,35,33,40,38,39,37,
                 43,41,44,42,47,45,48,46,52,50,51,49,
                 56,54,55,53,59,57,60,58,63,61,64,62]
channel_remap = [i - 1 for i in channel_remap]

# Define Cherenkov vs Scintillating (alternating by 4s)
is_cherenkov = np.array([(i % 8) < 4 for i in range(64)])
is_scint = ~is_cherenkov

# Define 32 pairs: one Cherenkov and one Scintillating channel per pair
pair_indices = []
for i in range(0, 64, 8):  # every group of 8 channels
    pair_indices.extend([
        (i + 0, i + 4), (i + 1, i + 5),
        (i + 2, i + 6), (i + 3, i + 7),
    ])

# Remap channel indices
pair_indices = [(channel_remap[ch], channel_remap[sc]) for ch, sc in pair_indices]

# Collect values for each of 32 pairs
scint_vals = [[] for _ in range(32)]
cherenkov_vals = [[] for _ in range(32)]

# Fill
for hg_event, lg_event in zip(hg_data, lg_data):
    reordered_hg = hg_event[channel_remap]
    reordered_lg = lg_event[channel_remap]
    for i, (ch_idx, sc_idx) in enumerate(pair_indices):
        ch_val = reordered_hg[ch_idx] - 200
        sc_val = reordered_lg[sc_idx] - 200  # Use LG for scintillating

        if ch_val >= 0 and sc_val >= 0:
            cherenkov_vals[i].append(ch_val)
            scint_vals[i].append(sc_val)

# Plot 2D histograms
pdf_name = f'S(LG)_C(HG)_Coor_Board{board_number}_R316.pdf'
with PdfPages(pdf_name) as pdf:
    for i in range(32):
        x = scint_vals[i]
        y = cherenkov_vals[i]
        if len(x) == 0:
            continue

        plt.figure(figsize=(6, 5))
        plt.hist2d(x, y, bins=100, cmap='plasma', cmin=1)
        plt.colorbar(label="Counts")
        plt.xlabel("Scintillating ADC (pedestal subtracted, LG)")
        plt.ylabel("Cherenkov ADC (pedestal subtracted, HG)")
        plt.title(f"Board{board_number} Pair {i+1} — Entries: {len(x):,}")
        plt.tight_layout()
        pdf.savefig()
        plt.close()

print(f"PDF saved as: {pdf_name}")
