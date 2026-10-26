#!/usr/bin/env python
"""
Figures demonstrating the effects of:
  1) SUSAN spatial smoothing at 3, 6, 9, and 12 mm FWHM
  2) High-pass temporal filtering (128 s cutoff)
for sub-021, using the outputs saved by psb6351_preproc.py.

Run on the HPC after the preproc pipeline finishes:
    python code/make_figures.py

Figures are saved to code/figures/
"""
import os
import re
import sys
from glob import glob

import numpy as np
import nibabel as nb
import matplotlib
matplotlib.use('Agg')  # no screen on the HPC, so draw straight to files
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------
base_dir = '/home/aoste011/Mattfeld_PSB6351'
sub = 'sub-021'
tr = 1.76            # seconds
hpf_cutoff = 128.0   # seconds (same as in the pipeline)

sink = os.path.join(base_dir, 'derivatives', 'preproc', sub)
out_dir = os.path.join(base_dir, 'code', 'figures')
os.makedirs(out_dir, exist_ok=True)


def find(pattern):
    return sorted(glob(os.path.join(sink, pattern), recursive=True))


# ------------------------------------------------------------------
# Find the output files
# ------------------------------------------------------------------
masked_files = find('masked/**/*.nii.gz')
if not masked_files:
    sys.exit('No files found in %s/masked -- has the pipeline finished?' % sink)

# Use task-study run-01 if it exists, otherwise the first run
pick = [f for f in masked_files if 'task-study' in f and 'run-01' in f]
masked_file = pick[0] if pick else masked_files[0]
m = re.search(r'(task-[A-Za-z0-9]+_run-\d+)', os.path.basename(masked_file))
run_id = m.group(1) if m else os.path.basename(masked_file).split('_bold')[0]
print('Using run:', run_id)


def same_run(files):
    return [f for f in files if run_id in os.path.basename(f)]


# Smoothed files sit in folders named _fwhm_3.0, _fwhm_6.0, ...
smoothed = {}
for f in same_run(find('smooth/**/*.nii.gz')):
    k = re.search(r'_fwhm_([0-9.]+)', f)
    if k:
        smoothed[float(k.group(1))] = f
if not smoothed:
    sys.exit('No smoothed files found for %s in %s/smooth' % (run_id, sink))
print('Kernels found:', sorted(smoothed))

hpf_files = same_run(find('hpf/**/*.nii.gz'))
mask_files = find('mask/**/*.nii.gz')

# ------------------------------------------------------------------
# FIGURE 1: same slice, unsmoothed vs. each smoothing kernel
# ------------------------------------------------------------------
ref = nb.load(masked_file)
nvol = ref.shape[3]
t = nvol // 2           # a volume from the middle of the run
z = ref.shape[2] // 2   # a slice from the middle of the brain


def get_slice(fname):
    return np.asarray(nb.load(fname).dataobj[:, :, z, t], dtype=float)


panels = [('Unsmoothed', masked_file)]
panels += [('%g mm FWHM' % k, smoothed[k]) for k in sorted(smoothed)]

base_slice = get_slice(masked_file)
brain_vals = base_slice[base_slice > 0]
vmin, vmax = np.percentile(brain_vals, [2, 98])  # same contrast for every panel

fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 4.4))
for ax, (title, fname) in zip(axes, panels):
    sl = base_slice if fname == masked_file else get_slice(fname)
    ax.imshow(sl.T, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=14)
    ax.axis('off')
fig.suptitle('%s %s: SUSAN spatial smoothing (axial slice %d, volume %d)'
             % (sub, run_id, z, t), fontsize=15)
fig.tight_layout()
fig1_path = os.path.join(out_dir, 'spatial_smoothing_%s.png' % run_id)
fig.savefig(fig1_path, dpi=150)
plt.close(fig)
print('Saved', fig1_path)

# ------------------------------------------------------------------
# FIGURE 2: time series before vs. after the high-pass filter
# ------------------------------------------------------------------
if not hpf_files:
    sys.exit('No high-pass filtered file found for %s in %s/hpf' % (run_id, sink))
hpf_file = hpf_files[0]

# Brain mask (same grid as the functional data)
if mask_files:
    mask = np.asarray(nb.load(mask_files[0]).dataobj) > 0
else:
    mask = np.asarray(ref.dataobj[..., t]) > 0

# Small 5x5x5 region near the center of the brain (averaging a few voxels
# makes the slow drift easier to see than a single noisy voxel)
coords = np.argwhere(mask)
com = coords.mean(axis=0)
cx, cy, cz = coords[np.argmin(((coords - com) ** 2).sum(axis=1))]
roi = (slice(max(cx - 2, 0), cx + 3),
       slice(max(cy - 2, 0), cy + 3),
       slice(max(cz - 2, 0), cz + 3))
roi_mask = mask[roi]


def roi_timeseries(fname):
    data = np.asarray(nb.load(fname).dataobj[roi[0], roi[1], roi[2], :], dtype=float)
    return data[roi_mask].mean(axis=0)


before = roi_timeseries(masked_file)
after = roi_timeseries(hpf_file)
removed = before - after
seconds = np.arange(len(before)) * tr

freqs = np.fft.rfftfreq(len(before), d=tr)
pow_before = np.abs(np.fft.rfft(before - before.mean())) ** 2
pow_after = np.abs(np.fft.rfft(after - after.mean())) ** 2

fig, axes = plt.subplots(3, 1, figsize=(12, 10))

axes[0].plot(seconds, before, color='0.55', label='Before high-pass filter')
axes[0].plot(seconds, after, color='C0', label='After high-pass filter')
axes[0].set_xlabel('Time (s)')
axes[0].set_ylabel('BOLD signal')
axes[0].set_title('Mean signal in a 5x5x5 voxel region centered at voxel (%d, %d, %d)'
                  % (cx, cy, cz))
axes[0].legend(loc='best')

axes[1].plot(seconds, removed, color='C3')
axes[1].axhline(0, color='k', lw=0.5)
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('Signal removed')
axes[1].set_title('What the filter removed (before minus after): slow drift')

keep = freqs <= 0.15
axes[2].plot(freqs[keep], pow_before[keep], color='0.55', label='Before')
axes[2].plot(freqs[keep], pow_after[keep], color='C0', label='After')
axes[2].axvline(1.0 / hpf_cutoff, color='C3', ls='--',
                label='Cutoff = 1/%g s = %.4f Hz' % (hpf_cutoff, 1.0 / hpf_cutoff))
axes[2].set_xlabel('Frequency (Hz)')
axes[2].set_ylabel('Power')
axes[2].set_title('Power spectrum: low frequencies (below the cutoff) are removed')
axes[2].legend(loc='best')

fig.suptitle('%s %s: high-pass temporal filter (sigma = %.2f volumes)'
             % (sub, run_id, hpf_cutoff / (2.0 * tr)), fontsize=15)
fig.tight_layout()
fig2_path = os.path.join(out_dir, 'highpass_filter_%s.png' % run_id)
fig.savefig(fig2_path, dpi=150)
plt.close(fig)
print('Saved', fig2_path)
print('Done.')

