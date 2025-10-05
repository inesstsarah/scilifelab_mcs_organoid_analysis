#!/usr/bin/env python3
"""
Organoid MEA analysis pipeline for **MCS HDF5** (mesh / 3D MEA)

Dependencies (install in your own env):
  pip install numpy scipy matplotlib pandas networkx neo elephant quantities
  pip install McsPyDataTools   # preferred for reading MCS HDF5 (optional, but recommended)
  pip install h5py             # fallback reader

Run:
  python mcs_organoid_analysis_mcs_h5.py --hdf5 your_file.h5 --outdir ./results_organoid
Optional:
  --stim_csv stim.csv   # CSV with a 't_s' (seconds) column for stimulus times
  --lowpass 200 --highpass 300 --notch 50
"""

import argparse
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import networkx as nx
import quantities as pq

# neo core class (NOT RawIO)
from neo.core import SpikeTrain

# elephant uses neo SpikeTrain objects
from elephant.spike_train_correlation import sttc
from elephant.conversion import BinnedSpikeTrain

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------- Utilities ----------------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _to_str_list(x):
    out = []
    for v in x:
        if isinstance(v, bytes):
            out.append(v.decode("utf-8", errors="ignore"))
        else:
            out.append(str(v))
    return out

def bandpass_filter(x, fs, lo, hi, order=3):
    sos = signal.butter(order, [lo, hi], btype='bandpass', fs=fs, output='sos')
    return signal.sosfiltfilt(sos, x, axis=0)

def lowpass_filter(x, fs, cutoff, order=4):
    sos = signal.butter(order, cutoff, btype='lowpass', fs=fs, output='sos')
    return signal.sosfiltfilt(sos, x, axis=0)

def notch_filter(x, fs, freq=50.0, q=30):
    if freq <= 0:
        return x
    b, a = signal.iirnotch(w0=freq, Q=q, fs=fs)
    return signal.filtfilt(b, a, x, axis=0)

def mad(x):
    return np.median(np.abs(x - np.median(x))) / 0.6745

def detect_spikes_mad(signal_bp, fs, thresh_factor=4.5, refractory_ms=1.0, detect_negative=True):
    """
    Threshold crossing on per-channel MAD (simple, robust MUA).
    Returns spike_times_seconds, spike_indices, threshold_used
    """
    s = signal_bp
    noise = mad(s)
    if noise == 0 or np.isnan(noise):
        return np.array([]), np.array([]), (-thresh_factor if detect_negative else thresh_factor)

    if detect_negative:
        thr = -thresh_factor * noise
        crossings = np.where((s[:-1] > thr) & (s[1:] <= thr))[0] + 1
    else:
        thr = thresh_factor * noise
        crossings = np.where((s[:-1] < thr) & (s[1:] >= thr))[0] + 1

    # refine peak in ±0.5 ms neighborhood
    pk_win = int(0.0005 * fs)
    if pk_win < 1:
        pk_win = 1
    spike_idx = []
    for c in crossings:
        i0 = max(c - pk_win, 0)
        i1 = min(c + pk_win, len(s) - 1)
        idx = i0 + (np.argmin(s[i0:i1+1]) if detect_negative else np.argmax(s[i0:i1+1]))
        spike_idx.append(idx)
    if not spike_idx:
        return np.array([]), np.array([]), thr

    spike_idx = np.asarray(spike_idx, dtype=int)
    # Enforce refractory
    refr = int(refractory_ms * 1e-3 * fs)
    keep = []
    last = -10**9
    for i in spike_idx:
        if i - last >= refr:
            keep.append(i)
            last = i
    spike_idx = np.asarray(keep, dtype=int)
    spike_times = spike_idx / fs
    return spike_times, spike_idx, thr

def burst_detect_max_interval(spike_times, max_isi=0.1, min_spikes=5):
    """
    Per-channel bursts: consecutive ISIs <= max_isi grouped into bursts with >= min_spikes.
    Returns list of (t_start, t_end)
    """
    n = len(spike_times)
    if n < min_spikes:
        return []
    bursts = []
    start = 0
    for i in range(1, n):
        if (spike_times[i] - spike_times[i-1]) > max_isi:
            if i - start >= min_spikes:
                bursts.append((spike_times[start], spike_times[i-1]))
            start = i
    if n - start >= min_spikes:
        bursts.append((spike_times[start], spike_times[-1]))
    return bursts

def compute_psd_welch(x, fs, nperseg_sec=2.0, noverlap_frac=0.5):
    nperseg = max(256, int(nperseg_sec * fs))
    noverlap = int(noverlap_frac * nperseg)
    f, Pxx = signal.welch(x, fs=fs, nperseg=nperseg, noverlap=noverlap)
    return f, Pxx

def band_power(f, Pxx, band):
    lo, hi = band
    idx = (f >= lo) & (f <= hi)
    if not np.any(idx):
        return 0.0
    return float(np.trapz(Pxx[idx], f[idx]))

def detect_network_bursts(spike_trains, window_ms=100, min_participation=0.2, duration_ms=200):
    """
    Detect network bursts: time bins where >=min_participation of channels fire
    for >=duration_ms. Returns list of (t_start, t_stop) in seconds.
    Compatible with Elephant versions that use either 'binsize' or 'bin_size'.
    """
    if not spike_trains:
        return []
    t_start = min(st.t_start for st in spike_trains)
    t_stop = max(st.t_stop for st in spike_trains)
    bin_q = window_ms * pq.ms

    # Elephant API compatibility: binsize vs bin_size
    try:
        bst = BinnedSpikeTrain(spike_trains, binsize=bin_q, t_start=t_start, t_stop=t_stop)
    except TypeError:
        # Older/newer Elephant may expect 'bin_size'
        bst = BinnedSpikeTrain(spike_trains, bin_size=bin_q, t_start=t_start, t_stop=t_stop)

    active = (bst.to_array() > 0).sum(axis=0)   # channels active per bin
    frac = active / len(spike_trains)
    thr = (frac >= min_participation).astype(int)

    min_bins = int(np.ceil(duration_ms / window_ms))
    events = []
    i = 0
    while i < len(thr):
        if thr[i] == 1:
            j = i
            while j < len(thr) and thr[j] == 1:
                j += 1
            if (j - i) >= min_bins:
                t0 = (t_start + i * bin_q).rescale(pq.s).magnitude
                t1 = (t_start + j * bin_q).rescale(pq.s).magnitude
                events.append((t0, t1))
            i = j
        else:
            i += 1
    return events


# ---------------------- MCS HDF5 loader ----------------------

def load_mcs_hdf5(filename):
    """
    Try McsPyDataTools first (preferred). Fallback to h5py parsing MCS HDF5 structure.

    Returns:
      data_volts: ndarray shape (nsamples, nchannels), float64
      fs: float sampling rate (Hz)
      ch_names: list[str] length = nchannels
    """
    # Preferred path: official MCS Python tools
    try:
        import McsPy.McsData as McsData  # installed via 'pip install McsPyDataTools'
        rawdata = McsData.RawData(str(filename))
        # Use first recording
        rec_idx = sorted(rawdata.recordings.keys())[0]
        rec = rawdata.recordings[rec_idx]
        # Select analog stream with most channels
        best_si, best_n = None, -1
        for si, stream in rec.analog_streams.items():
            n = stream.channel_data.shape[0]
            if n > best_n:
                best_n = n
                best_si = si
        s = rec.analog_streams[best_si]

        # ChannelData: shape (nchan, nsamp)
        ch_raw = s.channel_data[...]  # ints
        info = s.channel_infos       # structured info per channel

        row_idx = info['RowIndex'].astype(int).flatten()
        adzero  = info['ADZero'].astype(np.int64).flatten()
        conv    = info['ConversionFactor'].astype(np.float64).flatten()
        expo    = info['Exponent'].astype(np.int64).flatten()
        labels  = _to_str_list(info['Label'])

        # fs from Tick (microseconds)
        tick_us = info['Tick'].astype(np.int64).flatten()
        tick = int(np.median(tick_us))
        fs = 1e6 / tick

        nchan, nsamp = ch_raw.shape
        data_phys = np.zeros((nsamp, nchan), dtype=np.float64)
        for ch in range(nchan):
            r = row_idx[ch]
            data_phys[:, ch] = (ch_raw[r, :] - adzero[ch]) * conv[ch] * (10.0 ** expo[ch])

        return data_phys, float(fs), labels

    except Exception:
        # Fallback: parse via h5py
        import h5py
        with h5py.File(str(filename), 'r') as h5:
            # First recording group
            rec_keys = sorted([k for k in h5['/Data'].keys() if k.startswith('Recording_')],
                              key=lambda x: int(x.split('_')[1]))
            rec = h5['/Data'][rec_keys[0]]
            astreams = rec['AnalogStream']
            # pick stream with most channels
            best_key, best_n = None, -1
            for k in astreams.keys():
                n = astreams[k]['ChannelData'].shape[0]
                if n > best_n:
                    best_key, best_n = k, n
            s = astreams[best_key]
            ch_raw = s['ChannelData'][...]  # (nchan, nsamp)
            info   = s['InfoChannel'][...]  # structured array

            def field(name):
                f = info[name]
                # convert bytes → str where needed
                if f.dtype.kind in ('S', 'O', 'U'):
                    return np.array(_to_str_list(f))
                return np.array(f)

            row_idx = field('RowIndex').astype(int).flatten()
            adzero  = field('ADZero').astype(np.int64).flatten()
            conv    = field('ConversionFactor').astype(np.float64).flatten()
            expo    = field('Exponent').astype(np.int64).flatten()
            labels  = list(field('Label'))

            tick_us = field('Tick').astype(np.int64).flatten()
            tick = int(np.median(tick_us))
            fs = 1e6 / tick

            nchan, nsamp = ch_raw.shape
            data_phys = np.zeros((nsamp, nchan), dtype=np.float64)
            for ch in range(nchan):
                r = row_idx[ch]
                data_phys[:, ch] = (ch_raw[r, :] - adzero[ch]) * conv[ch] * (10.0 ** expo[ch])

            return data_phys, float(fs), labels

# ---------------------- Main Pipeline ----------------------

def main():
    ap = argparse.ArgumentParser(description="Organoid MEA analysis for MCS HDF5")
    ap.add_argument("--hdf5", required=True, help="Path to MCS HDF5 exported by DataManager")
    ap.add_argument("--outdir", required=True, help="Directory for outputs")
    ap.add_argument("--stim_csv", default=None, help="Optional CSV with column 't_s' (seconds)")
    ap.add_argument("--lowpass", type=float, default=200.0, help="LFP low-pass cutoff (Hz)")
    ap.add_argument("--highpass", type=float, default=300.0, help="Spike band high-pass cutoff (Hz)")
    ap.add_argument("--notch", type=float, default=50.0, help="Notch frequency (0 disables)")
    ap.add_argument("--sttc_dt_ms", type=float, default=10.0, help="STTC delta-t window (ms)")
    ap.add_argument("--conn_thr", type=float, default=0.3, help="STTC threshold for graph edges")
    ap.add_argument("--max_pairs", type=int, default=2000, help="Max channel pairs for STTC to cap runtime")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    ensure_dir(outdir); ensure_dir(outdir / "csv"); ensure_dir(outdir / "figs")

    # Load data
    data_volts, fs, ch_names = load_mcs_hdf5(args.hdf5)  # (nsamp, nch)
    nsamp, nch = data_volts.shape
    duration_sec = nsamp / fs
    print(f"[INFO] Loaded: {args.hdf5}")
    print(f"[INFO] Shape = {data_volts.shape}, fs = {fs:.2f} Hz, duration = {duration_sec/60:.1f} min")

    # Filters
    x = data_volts
    if args.notch > 0:
        x = notch_filter(x, fs=fs, freq=args.notch, q=30)
    lfp = lowpass_filter(x, fs=fs, cutoff=args.lowpass)
    spike_band = bandpass_filter(x, fs=fs, lo=args.highpass, hi=min(fs/2 - 1, 3000))

    # ---------- Spike detection & per-channel metrics ----------
    rows = []
    spike_trains = []
    for ch in range(nch):
        s_bp = spike_band[:, ch]
        stimes, sidx, thr = detect_spikes_mad(s_bp, fs=fs, thresh_factor=4.5, refractory_ms=1.0, detect_negative=True)
        fr = len(stimes) / duration_sec if duration_sec > 0 else 0.0
        if len(stimes) >= 2:
            isi_vals = np.diff(stimes)
            isi_mean = float(np.mean(isi_vals))
            isi_cv   = float(np.std(isi_vals) / (np.mean(isi_vals) + 1e-12))
            refrac   = float(np.mean(isi_vals < 0.002))  # 2 ms
        else:
            isi_mean = np.nan; isi_cv = np.nan; refrac = np.nan

        bursts = burst_detect_max_interval(stimes, max_isi=0.1, min_spikes=5)
        b_count = len(bursts)
        if b_count > 0:
            b_durs = [b[1]-b[0] for b in bursts]
            b_rate = b_count / duration_sec
            b_dur_mean = float(np.mean(b_durs)); b_dur_med = float(np.median(b_durs))
        else:
            b_rate = 0.0; b_dur_mean = np.nan; b_dur_med = np.nan

        rows.append(dict(
            channel = ch_names[ch] if ch < len(ch_names) else f"ch{ch}",
            ch_index = ch,
            spikes = len(stimes),
            firing_rate_hz = fr,
            isi_mean_s = isi_mean,
            isi_cv = isi_cv,
            refrac_violation_frac = refrac,
            burst_count = b_count,
            burst_rate_hz = b_rate,
            burst_dur_mean_s = b_dur_mean,
            burst_dur_median_s = b_dur_med,
            threshold_used = thr
        ))
        spike_trains.append(SpikeTrain(stimes * pq.s, t_start=0.0*pq.s, t_stop=duration_sec*pq.s))

    df_ch = pd.DataFrame(rows)
    df_ch.to_csv(outdir / "csv/per_channel_metrics.csv", index=False)

    # ---------- Network bursts ----------
    net_bursts = detect_network_bursts(spike_trains, window_ms=100, min_participation=0.2, duration_ms=200)
    pd.DataFrame(net_bursts, columns=["t_start_s", "t_stop_s"]).to_csv(outdir / "csv/network_bursts.csv", index=False)

    # ---------- Synchrony (STTC) & connectivity graph ----------
    dt_s = args.sttc_dt_ms / 1000.0
    all_pairs = [(i, j) for i in range(nch) for j in range(i+1, nch)]
    if len(all_pairs) > args.max_pairs:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(all_pairs), size=args.max_pairs, replace=False)
        pairs = [all_pairs[k] for k in idx]
    else:
        pairs = all_pairs

    sttc_vals = []
    for i, j in pairs:
        try:
            v = sttc(spike_trains[i], spike_trains[j], dt=dt_s*pq.s)
            sttc_vals.append(float(v))
        except Exception:
            sttc_vals.append(np.nan)
    df_sttc = pd.DataFrame([(i, j, v) for (i, j), v in zip(pairs, sttc_vals)], columns=["i", "j", "sttc"])
    df_sttc.to_csv(outdir / "csv/sttc_pairs.csv", index=False)

    # Build graph
    thr_conn = float(args.conn_thr)
    G = nx.Graph()
    for ch in range(nch):
        G.add_node(ch, name=(ch_names[ch] if ch < len(ch_names) else f"ch{ch}"),
                   fr=df_ch.loc[ch, "firing_rate_hz"] if ch < len(df_ch) else np.nan)
    for _, row in df_sttc.dropna().iterrows():
        if row["sttc"] >= thr_conn:
            G.add_edge(int(row["i"]), int(row["j"]), weight=float(row["sttc"]))

    deg = dict(G.degree())
    clust = nx.clustering(G, weight="weight")
    if (len(G) > 0) and nx.is_connected(G):
        avg_path = nx.average_shortest_path_length(G)
    else:
        avg_path = np.nan

    graph_summary = dict(
        n_nodes = G.number_of_nodes(),
        n_edges = G.number_of_edges(),
        mean_degree = float(np.mean(list(deg.values()))) if deg else 0.0,
        mean_clustering = float(np.mean(list(clust.values()))) if clust else 0.0,
        avg_shortest_path_length = float(avg_path)
    )
    (outdir / "csv/graph_summary.json").write_text(json.dumps(graph_summary, indent=2))

    # ---------- LFP PSD & band powers ----------
    bands = {"delta": (1, 4), "theta": (4, 8), "beta": (13, 30), "gamma": (30, 80)}
    bp_rows = []
    center = nsamp // 2
    win = int(min(fs * 120, nsamp))  # analyze up to 120 s
    i0 = max(0, center - win // 2); i1 = min(nsamp, i0 + win)

    for ch in range(nch):
        f, P = compute_psd_welch(lfp[i0:i1, ch], fs=fs, nperseg_sec=2.0, noverlap_frac=0.5)
        total_p = band_power(f, P, (1, 200))
        row = {"channel": ch_names[ch] if ch < len(ch_names) else f"ch{ch}", "total_1_200Hz": total_p}
        for name, (lo, hi) in bands.items():
            row[f"{name}_power"] = band_power(f, P, (lo, hi))
        bp_rows.append(row)
        if ch < 6:
            plt.figure()
            plt.semilogy(f, P)
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("PSD")
            plt.title(f"LFP PSD - {row['channel']}")
            plt.tight_layout()
            plt.savefig(outdir / f"figs/psd_{ch:03d}.png")
            plt.close()
    pd.DataFrame(bp_rows).to_csv(outdir / "csv/lfp_band_powers.csv", index=False)

    # ---------- Quick plots ----------
    # Raster (first 60 s)
    t_plot = min(60.0, duration_sec)
    plt.figure(figsize=(10, 6))
    for ch in range(nch):
        ts = spike_trains[ch].magnitude
        ts = ts[ts <= t_plot]
        plt.plot(ts, np.full_like(ts, ch), '|', markersize=3)
    for nb in net_bursts:
        if nb[0] <= t_plot:
            plt.axvspan(nb[0], min(nb[1], t_plot), alpha=0.2)
    plt.xlabel("Time (s)")
    plt.ylabel("Channel")
    plt.title("Raster (first 60 s) + network bursts")
    plt.tight_layout()
    plt.savefig(outdir / "figs/raster_60s.png")
    plt.close()

    # Firing-rate bars
    plt.figure(figsize=(10, 4))
    plt.bar(np.arange(nch), df_ch["firing_rate_hz"].values)
    plt.xlabel("Channel index")
    plt.ylabel("Firing rate (Hz)")
    plt.title("Per-channel firing rate")
    plt.tight_layout()
    plt.savefig(outdir / "figs/firing_rate_bar.png")
    plt.close()

    # Degree histogram
    plt.figure()
    plt.hist(list(deg.values()) if deg else [0], bins=20)
    plt.xlabel("Degree")
    plt.ylabel("#Nodes")
    plt.title(f"Connectivity degree distribution (STTC>{thr_conn})")
    plt.tight_layout()
    plt.savefig(outdir / "figs/degree_hist.png")
    plt.close()

    # ---------- Optional: stimulation analysis ----------
    if args.stim_csv:
        stim_path = Path(args.stim_csv)
        if stim_path.exists():
            stim_df = pd.read_csv(stim_path)
            if 't_s' not in stim_df.columns:
                raise ValueError("stim_csv must have a 't_s' column (seconds).")
            # 10 ms response probability per channel
            resp_prob = []
            for ch in range(nch):
                st = spike_trains[ch].magnitude
                w = 0.01  # 10 ms
                prob = 0.0
                for ts in stim_df['t_s'].values:
                    prob += np.any((st >= ts) & (st <= ts + w))
                resp_prob.append(prob / max(1, len(stim_df)))
            pd.DataFrame({
                "channel": [ch_names[ch] if ch < len(ch_names) else f"ch{ch}" for ch in range(nch)],
                "response_prob_10ms": resp_prob
            }).to_csv(outdir / "csv/stim_response_prob.csv", index=False)

            # PSTHs (±100 ms) for first few channels
            psth_win = 0.1
            psth_bins = np.linspace(-psth_win, psth_win, int(2*psth_win/0.001)+1)
            for ch in range(min(6, nch)):
                st = spike_trains[ch].magnitude
                rel = []
                for ts in stim_df['t_s'].values:
                    rel.extend(st[(st >= ts - psth_win) & (st <= ts + psth_win)] - ts)
                rel = np.array(rel)
                H, edges = np.histogram(rel, bins=psth_bins)
                centers = 0.5*(edges[:-1] + edges[1:])
                plt.figure()
                plt.bar(centers, H, width=np.diff(edges))
                plt.xlabel("Time relative to stim (s)")
                plt.ylabel("Spike count/bin")
                plt.title(f"PSTH - {ch_names[ch] if ch < len(ch_names) else f'ch{ch}'}")
                plt.tight_layout()
                plt.savefig(outdir / f"figs/psth_{ch:03d}.png")
                plt.close()

    # ---------- README ----------
    readme = f"""Organoid MEA analysis (MCS HDF5)
================================
File       : {args.hdf5}
Fs (Hz)    : {fs:.2f}
Duration   : {duration_sec/60:.2f} min
Channels   : {nch}

Key CSVs: per_channel_metrics.csv, network_bursts.csv, sttc_pairs.csv, graph_summary.json, lfp_band_powers.csv
Figures : raster_60s.png, firing_rate_bar.png, degree_hist.png, psd_*.png, (optional) psth_*.png

Parameters:
- Notch: {args.notch} Hz | LFP low-pass: {args.lowpass} Hz | Spike high-pass: {args.highpass} Hz
- Spike thresh: -4.5 x MAD | refractory 1 ms
- Channel burst: max ISI 100 ms, ≥5 spikes
- Network burst: 100 ms bins, ≥20% channels active, ≥200 ms
- STTC: dt={args.sttc_dt_ms} ms | graph edge threshold={args.conn_thr}
"""
    (outdir / "README.txt").write_text(readme)

    print("[DONE] Analysis complete →", str(outdir))

if __name__ == "__main__":
    main()
