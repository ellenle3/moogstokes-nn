"""
spectrum_gui.py
GUI for setting spectrum preprocessing parameters (renormalization, regions, shifts)
and model parameters (Teff, logg, rK, B, vsini).

Plot is embedded and updates in-place on each "Run model" call.
Disabled regions are still plotted; the model line is drawn in gray.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import csv
import os
import ast

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# ── Constants ─────────────────────────────────────────────────────────────────
N_REGIONS   = 7
PARAMS_FILE = "data/spectrum_params.csv"
KERNELS     = ["box", "SPEX", "KECK", "IGRINS", None]
BTN         = dict(fg="black", bg="#d9d9d9", activeforeground="black",
                   activebackground="#c0c0c0", relief="raised", padx=6, pady=2)
CSV_COLS    = (
    ["filename", "kernel", "nyquist_bin", "regions"]
    + [f"shift_{i}" for i in range(N_REGIONS)]
    + [f"renorm_{i}" for i in range(N_REGIONS)]
    + [f"mask_{i}" for i in range(N_REGIONS)]
)

# ── CSV helpers ───────────────────────────────────────────────────────────────
def load_params_file(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, newline="") as f:
        return {row["filename"]: row for row in csv.DictReader(f)}


def save_params_file(records: dict, path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS)
        w.writeheader()
        w.writerows(records.values())


# ── Main application ──────────────────────────────────────────────────────────
class SpectrumGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("MoogStokes Spectrum Prep")
        self.resizable(True, True)
        self._csv_path_var = tk.StringVar(value=os.path.abspath(PARAMS_FILE))
        self._records = {}
        # Mask state: list of (xlo, xhi) per region
        self._masks         = [[] for _ in range(N_REGIONS)]
        # Per-region first-click x value (None = no pending click)
        self._pending_click = [None] * N_REGIONS
        # Matplotlib patch objects per region (rebuilt on each run)
        self._mask_patches  = [[] for _ in range(N_REGIONS)]
        # Dashed vline shown after first click (feedback indicator)
        self._pending_lines = [None] * N_REGIONS
        # Global undo history: list of region indices in insertion order
        self._mask_history  = []
        self._reload_csv(silent=True)
        self._build_ui()
        # Set trace after _build_ui so _status_var exists when it fires
        self._csv_path_var.trace_add("write", lambda *_: self._reload_csv())

    # ── UI construction ───────────────────────────────────────────────────────
    def _build_ui(self):
        PAD = dict(padx=8, pady=4)

        # Top pane: controls
        ctrl_frame = ttk.Frame(self, padding=10)
        ctrl_frame.pack(side="top", fill="x")

        row = 0

        # ── CSV path ──────────────────────────────────────────────────────────
        ttk.Label(ctrl_frame, text="Params CSV:").grid(
            row=row, column=0, sticky="w", **PAD)
        ttk.Entry(ctrl_frame, textvariable=self._csv_path_var, width=56).grid(
            row=row, column=1, columnspan=N_REGIONS - 1, sticky="ew", **PAD)
        tk.Button(ctrl_frame, text="Browse…",
                  command=self._browse_csv, **BTN).grid(
            row=row, column=N_REGIONS, sticky="w", **PAD)
        row += 1

        ttk.Separator(ctrl_frame, orient="horizontal").grid(
            row=row, column=0, columnspan=N_REGIONS + 1, sticky="ew", pady=4)
        row += 1

        # ── File selection ────────────────────────────────────────────────────
        ttk.Label(ctrl_frame, text="Spectrum file:").grid(
            row=row, column=0, sticky="w", **PAD)
        self._file_var = tk.StringVar()
        ttk.Entry(ctrl_frame, textvariable=self._file_var, width=56).grid(
            row=row, column=1, columnspan=N_REGIONS - 2, sticky="ew", **PAD)
        tk.Button(ctrl_frame, text="Browse…", command=self._browse, **BTN).grid(
            row=row, column=N_REGIONS - 1, sticky="w", **PAD)
        tk.Button(ctrl_frame, text="Reload CSV", command=self._reload_csv, **BTN).grid(
            row=row, column=N_REGIONS, sticky="w", **PAD)
        self._file_var.trace_add("write", lambda *_: self._on_file_change())
        row += 1

        ttk.Separator(ctrl_frame, orient="horizontal").grid(
            row=row, column=0, columnspan=N_REGIONS + 1, sticky="ew", pady=4)
        row += 1

        # ── Kernel ────────────────────────────────────────────────────────────
        ttk.Label(ctrl_frame, text="Kernel:").grid(row=row, column=0, sticky="w", **PAD)
        self._kernel_var = tk.StringVar(value="None")
        kernel_menu = tk.OptionMenu(
            ctrl_frame, self._kernel_var,
            *[str(k) if k is not None else "None" for k in KERNELS],
        )
        kernel_menu.config(fg="black", bg="#d9d9d9", activeforeground="black",
                           activebackground="#c0c0c0", highlightthickness=0)
        kernel_menu["menu"].config(fg="black", bg="#d9d9d9",
                                   activeforeground="black", activebackground="#c0c0c0")
        kernel_menu.grid(row=row, column=1, sticky="w", **PAD)
        ttk.Label(ctrl_frame, text="Nyquist bin:").grid(row=row, column=2, sticky="e", **PAD)
        self._nyquist_var = tk.StringVar(value="3")
        self._prev_nyquist = 3
        self._nyquist_var.trace_add("write", lambda *_: self._on_nyquist_change())
        ttk.Entry(ctrl_frame, textvariable=self._nyquist_var, width=5,
                  justify="center").grid(row=row, column=3, sticky="w", **PAD)
        row += 1
        ttk.Label(ctrl_frame, text="").grid(row=row, column=0)
        for i in range(N_REGIONS):
            ttk.Label(ctrl_frame, text=f"R{i}", anchor="center", width=7).grid(
                row=row, column=i + 1, **PAD)
        row += 1

        # ── Regions checkboxes ────────────────────────────────────────────────
        ttk.Label(ctrl_frame, text="Enabled:").grid(row=row, column=0, sticky="w", **PAD)
        self._region_vars = []
        for i in range(N_REGIONS):
            v = tk.BooleanVar(value=True)
            self._region_vars.append(v)
            ttk.Checkbutton(ctrl_frame, variable=v).grid(row=row, column=i + 1, **PAD)
        row += 1

        # ── Shifts ────────────────────────────────────────────────────────────
        ttk.Label(ctrl_frame, text="Shifts:").grid(row=row, column=0, sticky="w", **PAD)
        self._shift_vars = []
        for i in range(N_REGIONS):
            v = tk.StringVar(value="0")
            self._shift_vars.append(v)
            ttk.Entry(ctrl_frame, textvariable=v, width=7,
                      justify="center").grid(row=row, column=i + 1, **PAD)
        row += 1

        # ── Renormalization ───────────────────────────────────────────────────
        ttk.Label(ctrl_frame, text="Renorm:").grid(row=row, column=0, sticky="w", **PAD)
        self._renorm_vars = []
        for i in range(N_REGIONS):
            v = tk.StringVar(value="1.0")
            self._renorm_vars.append(v)
            ttk.Entry(ctrl_frame, textvariable=v, width=7,
                      justify="center").grid(row=row, column=i + 1, **PAD)
        row += 1

        ttk.Separator(ctrl_frame, orient="horizontal").grid(
            row=row, column=0, columnspan=N_REGIONS + 1, sticky="ew", pady=4)
        row += 1

        # ── Model parameters ──────────────────────────────────────────────────
        model_frame = ttk.LabelFrame(ctrl_frame, text="Model parameters", padding=6)
        model_frame.grid(row=row, column=0, columnspan=N_REGIONS + 1,
                         sticky="ew", **PAD)
        model_params = [("Teff", "3888"), ("logg", "3.74"), ("rK", "1.87"),
                        ("B", "1.78"), ("vsini", "12.71"), ("guess_shift", "0"),
                        ("ymin", "0.75"), ("ymax", "1.05")]
        self._model_vars = {}
        for col, (label, default) in enumerate(model_params):
            ttk.Label(model_frame, text=f"{label}:").grid(
                row=0, column=col * 2, sticky="e", padx=(8, 2))
            v = tk.StringVar(value=default)
            self._model_vars[label] = v
            ttk.Entry(model_frame, textvariable=v, width=8,
                      justify="center").grid(row=0, column=col * 2 + 1, padx=(0, 8))
        row += 1

        # ── Buttons + status ──────────────────────────────────────────────────
        bottom = ttk.Frame(ctrl_frame)
        bottom.grid(row=row, column=0, columnspan=N_REGIONS + 1,
                    sticky="ew", pady=(6, 2))
        tk.Button(bottom, text="Save parameters",
                   command=self._save, **BTN).pack(side="left", padx=4)
        tk.Button(bottom, text="Auto shifts",
                   command=self._auto_shifts, **BTN).pack(side="left", padx=4)
        tk.Button(bottom, text="Run model",
                   command=self._run, **BTN).pack(side="left", padx=4)
        self._status_var = tk.StringVar(value="Ready.")
        ttk.Label(bottom, textvariable=self._status_var,
                  foreground="gray").pack(side="left", padx=12)

        # ── Embedded plot ─────────────────────────────────────────────────────
        plot_frame = ttk.Frame(self)
        plot_frame.pack(side="top", fill="both", expand=True)

        # Two-column grid; 7 regions → 4 rows × 2 cols, last cell hidden
        self._fig, axs_grid = plt.subplots(4, 2, figsize=(8, 10), sharey=True)
        self._fig.patch.set_facecolor("#f0f0f0")
        self._axs = axs_grid.flatten()
        for ax in self._axs:
            ax.set_visible(False)          # blank until first run
        self._axs[N_REGIONS].set_visible(False)  # permanently hide spare cell

        self._canvas = FigureCanvasTkAgg(self._fig, master=plot_frame)
        self._canvas.get_tk_widget().pack(fill="both", expand=True)

        toolbar_frame = ttk.Frame(plot_frame)
        toolbar_frame.pack(fill="x")
        NavigationToolbar2Tk(self._canvas, toolbar_frame)

        # Mask interaction events
        self._canvas.mpl_connect("button_press_event", self._on_canvas_click)
        self.bind("<u>", lambda _e: self._undo_last_mask())

    # ── Mask interaction ──────────────────────────────────────────────────────
    def _on_canvas_click(self, event):
        """Handle click on a subplot to define mask boundaries."""
        # Ignore clicks outside axes or when a navigation tool is active
        if event.inaxes is None or event.button != 1:
            return
        toolbar = self._canvas.toolbar
        if toolbar is not None and toolbar.mode:
            return  # pan / zoom active — don't intercept

        region = next(
            (i for i, ax in enumerate(self._axs[:N_REGIONS])
             if event.inaxes is ax),
            None,
        )
        if region is None:
            return

        x   = event.xdata
        ax  = self._axs[region]

        if self._pending_click[region] is None:
            # ── First click: record x and draw a dashed guide line ───────────
            self._pending_click[region] = x
            line = ax.axvline(x, color="green", linestyle="--",
                              alpha=0.75, linewidth=1, zorder=5)
            self._pending_lines[region] = line
            self._status_var.set(
                f"R{region}: first boundary at {x:.2f} Å — click again for second boundary.")
            self._canvas.draw_idle()

        else:
            # ── Second click: finalise mask ───────────────────────────────────
            x1 = self._pending_click[region]
            xlo, xhi = min(x1, x), max(x1, x)

            # Remove guide line
            if self._pending_lines[region] is not None:
                try:
                    self._pending_lines[region].remove()
                except ValueError:
                    pass
                self._pending_lines[region] = None
            self._pending_click[region] = None

            # Store and draw
            self._masks[region].append((xlo, xhi))
            self._mask_history.append(region)
            patch = ax.axvspan(xlo, xhi, color="lightgreen",
                               alpha=0.35, zorder=0)
            self._mask_patches[region].append(patch)
            self._status_var.set(
                f"R{region}: mask added [{xlo:.2f}, {xhi:.2f}] Å  "
                f"(total masks in region: {len(self._masks[region])})")
            self._canvas.draw_idle()

    def _undo_last_mask(self):
        """Remove the most recently added mask across all regions."""
        if not self._mask_history:
            self._status_var.set("Nothing to undo.")
            return
        region = self._mask_history.pop()
        if self._masks[region]:
            self._masks[region].pop()
        if self._mask_patches[region]:
            try:
                self._mask_patches[region][-1].remove()
            except ValueError:
                pass
            self._mask_patches[region].pop()
        # Also cancel any pending click for that region
        if self._pending_click[region] is not None:
            self._pending_click[region] = None
            if self._pending_lines[region] is not None:
                try:
                    self._pending_lines[region].remove()
                except ValueError:
                    pass
                self._pending_lines[region] = None
        self._status_var.set(
            f"Undo: removed last mask from R{region}.")
        self._canvas.draw_idle()

    def _redraw_masks(self, region: int) -> None:
        """Re-draw all saved mask spans for a region (called after ax.cla())."""
        ax = self._axs[region]
        self._mask_patches[region] = []
        for xlo, xhi in self._masks[region]:
            patch = ax.axvspan(xlo, xhi, color="lightgreen",
                               alpha=0.35, zorder=0)
            self._mask_patches[region].append(patch)
        # Redraw pending guide line if mid-click
        self._pending_lines[region] = None  # old artist is gone after cla()
        if self._pending_click[region] is not None:
            line = ax.axvline(self._pending_click[region], color="green",
                              linestyle="--", alpha=0.75, linewidth=1, zorder=5)
            self._pending_lines[region] = line

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _browse_csv(self):
        path = filedialog.askopenfilename(
            title="Select params CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile=os.path.basename(self._csv_path_var.get()),
        )
        if path:
            self._csv_path_var.set(path)

    def _reload_csv(self, silent=False):
        path = self._csv_path_var.get().strip()
        self._records = load_params_file(path)
        if not silent:
            self._status_var.set(
                f"Loaded {len(self._records)} entries from {os.path.basename(path)}")

    def _on_nyquist_change(self):
        try:
            new = int(self._nyquist_var.get())
        except ValueError:
            return
        if new <= 0:
            self._nyquist_var.set(str(self._prev_nyquist))
            return
        if new == self._prev_nyquist:
            return
        scale = self._prev_nyquist / new
        for v in self._shift_vars:
            try:
                v.set(str(round(float(v.get()) * scale)))
            except ValueError:
                pass
        self._prev_nyquist = new

    def _browse(self):
        path = filedialog.askopenfilename(
            title="Select spectrum file",
            filetypes=[("nspec files", "*.nspec"), ("All files", "*.*")])
        if path:
            self._file_var.set(path)

    def _on_file_change(self):
        fname = self._file_var.get().strip()
        if not fname:
            return

        entered_base = os.path.basename(fname)
        record = self._records.get(entered_base) or self._records.get(fname)

        if record is None:
            return

        regions = ast.literal_eval(record["regions"])
        self._kernel_var.set(record.get("kernel", "None"))
        self._prev_nyquist = int(record.get("nyquist_bin", 3))
        self._nyquist_var.set(record.get("nyquist_bin", "3"))
        for i, v in enumerate(self._region_vars):
            v.set(i in regions)
        for i, v in enumerate(self._shift_vars):
            v.set(record[f"shift_{i}"])
        for i, v in enumerate(self._renorm_vars):
            v.set(record[f"renorm_{i}"])

        # Load masks
        self._masks        = [[] for _ in range(N_REGIONS)]
        self._mask_history = []
        for i in range(N_REGIONS):
            raw = record.get(f"mask_{i}", "None")
            if raw and raw.strip() not in ("None", ""):
                try:
                    parsed = ast.literal_eval(raw)
                    if isinstance(parsed, list):
                        self._masks[i] = [tuple(m) for m in parsed]
                        self._mask_history.extend([i] * len(self._masks[i]))
                except Exception:
                    pass
        # Redraw masks on any already-visible axes
        for i in range(N_REGIONS):
            if self._axs[i].get_visible() and self._masks[i]:
                self._redraw_masks(i)
        self._canvas.draw_idle()

        self._status_var.set(
            f"Loaded existing params for: {entered_base}")

    def _collect(self):
        fname = self._file_var.get().strip()
        if not fname:
            messagebox.showerror("Missing file", "Please select a spectrum file.")
            return None
        try:
            nyquist_bin = int(self._nyquist_var.get())
        except ValueError:
            messagebox.showerror("Invalid input", "Nyquist bin must be an integer.")
            return None
        try:
            shifts = [float(v.get()) for v in self._shift_vars]
            renorm = [float(v.get()) for v in self._renorm_vars]
        except ValueError:
            messagebox.showerror("Invalid input",
                                 "Shifts and renormalization must be numeric.")
            return None
        return fname, shifts, renorm, nyquist_bin

    def _collect_model(self):
        try:
            return {k: float(v.get()) for k, v in self._model_vars.items()}
        except ValueError:
            messagebox.showerror("Invalid input",
                                 "All model parameters must be numeric.")
            return None

    # ── Actions ───────────────────────────────────────────────────────────────
    def _save(self):
        parsed = self._collect()
        if parsed is None:
            return
        fname, shifts, renorm, nyquist_bin = parsed
        enabled = [i for i, v in enumerate(self._region_vars) if v.get()]

        row = {"filename": os.path.basename(fname), "kernel": self._kernel_var.get() or "None",
               "nyquist_bin": nyquist_bin, "regions": repr(enabled)}
        for i, s in enumerate(shifts):
            row[f"shift_{i}"] = s
        for i, r in enumerate(renorm):
            row[f"renorm_{i}"] = r
        for i in range(N_REGIONS):
            masks = self._masks[i]
            clean = [(float(a), float(b)) for a, b in masks]
            row[f"mask_{i}"] = repr(clean) if clean else "None"

        self._records[os.path.basename(fname)] = row
        csv_path = self._csv_path_var.get().strip()
        save_params_file(self._records, csv_path)
        self._status_var.set(
            f"Saved → {os.path.basename(csv_path)}  ({len(self._records)} entries)")

    def _auto_shifts(self):
        parsed = self._collect()
        if parsed is None:
            return
        model = self._collect_model()
        if model is None:
            return

        fname, shifts, renorm, nyquist_bin = parsed

        try:
            import copy
            from fitting import automatic_wavelength_shifts_values
            from nn_helpers import MoogStokesNN
            from spectra import SpectralDataForMoogStokes
        except ImportError as e:
            messagebox.showerror("Import error", str(e))
            return

        try:
            self._status_var.set("Computing auto shifts…")
            self.update_idletasks()

            obj = SpectralDataForMoogStokes(
                fname,
                name=os.path.splitext(os.path.basename(fname))[0],
                regions=range(N_REGIONS),
                shifts=np.array(shifts),
                renormalization=np.array(renorm),
            )
            obj.Nyquist_bin_spectrum(nyquist_bin)

            auto = automatic_wavelength_shifts_values(
                copy.deepcopy(obj),
                Teff=model["Teff"], logg=model["logg"],
                rK=model["rK"], B=model["B"],
                vsini=model["vsini"],
                guess_shift=int(model["guess_shift"]),
                use_nn=True,
            )

            for v, s in zip(self._shift_vars, auto):
                v.set(f"{s:.4g}")

            self._status_var.set("Auto shifts applied.")

        except Exception as e:
            messagebox.showerror("Runtime error", str(e))

    def _run(self):
        parsed = self._collect()
        if parsed is None:
            return
        model = self._collect_model()
        if model is None:
            return

        fname, shifts, renorm, nyquist_bin = parsed
        enabled = {i for i, v in enumerate(self._region_vars) if v.get()}

        try:
            from nn_helpers import MoogStokesNN
            from spectra import SpectralDataForMoogStokes, MoogStokesModel
        except ImportError as e:
            messagebox.showerror("Import error", str(e))
            return

        # Instantiate once and cache
        if not hasattr(self, "_moognn"):
            self._moognn = MoogStokesNN()

        try:
            testdata = SpectralDataForMoogStokes(
                fname,
                name=os.path.splitext(os.path.basename(fname))[0],
                regions=range(N_REGIONS),      # always load all regions
                shifts=np.array(shifts),
                renormalization=np.array(renorm),
            )
            testdata.Nyquist_bin_spectrum(nyquist_bin)

            for r, ax in enumerate(self._axs[:N_REGIONS]):
                ax.cla()
                ax.set_visible(True)

                x, y, yerr = testdata.get_region(r)
                ax.plot(x, y, color="black", label="Observed")

                testmodel = self._moognn.make_moogstokes_model(
                    Teff=model["Teff"], logg=model["logg"],
                    rK=model["rK"], B=model["B"],
                    vsini=model["vsini"], region=r,
                )
                model_color = "red" if r in enabled else "#aaaaaa"
                model_label = "Model" if r in enabled else "Model (disabled)"
                ax.plot(testmodel.x, testmodel.y,
                        color=model_color, label=model_label)

                xlo, xhi = MoogStokesModel.region_xlims(r)
                ax.set_xlim(xlo, xhi)
                ax.set_ylim(model["ymin"], model["ymax"])
                ax.set_title(f"R{r}", fontsize=9)
                ax.tick_params(labelsize=7)
                self._redraw_masks(r)

            self._fig.suptitle(
                f"Teff={model['Teff']}  logg={model['logg']}  "
                f"rK={model['rK']}  B={model['B']}  vsini={model['vsini']}",
                fontsize=9,
            )
            self._fig.tight_layout(pad=1.5)
            self._canvas.draw()
            self._status_var.set("Plot updated.")

        except Exception as e:
            messagebox.showerror("Runtime error", str(e))


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = SpectrumGUI()
    app.mainloop()
