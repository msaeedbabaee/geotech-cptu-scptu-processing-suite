"""
Advanced CPTu / SCPTu Continuous Processing & Subsurface Characterization Engine.
Integrated with Robertson (1990) 9-Zone Normalized SBTn & Jefferies & Been (2015) Ic Boundaries.
Strictly compliant with the Canadian Foundation Engineering Manual (CFEM Ch 5).
"""

import io
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# -----------------------------------------------------------------------------
# 1. CORE DOMAIN LOGIC & COMPUTATIONAL ENGINE (OOP)
# -----------------------------------------------------------------------------

@dataclass
class CPTuCalibrationParams:
    """Hardware calibration and reference geostatic boundary conditions."""
    a_net: float = 0.75           # Net area ratio of cone (0.35 - 0.85)
    gw_table_m: float = 2.0       # Depth to static groundwater table (m)
    gamma_w: float = 9.81         # Unit weight of water (kN/m3)
    p_atm: float = 100.0          # Atmospheric reference stress (kPa)
    default_gamma: float = 18.5   # Seed unit weight for initial stress (kN/m3)


class CPTuProcessingEngine:
    """Core geotechnical calculations for continuous piezocone logging data."""

    def __init__(self, df_raw: pd.DataFrame, params: CPTuCalibrationParams):
        self.df = df_raw.copy()
        self.params = params
        self._ensure_required_columns()

    def _ensure_required_columns(self):
        """Verifies mandatory input columns exist before executing numerical pipeline."""
        required = ["Depth_m", "qc_MPa", "fs_MPa", "u2_kPa"]
        for col in required:
            if col not in self.df.columns:
                raise ValueError(f"Input dataset is missing required column: {col}")

    def process_continuous_profile(self) -> pd.DataFrame:
        """Executes full normalization, classification, and parameter extraction pipeline."""
        df = self.df.sort_values(by="Depth_m").reset_index(drop=True)
        z = df["Depth_m"].values
        qc = df["qc_MPa"].values * 1000.0  # Convert MPa to kPa
        fs = df["fs_MPa"].values * 1000.0  # Convert MPa to kPa
        u2 = df["u2_kPa"].values           # kPa

        n_pts = len(z)
        p_atm = self.params.p_atm
        gw = self.params.gw_table_m
        gamma_w = self.params.gamma_w
        a_net = self.params.a_net

        # Hydrostatic porewater pressure u0 (kPa)
        u0 = np.where(z > gw, (z - gw) * gamma_w, 0.0)

        # 1. Corrected cone tip resistance qt (kPa) [CFEM Eq. 5.16]
        qt = qc + (1.0 - a_net) * u2

        # 2. Friction ratio Rf (%) [CFEM Eq. 5.17]
        rf = np.where(qt > 1e-3, (fs / qt) * 100.0, 0.0)

        # 3. Unit weight estimation using comparative ensemble methods
        # Robertson & Cabal (2010) [CFEM Eq. 5.26]
        term_rf = np.maximum(rf, 0.001)
        term_qt = np.maximum(qt / p_atm, 0.01)
        gamma_robertson = gamma_w * (0.27 * np.log10(term_rf) + 0.36 * np.log10(term_qt) + 1.236)
        gamma_robertson = np.clip(gamma_robertson, 12.0, 24.0)

        # Mayne & Peuchen (2012) [CFEM Eq. 5.27]
        term_fs = np.maximum(fs / p_atm, 0.0001)
        gamma_mayne_fs = gamma_w * (1.22 + 0.345 * np.log10(100.0 * term_fs + 0.01))
        gamma_mayne_fs = np.clip(gamma_mayne_fs, 12.0, 24.0)

        # Mayne et al. (2022) Effective cone resistance [CFEM Eq. 5.28]
        q_e = np.maximum(qt - u2, 1.0)
        gamma_mayne_qe = gamma_w * (1.54 + 0.254 * np.log10(q_e / p_atm))
        gamma_mayne_qe = np.clip(gamma_mayne_qe, 12.0, 24.0)

        # Ensemble mean total unit weight [CFEM Section 5.4.4.4.1]
        gamma_t = (gamma_robertson + gamma_mayne_fs + gamma_mayne_qe) / 3.0

        # 4. Total vertical geostatic stress (sigma_v0) and effective stress (sigma'_v0)
        sigma_v0 = np.zeros(n_pts)
        for i in range(1, n_pts):
            dz = z[i] - z[i - 1]
            avg_gamma = 0.5 * (gamma_t[i] + gamma_t[i - 1])
            sigma_v0[i] = sigma_v0[i - 1] + avg_gamma * dz
        sigma_v0[0] = gamma_t[0] * z[0] if z[0] > 0 else 0.0
        sigma_v0_eff = np.maximum(sigma_v0 - u0, 1.0)

        # 5. Dimensionless normalized parameters
        # Net cone tip resistance qnet (kPa) [CFEM Eq. 5.18]
        q_net = qt - sigma_v0

        # Normalized cone resistance Q [CFEM Eq. 5.19]
        Q = np.maximum(q_net / sigma_v0_eff, 0.01)

        # Normalized friction ratio Fr (%) [CFEM Eq. 5.20]
        Fr = np.where(q_net > 1.0, (fs / q_net) * 100.0, 0.0)
        Fr = np.clip(Fr, 0.01, 20.0)

        # Normalized pore pressure parameter Bq [CFEM Eq. 5.21]
        delta_u = u2 - u0
        Bq = np.where(q_net > 1.0, delta_u / q_net, 0.0)
        Bq = np.clip(Bq, -0.2, 1.5)

        # 6. Soil Behaviour Type Index Ic (Jefferies & Been 2015) [CFEM Eq. 5.40]
        dim_qt = np.maximum(Q * (1.0 - Bq) + 1.0, 0.01)
        term1 = 3.0 - np.log10(dim_qt)
        term2 = 1.5 + 1.3 * np.log10(Fr)
        Ic = np.sqrt(term1**2 + term2**2)

        # Soil Behaviour Classification Zone (Robertson 1990 9-Zone SBTn)
        sbtn_zones = []
        sbtn_desc = []
        for q_val, fr_val in zip(Q, Fr):
            zone, desc = self._classify_robertson_1990(q_val, fr_val)
            sbtn_zones.append(zone)
            sbtn_desc.append(desc)

        # 7. Geotechnical engineering parameter extractions
        # Peak effective friction angle phi'_peak for sands (Kulhawy & Mayne 1990) [CFEM Eq. 5.34]
        phi_term = (qt / p_atm) / np.sqrt(sigma_v0_eff / p_atm)
        phi_peak = 17.6 + 11.0 * np.log10(np.maximum(phi_term, 0.1))
        phi_peak = np.clip(phi_peak, 20.0, 48.0)

        # Undrained shear strength su for cohesive deposits [CFEM Section 5.4.4.4.2]
        # Bearing factor Nkt [CFEM Fig. 5.12]
        Nkt = 10.5 - 4.6 * np.log(np.maximum(Bq + 0.1, 0.05))
        Nkt = np.clip(Nkt, 8.0, 25.0)
        su_Nkt = np.maximum(q_net, 0.0) / Nkt

        # Bearing factor N_delta_u [CFEM Eq. 5.30 & 5.31]
        N_delta_u = np.maximum(7.9 - 6.5 * np.log(np.maximum(Bq + 0.3, 0.05)), 4.0)
        su_Ndu = np.where(delta_u > 0.0, delta_u / N_delta_u, 0.0)

        # Bearing factor NkE [CFEM Eq. 5.32 & 5.33]
        NkE = np.maximum(4.5 - 10.66 * np.log(np.maximum(Bq + 0.2, 0.05)), 4.0)
        su_NkE = np.maximum(qt - u2, 0.0) / NkE

        # Preconsolidation yield stress and OCR (Mayne 1991, Jefferies & Been 2015) [CFEM Eqs. 5.42 & 5.43]
        m_prime = 1.0 - (0.28 / (1.0 + (Ic / 2.7)**3.0))
        sigma_p = 0.33 * (np.maximum(q_net, 1.0) ** m_prime)
        ocr = np.clip(sigma_p / sigma_v0_eff, 0.8, 40.0)

        # 8. Seismic SCPTu Processing (if Vs column is present)
        has_vs = "Vs_m_s" in df.columns
        if has_vs:
            vs = df["Vs_m_s"].values
            rho = (gamma_t * 1000.0) / 9.81  # Mass density in kg/m3
            g0 = (rho * (vs ** 2)) / 1e6      # Small-strain shear modulus G0 in MPa [CFEM Eq. 5.44]

            # Aging / Cementation Factor alpha (Schnaid 2005) [CFEM Eqs. 5.9 & 5.10]
            term_schnaid = (np.maximum(q_net, 1.0) / p_atm) ** (1.0 / 3.0)
            alpha_aging = np.where(term_schnaid > 0, (g0 * 1000.0) / (200.0 * term_schnaid * p_atm), 1.0)
        else:
            vs = np.full(n_pts, np.nan)
            g0 = np.full(n_pts, np.nan)
            alpha_aging = np.full(n_pts, np.nan)

        # Assemble final processed output DataFrame
        df_out = pd.DataFrame({
            "Depth_m": z,
            "qc_MPa": df["qc_MPa"],
            "fs_MPa": df["fs_MPa"],
            "u2_kPa": df["u2_kPa"],
            "u0_kPa": np.round(u0, 1),
            "qt_MPa": np.round(qt / 1000.0, 3),
            "Rf_pct": np.round(rf, 2),
            "sigma_v0_kPa": np.round(sigma_v0, 1),
            "sigma_v0_eff_kPa": np.round(sigma_v0_eff, 1),
            "q_net_MPa": np.round(q_net / 1000.0, 3),
            "Q_norm": np.round(Q, 2),
            "Fr_norm_pct": np.round(Fr, 2),
            "Bq_norm": np.round(Bq, 3),
            "Ic_Jefferies": np.round(Ic, 2),
            "SBTn_Zone": sbtn_zones,
            "SBTn_Description": sbtn_desc,
            "gamma_Robertson_kNm3": np.round(gamma_robertson, 2),
            "gamma_Mayne_fs_kNm3": np.round(gamma_mayne_fs, 2),
            "gamma_Mayne_qe_kNm3": np.round(gamma_mayne_qe, 2),
            "gamma_t_mean_kNm3": np.round(gamma_t, 2),
            "phi_peak_sand_deg": np.round(phi_peak, 1),
            "su_Nkt_kPa": np.round(su_Nkt, 1),
            "su_Ndu_kPa": np.round(su_Ndu, 1),
            "su_NkE_kPa": np.round(su_NkE, 1),
            "sigma_p_kPa": np.round(sigma_p, 1),
            "OCR": np.round(ocr, 2),
            "Vs_m_s": np.round(vs, 1),
            "G0_MPa": np.round(g0, 1),
            "Alpha_Aging": np.round(alpha_aging, 2)
        })

        return df_out

    @staticmethod
    def _classify_robertson_1990(Q: float, Fr: float) -> Tuple[int, str]:
        """Classifies normalized CPT data points into Robertson (1990) 9-zone SBTn."""
        if Q < 1.0 and Fr < 1.0:
            return 1, "Sensitive Fine-Grained"
        if Fr > 4.5:
            return 2, "Organic Soils / Peat"
        if Q < 12.0 and Fr >= 1.5:
            return 3, "Clays (Clay to Silty Clay)"
        if Q < 30.0 and Fr >= 1.0:
            return 4, "Silt Mixtures (Clayey Silt to Silty Clay)"
        if Q < 80.0 and Fr < 2.0:
            return 5, "Sand Mixtures (Silty Sand to Sandy Silt)"
        if Q >= 30.0 and Fr < 1.0:
            return 6, "Sands (Clean Sand to Silty Sand)"
        if Q >= 80.0 and Fr < 0.6:
            return 7, "Gravelly Sand to Dense Sand"
        if Q >= 100.0 and Fr >= 1.5:
            return 8, "Very Stiff Sand to Clayey Sand (Heavily OC)"
        if Q >= 15.0 and Fr >= 3.0:
            return 9, "Very Stiff Fine-Grained (Heavily OC / Cemented)"
        return 5, "Sand Mixtures"


# -----------------------------------------------------------------------------
# 2. ADVANCED ROBERTSON CHART GEOMETRY & CONTOURS ENGINE
# -----------------------------------------------------------------------------

ZONE_PALETTE = {
    1: {"name": "1: Sensitive Fine-Grained", "fill": "rgba(220, 220, 220, 0.45)", "edge": "#7f7f7f", "rgb": (0.86, 0.86, 0.86)},
    2: {"name": "2: Organic Soils / Peat", "fill": "rgba(141, 110, 99, 0.45)", "edge": "#5d4037", "rgb": (0.55, 0.43, 0.39)},
    3: {"name": "3: Clays (Clay to Silty Clay)", "fill": "rgba(179, 157, 219, 0.45)", "edge": "#512da8", "rgb": (0.70, 0.62, 0.86)},
    4: {"name": "4: Silt Mixtures", "fill": "rgba(129, 212, 250, 0.45)", "edge": "#0288d1", "rgb": (0.51, 0.83, 0.98)},
    5: {"name": "5: Sand Mixtures", "fill": "rgba(165, 214, 167, 0.45)", "edge": "#388e3c", "rgb": (0.65, 0.84, 0.65)},
    6: {"name": "6: Clean Sands to Silty Sands", "fill": "rgba(255, 224, 130, 0.45)", "edge": "#f57f17", "rgb": (1.0, 0.88, 0.51)},
    7: {"name": "7: Gravelly Sand to Dense Sand", "fill": "rgba(255, 171, 145, 0.45)", "edge": "#d84315", "rgb": (1.0, 0.67, 0.57)},
    8: {"name": "8: Very Stiff Sand (Heavily OC)", "fill": "rgba(206, 147, 216, 0.45)", "edge": "#7b1fa2", "rgb": (0.81, 0.58, 0.85)},
    9: {"name": "9: Very Stiff Fine-Grained (OC)", "fill": "rgba(176, 190, 197, 0.45)", "edge": "#455a64", "rgb": (0.69, 0.75, 0.77)},
}

ZONE_POLYGONS_LOG = {
    1: [(-1.0, -0.3), (-0.05, -0.3), (-0.05, 0.48), (-1.0, 0.48)],
    2: [(0.65, -0.3), (1.1, -0.3), (1.1, 1.48), (0.65, 1.48)],
    8: [(0.35, 1.7), (1.1, 1.7), (1.1, 3.0), (0.35, 3.0)],
    9: [(-0.1, 2.0), (0.35, 2.0), (0.35, 3.0), (-0.1, 3.0)],
}

ZONE_ANNOTATIONS = [
    {"text": "<b>1: Sensitive<br>Fine-Grained</b>", "fr": 0.28, "q": 0.8},
    {"text": "<b>2: Organic Soils<br>Peat</b>", "fr": 6.8, "q": 2.5},
    {"text": "<b>3: Clays<br>(Clay to Silty Clay)</b>", "fr": 3.0, "q": 6.5},
    {"text": "<b>4: Silt Mixtures<br>(Clayey Silt)</b>", "fr": 1.7, "q": 20.0},
    {"text": "<b>5: Sand Mixtures<br>(Silty Sand)</b>", "fr": 1.0, "q": 55.0},
    {"text": "<b>6: Sands<br>(Clean Sand)</b>", "fr": 0.42, "q": 150.0},
    {"text": "<b>7: Dense Sand /<br>Gravelly Sand</b>", "fr": 0.18, "q": 450.0},
    {"text": "<b>8: Stiff Sand<br>(Heavily OC)</b>", "fr": 4.5, "q": 160.0},
    {"text": "<b>9: Stiff Fine-Grained<br>(Cemented / OC)</b>", "fr": 1.4, "q": 220.0},
]


def calculate_ic_contour_arcs(ic_values: List[float]) -> Dict[float, Tuple[np.ndarray, np.ndarray]]:
    """Calculates constant Ic circular contour curves per Jefferies & Been (2015)."""
    contours = {}
    center_x = -1.5 / 1.3  # approx -1.1538
    center_y = 3.0

    for ic in ic_values:
        theta = np.linspace(-0.95 * np.pi / 2, 0.05 * np.pi, 100)
        log_fr = center_x + (ic * np.cos(theta)) / 1.3
        log_q = center_y - (ic * np.sin(theta))

        valid = (log_fr >= -1.0) & (log_fr <= 1.1) & (log_q >= -0.3) & (log_q <= 3.0)
        fr_vals = 10.0 ** log_fr[valid]
        q_vals = 10.0 ** log_q[valid]
        contours[ic] = (fr_vals, q_vals)

    return contours


# -----------------------------------------------------------------------------
# 3. BENCHMARK DATASET GENERATOR (CPTu / SCPTu)
# -----------------------------------------------------------------------------

def generate_benchmark_scptu_dataset() -> pd.DataFrame:
    """Generates realistic synthetic continuous CPTu and downhole shear wave velocity data."""
    np.random.seed(101)
    depths = np.linspace(0.2, 24.0, 240)
    rows = []

    for z in depths:
        # Layer 1: Sand Fill & Dense Sand (0 - 5.0m)
        if z <= 5.0:
            qc = 7.5 + 1.8 * z + np.random.normal(0, 0.3)
            fs = qc * 0.008 + np.random.normal(0, 0.005)
            u2 = 9.81 * max(0.0, z - 2.0)
            vs = 180.0 + 15.0 * z + np.random.normal(0, 5.0)
        # Layer 2: Soft to Firm Sensitive Silty Clay (5.0 - 14.0m)
        elif z <= 14.0:
            qc = 1.1 + 0.12 * (z - 5.0) + np.random.normal(0, 0.06)
            fs = qc * 0.032 + np.random.normal(0, 0.003)
            u2 = (9.81 * max(0.0, z - 2.0)) + 35.0 * (z - 5.0) + np.random.normal(0, 4.0)
            vs = 135.0 + 4.5 * (z - 5.0) + np.random.normal(0, 3.0)
        # Layer 3: Dense Glacial Sand-Gravel Till (> 14.0m)
        else:
            qc = 16.0 + 1.1 * (z - 14.0) + np.random.normal(0, 0.6)
            fs = qc * 0.015 + np.random.normal(0, 0.02)
            u2 = 9.81 * max(0.0, z - 2.0)
            vs = 320.0 + 8.0 * (z - 14.0) + np.random.normal(0, 8.0)

        rows.append({
            "Depth_m": round(z, 2),
            "qc_MPa": round(max(qc, 0.2), 3),
            "fs_MPa": round(max(fs, 0.002), 4),
            "u2_kPa": round(max(u2, 0.0), 1),
            "Vs_m_s": round(max(vs, 80.0), 1)
        })

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# 4. VISUALIZATION PLATFORMS (PLOTLY & MATPLOTLIB)
# -----------------------------------------------------------------------------

class CPTuVisualizer:
    """Multi-panel interactive Plotly visualization and 300-DPI Matplotlib publication graphics."""

    @staticmethod
    def create_continuous_profile_dashboard(df: pd.DataFrame) -> go.Figure:
        """Constructs synchronized 4-panel interactive geotechnical sounding log."""
        fig = make_subplots(
            rows=1, cols=4,
            shared_yaxes=True,
            horizontal_spacing=0.04,
            subplot_titles=(
                "Corrected Tip Resistance qt (MPa)",
                "Friction Sleeve fs (MPa) & Rf (%)",
                "Pore Pressure u2 & Hydrostatic u0 (kPa)",
                "SCPTu Vs (m/s) & G0 (MPa)"
            )
        )

        # Panel 1: Corrected cone tip resistance qt
        fig.add_trace(
            go.Scatter(x=df["qt_MPa"], y=df["Depth_m"], mode='lines', name='qt (MPa)',
                       line=dict(color='#1f77b4', width=2),
                       hovertemplate='Depth: %{y:.2f} m<br>qt: %{x:.2f} MPa'),
            row=1, col=1
        )

        # Panel 2: Sleeve friction fs and friction ratio Rf
        fig.add_trace(
            go.Scatter(x=df["fs_MPa"], y=df["Depth_m"], mode='lines', name='fs (MPa)',
                       line=dict(color='#d62728', width=1.5),
                       hovertemplate='Depth: %{y:.2f} m<br>fs: %{x:.3f} MPa'),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=df["Rf_pct"] * 0.05, y=df["Depth_m"], mode='lines', name='Rf (%) [scaled x0.05]',
                       line=dict(color='#ff7f0e', dash='dot', width=1.2)),
            row=1, col=2
        )

        # Panel 3: Penetration pore pressure u2 and static u0
        fig.add_trace(
            go.Scatter(x=df["u2_kPa"], y=df["Depth_m"], mode='lines', name='u2 (kPa)',
                       line=dict(color='#9467bd', width=2),
                       hovertemplate='Depth: %{y:.2f} m<br>u2: %{x:.1f} kPa'),
            row=1, col=3
        )
        fig.add_trace(
            go.Scatter(x=df["u0_kPa"], y=df["Depth_m"], mode='lines', name='u0 Static (kPa)',
                       line=dict(color='#2ca02c', dash='dash', width=1.5)),
            row=1, col=3
        )

        # Panel 4: Shear wave velocity Vs and small-strain shear modulus G0
        fig.add_trace(
            go.Scatter(x=df["Vs_m_s"], y=df["Depth_m"], mode='lines+markers', name='Vs (m/s)',
                       line=dict(color='#8c564b', width=1.8), marker=dict(size=3),
                       hovertemplate='Depth: %{y:.2f} m<br>Vs: %{x:.1f} m/s'),
            row=1, col=4
        )
        fig.add_trace(
            go.Scatter(x=df["G0_MPa"], y=df["Depth_m"], mode='lines', name='G0 (MPa)',
                       line=dict(color='#e377c2', dash='dot', width=1.5)),
            row=1, col=4
        )

        fig.update_yaxes(autorange='reversed', title_text="Depth Below Ground (m)", row=1, col=1)
        fig.update_xaxes(title_text="qt (MPa)", row=1, col=1)
        fig.update_xaxes(title_text="fs (MPa)", row=1, col=2)
        fig.update_xaxes(title_text="Pore Pressure (kPa)", row=1, col=3)
        fig.update_xaxes(title_text="Vs (m/s) / G0 (MPa)", row=1, col=4)

        fig.update_layout(
            height=700,
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=-0.14, xanchor="center", x=0.5),
            margin=dict(l=60, r=40, t=60, b=80)
        )
        return fig

    @staticmethod
    def generate_static_profiles_figure(df: pd.DataFrame) -> io.BytesIO:
        """Renders a 300 DPI, multi-panel static report figure ready for publication."""
        plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
        fig, axes = plt.subplots(1, 4, figsize=(16, 8), sharey=True, dpi=300)

        depths = df["Depth_m"]

        # Panel 1: Corrected cone resistance qt
        axes[0].plot(df["qt_MPa"], depths, color='#1f77b4', lw=1.8)
        axes[0].set_xlabel(r'Corrected $q_t$ (MPa)', fontsize=10, fontweight='bold')
        axes[0].set_ylabel('Depth Below Ground (m)', fontsize=10, fontweight='bold')
        axes[0].set_title('(a) Tip Resistance', fontsize=11, fontweight='bold')
        axes[0].invert_yaxis()
        axes[0].grid(True, which='both', ls='--', alpha=0.6)

        # Panel 2: Comparative unit weights
        axes[1].plot(df["gamma_Robertson_kNm3"], depths, '--', color='#ff7f0e', lw=1.3, label='Robertson (2010)')
        axes[1].plot(df["gamma_Mayne_fs_kNm3"], depths, ':', color='#2ca02c', lw=1.3, label='Mayne $f_s$ (2012)')
        axes[1].plot(df["gamma_Mayne_qe_kNm3"], depths, '-.', color='#9467bd', lw=1.3, label=r'Mayne $q_E$ (2022)')
        axes[1].plot(df["gamma_t_mean_kNm3"], depths, '-', color='black', lw=2.0, label=r'Ensemble Mean $\gamma_t$')
        axes[1].set_xlabel(r'Unit Weight $\gamma_t$ (kN/m$^3$)', fontsize=10, fontweight='bold')
        axes[1].set_title(r'(b) In-Situ $\gamma_t$ Evaluation', fontsize=11, fontweight='bold')
        axes[1].legend(loc='lower left', frameon=True, fontsize=8)
        axes[1].grid(True, which='both', ls='--', alpha=0.6)

        # Panel 3: Clay undrained shear strength interpretations
        axes[2].plot(df["su_Nkt_kPa"], depths, '-', color='#d62728', lw=1.8, label=r'$s_u$ ($N_{kt}$ Mode)')
        axes[2].plot(df["su_Ndu_kPa"], depths, '--', color='#8c564b', lw=1.4, label=r'$s_u$ ($N_{\Delta u}$ Mode)')
        axes[2].plot(df["su_NkE_kPa"], depths, ':', color='#e377c2', lw=1.4, label=r'$s_u$ ($N_{kE}$ Mode)')
        axes[2].set_xlabel(r'Undrained Shear Strength $s_u$ (kPa)', color='#d62728', fontsize=10, fontweight='bold')
        axes[2].set_title('(c) Clay Undrained Strength', fontsize=11, fontweight='bold')
        axes[2].legend(loc='lower right', frameon=True, fontsize=8)
        axes[2].grid(True, which='both', ls='--', alpha=0.6)

        # Panel 4: Small-strain stiffness G0 and shear wave velocity Vs
        ax4_twin = axes[3].twiny()
        axes[3].plot(df["G0_MPa"], depths, color='#17becf', lw=1.8, label=r'$G_0 = \rho V_s^2$')
        ax4_twin.plot(df["Vs_m_s"], depths, '--', color='#7f7f7f', lw=1.4, label=r'Shear Wave $V_s$')
        axes[3].set_xlabel(r'Small-Strain Modulus $G_0$ (MPa)', color='#17becf', fontsize=10, fontweight='bold')
        ax4_twin.set_xlabel(r'Shear Wave Velocity $V_s$ (m/s)', color='#7f7f7f', fontsize=10, fontweight='bold')
        axes[3].set_title('(d) Dynamic Elastic Properties', fontsize=11, fontweight='bold', pad=15)
        axes[3].grid(True, which='both', ls='--', alpha=0.6)

        fig.suptitle('Continuous Piezocone (CPTu) & Seismic (SCPTu) Geotechnical Suite (CFEM Ch 5)',
                     fontsize=13, fontweight='bold', y=0.98)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        return buf


class RobertsonChartPlotter:
    """Produces publication-standard vector graphics and interactive dashboards for 9-Zone SBTn."""

    @staticmethod
    def build_interactive_chart(df_cpt: pd.DataFrame) -> go.Figure:
        """Renders fully interactive Robertson (1990) chart with constant Ic arcs."""
        fig = go.Figure()

        # 1. Background shading for discrete zones (Zones 1, 2, 8, 9)
        for zone_id, poly_pts in ZONE_POLYGONS_LOG.items():
            fr_coords = [10.0 ** pt[0] for pt in poly_pts]
            q_coords = [10.0 ** pt[1] for pt in poly_pts]
            meta = ZONE_PALETTE[zone_id]
            fig.add_trace(
                go.Scatter(
                    x=fr_coords + [fr_coords[0]],
                    y=q_coords + [q_coords[0]],
                    fill="toself",
                    fillcolor=meta["fill"],
                    line=dict(color=meta["edge"], width=1.5, dash="dash"),
                    mode="lines",
                    name=meta["name"],
                    hoverinfo="skip",
                    showlegend=False
                )
            )

        # 2. Ic contour boundaries
        ic_contours = calculate_ic_contour_arcs([1.31, 2.05, 2.60, 2.95, 3.60])
        ic_names = {
            1.31: "Ic = 1.31 (Gravelly Sand)",
            2.05: "Ic = 2.05 (Zone 7 / Zone 6)",
            2.60: "Ic = 2.60 (Zone 6 / Zone 5)",
            2.95: "Ic = 2.95 (Zone 5 / Zone 4)",
            3.60: "Ic = 3.60 (Zone 4 / Zone 3)"
        }

        for ic_val, (fr_line, q_line) in ic_contours.items():
            fig.add_trace(
                go.Scatter(
                    x=fr_line,
                    y=q_line,
                    mode="lines",
                    line=dict(color="#37474f", width=2.0),
                    name=ic_names[ic_val],
                    hoverinfo="text",
                    text=f"Boundary Ic = {ic_val:.2f}",
                    showlegend=True
                )
            )

        # 3. Zone text annotations
        for item in ZONE_ANNOTATIONS:
            fig.add_annotation(
                x=math.log10(item["fr"]),
                y=math.log10(item["q"]),
                text=item["text"],
                showarrow=False,
                font=dict(size=10, color="#212121", family="Arial"),
                align="center",
                bgcolor="rgba(255, 255, 255, 0.65)",
                bordercolor="rgba(0,0,0,0.15)",
                borderwidth=1,
                borderpad=3
            )

        # 4. Continuous sounding points colored by depth
        fig.add_trace(
            go.Scatter(
                x=df_cpt["Fr_norm_pct"],
                y=df_cpt["Q_norm"],
                mode="markers",
                marker=dict(
                    size=8,
                    color=df_cpt["Depth_m"],
                    colorscale="Turbo",
                    colorbar=dict(
                        title=dict(text="Depth (m)", font=dict(size=12, family="Arial")),
                        thickness=16,
                        len=0.85
                    ),
                    line=dict(color="black", width=0.8),
                    showscale=True
                ),
                name="Sounding Data",
                text=[f"Depth: {z:.2f} m<br>Stratum: {desc}<br>Ic: {ic:.2f}"
                      for z, desc, ic in zip(df_cpt["Depth_m"], df_cpt["SBTn_Description"], df_cpt["Ic_Jefferies"])],
                hovertemplate="<b>%{text}</b><br>Fr: %{x:.2f}%<br>Q: %{y:.1f}<extra></extra>"
            )
        )

        fig.update_xaxes(
            type="log",
            title=dict(text="Normalized Friction Ratio, Fr (%)", font=dict(size=13, weight="bold")),
            range=[-1.0, 1.1],
            showgrid=True,
            gridcolor="#e0e0e0",
            dtick=1
        )
        fig.update_yaxes(
            type="log",
            title=dict(text="Normalized Cone Tip Resistance, Q", font=dict(size=13, weight="bold")),
            range=[-0.3, 3.0],
            showgrid=True,
            gridcolor="#e0e0e0",
            dtick=1
        )

        fig.update_layout(
            title=dict(
                text="<b>Robertson (1990) 9-Zone Normalized Soil Behavior Type Chart</b>",
                font=dict(size=15, family="Arial")
            ),
            template="plotly_white",
            height=720,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.22,
                xanchor="center",
                x=0.5,
                font=dict(size=10)
            ),
            margin=dict(l=70, r=40, t=60, b=110)
        )
        return fig

    @staticmethod
    def generate_static_300dpi_figure(df_cpt: pd.DataFrame) -> io.BytesIO:
        """Renders publication-grade 300 DPI figure for Robertson chart with full annotations."""
        plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
        fig, ax = plt.subplots(figsize=(9, 8), dpi=300)

        # Plot discrete zone boundary patches
        for zone_id, poly_pts in ZONE_POLYGONS_LOG.items():
            fr_coords = [10.0 ** pt[0] for pt in poly_pts]
            q_coords = [10.0 ** pt[1] for pt in poly_pts]
            polygon = patches.Polygon(
                list(zip(fr_coords, q_coords)),
                closed=True,
                facecolor=ZONE_PALETTE[zone_id]["rgb"],
                edgecolor=ZONE_PALETTE[zone_id]["edge"],
                alpha=0.45,
                linestyle="--",
                linewidth=1.2
            )
            ax.add_patch(polygon)

        # Draw constant Ic contour boundaries
        ic_contours = calculate_ic_contour_arcs([1.31, 2.05, 2.60, 2.95, 3.60])
        for ic_val, (fr_line, q_line) in ic_contours.items():
            ax.plot(fr_line, q_line, color="#263238", lw=1.8, zorder=2)
            idx_mid = int(len(fr_line) * 0.45)
            ax.text(
                fr_line[idx_mid], q_line[idx_mid], f"$I_c={ic_val}$",
                fontsize=8, fontweight='bold', color="#263238",
                rotation=-40, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.7)
            )

        # Add zone label boxes
        for item in ZONE_ANNOTATIONS:
            clean_text = item["text"].replace("<b>", "").replace("</b>", "").replace("<br>", "\n")
            ax.text(
                item["fr"], item["q"], clean_text,
                fontsize=7.5, color="#212121", ha="center", va="center", weight="bold",
                bbox=dict(boxstyle="square,pad=0.2", facecolor="white", edgecolor="#bdbdbd", lw=0.6, alpha=0.75)
            )

        # Overlay sounding measurement scatter points
        scatter = ax.scatter(
            df_cpt["Fr_norm_pct"],
            df_cpt["Q_norm"],
            c=df_cpt["Depth_m"],
            cmap="turbo",
            s=28,
            edgecolors="black",
            linewidths=0.6,
            zorder=4,
            alpha=0.9
        )
        cbar = plt.colorbar(scatter, ax=ax, pad=0.03, aspect=25)
        cbar.set_label("Depth Below Ground Surface (m)", fontsize=10, fontweight="bold")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.1, 12.0)
        ax.set_ylim(0.5, 1000.0)

        ax.set_xlabel(r"Normalized Friction Ratio, $F_r = \frac{f_s}{q_t - \sigma_{v0}} \times 100\%$", fontsize=11, fontweight="bold")
        ax.set_ylabel(r"Normalized Cone Resistance, $Q = \frac{q_t - \sigma_{v0}}{\sigma'_{v0}}$", fontsize=11, fontweight="bold")
        ax.set_title("Robertson (1990) Normalized Soil Behavior Type Chart (SBTn)\nCompliant with CFEM Ch 5 & Jefferies and Been (2015)",
                     fontsize=12, fontweight="bold", pad=12)

        ax.grid(True, which="both", ls=":", color="#b0bec5", alpha=0.7)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        buf.seek(0)
        plt.close(fig)
        return buf


# -----------------------------------------------------------------------------
# 5. EXCEL EXPORT ENGINE (OPENPYXL)
# -----------------------------------------------------------------------------

class CPTuExcelExporter:
    """Exports raw and processed continuous datasets into a formatted Excel sheet."""

    @staticmethod
    def export(df: pd.DataFrame, params: CPTuCalibrationParams) -> io.BytesIO:
        wb = openpyxl.Workbook()

        header_fill = PatternFill(start_color="1F497D", end_color="1F497D", fill_type="solid")
        sub_fill = PatternFill(start_color="DCE6F1", end_color="DCE6F1", fill_type="solid")
        font_header = Font(name="Calibri", size=10, bold=True, color="FFFFFF")
        font_sub = Font(name="Calibri", size=11, bold=True, color="1F497D")
        font_data = Font(name="Calibri", size=9)
        thin_border = Border(
            left=Side(style='thin', color='B0B0B0'),
            right=Side(style='thin', color='B0B0B0'),
            top=Side(style='thin', color='B0B0B0'),
            bottom=Side(style='thin', color='B0B0B0')
        )

        ws = wb.active
        ws.title = "CPTu_Processed_Profile"
        ws.views.sheetView[0].showGridLines = True

        ws["A1"] = "CONTINUOUS CPTu / SCPTu INTERPRETATION DATA SHEET"
        ws["A1"].font = Font(name="Calibri", size=13, bold=True, color="1F497D")
        ws["A2"] = "Compliant with Canadian Foundation Engineering Manual (CFEM Ch 5)"
        ws["A2"].font = Font(name="Calibri", size=10, italic=True)

        ws["A4"] = "CALIBRATION METADATA"
        ws["A4"].font = font_sub
        ws["A4"].fill = sub_fill

        calib_items = [
            ("Cone Net Area Ratio (a_net)", params.a_net),
            ("Groundwater Table Depth (m)", params.gw_table_m),
            ("Atmospheric Pressure Reference (kPa)", params.p_atm),
            ("Water Unit Weight (kN/m3)", params.gamma_w),
        ]
        r_idx = 5
        for k, v in calib_items:
            ws[f"A{r_idx}"] = k
            ws[f"B{r_idx}"] = v
            ws[f"A{r_idx}"].font = font_data
            ws[f"B{r_idx}"].font = font_data
            ws[f"A{r_idx}"].border = thin_border
            ws[f"B{r_idx}"].border = thin_border
            r_idx += 1

        r_idx += 2
        cols = list(df.columns)
        for c_idx, col_name in enumerate(cols, start=1):
            cell = ws.cell(row=r_idx, column=c_idx, value=col_name)
            cell.fill = header_fill
            cell.font = font_header
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            cell.border = thin_border

        for _, row_vals in df.iterrows():
            r_idx += 1
            for c_idx, col_name in enumerate(cols, start=1):
                val = row_vals[col_name]
                c = ws.cell(row=r_idx, column=c_idx, value=val)
                c.font = font_data
                c.border = thin_border
                c.alignment = Alignment(horizontal="center" if isinstance(val, (int, float)) else "left")

        for col in ws.columns:
            max_len = max(len(str(cell.value or '')) for cell in col)
            col_letter = get_column_letter(col[0].column)
            ws.column_dimensions[col_letter].width = max(max_len + 2, 10)

        buf = io.BytesIO()
        wb.save(buf)
        buf.seek(0)
        return buf


# -----------------------------------------------------------------------------
# 6. STREAMLIT APPLICATION (ENTRY POINT)
# -----------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="CFEM CPTu / SCPTu Characterization Suite",
        page_icon="⚡",
        layout="wide"
    )

    st.title("⚡ Advanced CPTu & SCPTu Continuous Processing Engine")
    st.markdown(
        """
        **Automated Piezocone Penetrometer Analysis & Robertson 9-Zone Normalized SBTn Characterization**  
        *Compliant with the Canadian Foundation Engineering Manual (CFEM Ch 5), Robertson (1990, 2010), Jefferies & Been (2015), and Mayne (2022).*
        """
    )
    st.write("---")

    # Sidebar: Cone Calibration & Subsurface Properties
    st.sidebar.header("⚙️ 1. Piezocone Hardware Calibration")
    a_net = st.sidebar.slider(
        "Net Area Ratio (a_net)",
        min_value=0.35, max_value=0.85, value=0.75, step=0.01,
        help="ASTM D5778 net area ratio determined in calibration triaxial cell (CFEM Section 5.4.4.1)."
    )

    gw_table = st.sidebar.number_input(
        "Static Groundwater Table Depth (m)",
        min_value=0.0, max_value=50.0, value=2.0, step=0.5
    )

    p_atm = st.sidebar.number_input(
        "Atmospheric Pressure Reference p_atm (kPa)",
        min_value=90.0, max_value=110.0, value=100.0, step=0.5
    )

    params = CPTuCalibrationParams(
        a_net=a_net,
        gw_table_m=gw_table,
        p_atm=p_atm
    )

    # Data Upload or Synthetic Fallback
    st.sidebar.header("📂 2. CPT Data Acquisition")
    uploaded_file = st.sidebar.file_uploader("Upload CPT Data (CSV or Excel)", type=["csv", "xlsx"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df_raw = pd.read_csv(uploaded_file)
            else:
                df_raw = pd.read_excel(uploaded_file)
            st.sidebar.success("Field sounding data imported successfully!")
        except Exception as e:
            st.sidebar.error(f"Error parsing file: {e}. Fallback to synthetic benchmark sounding.")
            df_raw = generate_benchmark_scptu_dataset()
    else:
        df_raw = generate_benchmark_scptu_dataset()

    # Execute core processing
    engine = CPTuProcessingEngine(df_raw, params)
    df_processed = engine.process_continuous_profile()

    # Top-Level Engineering Metrics
    max_depth = df_processed["Depth_m"].max()
    mean_qt = df_processed["qt_MPa"].mean()
    max_u2 = df_processed["u2_kPa"].max()
    mean_g0 = df_processed["G0_MPa"].mean()

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Sounding Depth Explored", f"{max_depth:.1f} m", delta="Continuous Profile")
    with m2:
        st.metric("Mean Corrected Tip (qt)", f"{mean_qt:.2f} MPa")
    with m3:
        st.metric("Max Pore Pressure (u2)", f"{max_u2:.0f} kPa", delta=f"{max_u2 - (max_depth*9.81):.0f} kPa Excess")
    with m4:
        st.metric("Mean Small-Strain G0", f"{mean_g0:.1f} MPa" if not np.isnan(mean_g0) else "N/A")

    st.write("")

    # Output Tabs
    tab_profiles, tab_sbtn, tab_data, tab_export = st.tabs([
        "📊 Continuous Geotechnical Profiles",
        "🧭 Robertson 9-Zone SBTn Chart",
        "📋 Processed Numerical Matrix",
        "📥 Professional Export Suite"
    ])

    with tab_profiles:
        st.subheader("Subsurface Continuous Profiling Log")
        fig_prof = CPTuVisualizer.create_continuous_profile_dashboard(df_processed)
        st.plotly_chart(fig_prof, use_container_width=True)

        st.info(
            "💡 **Interpretation Insights (CFEM Ch 5):**\n"
            "- Dense sand layers exhibit tip resistance qt > 5 MPa and friction ratio Rf < 1%.\n"
            "- Sensitive clays exhibit high penetration porewater pressures (u2 >> u0) and low friction ratios.\n"
            "- Small-strain shear modulus G0 = rho * Vs^2 establishes the baseline threshold for dynamic and seismic liquefaction assessments."
        )

    with tab_sbtn:
        st.subheader("🧭 Robertson (1990) 9-Zone Normalized SBTn & Soil Behavior Index (Ic)")
        st.markdown(
            """
            This classification system maps normalized cone resistance (Q) against normalized friction ratio (Fr).
            Curved arcs depict the continuous Soil Behavior Type Index (Ic) contours per Jefferies & Been (2015).
            """
        )

        col_sbtn_graph, col_sbtn_down = st.columns([3, 1])

        with col_sbtn_graph:
            fig_sbtn = RobertsonChartPlotter.build_interactive_chart(df_processed)
            st.plotly_chart(fig_sbtn, use_container_width=True)

        with col_sbtn_down:
            st.markdown("#### 📥 Publication Deliverables (300 DPI)")
            st.write("Vector/raster graphic ready for technical reports and engineering submittals:")

            img_sbtn_buf = RobertsonChartPlotter.generate_static_300dpi_figure(df_processed)
            st.image(img_sbtn_buf, caption="High-Resolution Print Preview (300 DPI)", use_container_width=True)

            st.download_button(
                label="⬇️ Download Robertson Chart (PNG - 300 DPI)",
                data=img_sbtn_buf,
                file_name="Robertson_1990_SBTn_9Zones_300DPI.png",
                mime="image/png"
            )

            st.markdown("---")
            st.markdown(
                """
                **Ic Boundary Classification Reference:**
                * Ic < 2.05: Gravelly Sand to Dense Sand
                * 2.05 <= Ic < 2.60: Clean Sand to Silty Sand
                * 2.60 <= Ic < 2.95: Silt Mixtures (Silty Sand to Sandy Silt)
                * 2.95 <= Ic < 3.60: Clays (Clayey Silt to Silty Clay)
                * Ic >= 3.60: Fine-Grained Clays & Sensitive Deposits
                """
            )

    with tab_data:
        st.subheader("Interpreted Continuous Geotechnical Parameters")
        st.dataframe(df_processed, use_container_width=True)

    with tab_export:
        st.subheader("Executive Deliverables for Geotechnical Reporting")
        st.markdown("Download publication-ready figures and structured spreadsheet calculation workbooks:")

        col_ex1, col_ex2 = st.columns(2)
        with col_ex1:
            st.markdown("##### 📄 Multi-Panel Sounding Profiles (300 DPI)")
            fig_buf = CPTuVisualizer.generate_static_profiles_figure(df_processed)
            st.image(fig_buf, caption="Multi-Panel Sounding Profile Preview", use_container_width=True)
            st.download_button(
                label="⬇️ Download Profiles Figure (PNG)",
                data=fig_buf,
                file_name="CFEM_CPTu_SCPTu_Profile.png",
                mime="image/png"
            )

        with col_ex2:
            st.markdown("##### 📊 Full Geotechnical Data Schedule (.xlsx)")
            st.markdown(
                """
                Includes:
                - **Metadata & Calibration:** Net area ratio a_net, water table depth, atmospheric pressure.
                - **Processed Sounding Dataset:** Normalized parameters (Q, Fr, Bq, Ic), shear strengths (su, phi'_peak), three-way unit weights, and dynamic properties (Vs, G0, alpha).
                """
            )
            excel_buf = CPTuExcelExporter.export(df_processed, params)
            st.download_button(
                label="⬇️ Download Processed Excel Schedule (.xlsx)",
                data=excel_buf,
                file_name="CFEM_CPTu_Processed_Schedule.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )


if __name__ == "__main__":
    main()
