"""
Advanced CPTu / SCPTu Continuous Processing & Subsurface Characterization Engine
Strictly based on Canadian Foundation Engineering Manual (CFEM Ch 5), Robertson (1990, 2010),
Jefferies & Been (2015), Kulhawy & Mayne (1990), Mayne (2014, 2022), and Schnaid (2005).
"""

import io
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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
    a_net: float = 0.75           # Net area ratio of cone (0.35 - 0.85)
    gw_table_m: float = 2.0       # Depth to static water table (m)
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
        required = ["Depth_m", "qc_MPa", "fs_MPa", "u2_kPa"]
        for col in required:
            if col not in self.df.columns:
                raise ValueError(f"Input data missing mandatory column: {col}")

    def process_continuous_profile(self) -> pd.DataFrame:
        df = self.df.sort_values(by="Depth_m").reset_index(drop=True)
        z = df["Depth_m"].values
        qc = df["qc_MPa"].values * 1000.0  # Convert to kPa
        fs = df["fs_MPa"].values * 1000.0  # Convert to kPa
        u2 = df["u2_kPa"].values           # kPa

        n_pts = len(z)
        p_atm = self.params.p_atm
        gw = self.params.gw_table_m
        gamma_w = self.params.gamma_w
        a_net = self.params.a_net

        # Hydrostatic pore pressure u0 (kPa)
        u0 = np.where(z > gw, (z - gw) * gamma_w, 0.0)

        # 1. Corrected cone resistance qt (kPa) [CFEM Eq. 5.16]
        qt = qc + (1.0 - a_net) * u2

        # 2. Friction ratio Rf (%) [CFEM Eq. 5.17]
        rf = np.where(qt > 1e-3, (fs / qt) * 100.0, 0.0)

        # 3. Unit weight estimation (Iterative / Comparative)
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

        # Recommended average unit weight [CFEM Section 5.4.4.4.1]
        gamma_t = (gamma_robertson + gamma_mayne_fs + gamma_mayne_qe) / 3.0

        # 4. Vertical total stress (sigma_v0) and effective stress (sigma'_v0)
        sigma_v0 = np.zeros(n_pts)
        for i in range(1, n_pts):
            dz = z[i] - z[i - 1]
            avg_gamma = 0.5 * (gamma_t[i] + gamma_t[i - 1])
            sigma_v0[i] = sigma_v0[i - 1] + avg_gamma * dz
        sigma_v0[0] = gamma_t[0] * z[0] if z[0] > 0 else 0.0
        sigma_v0_eff = np.maximum(sigma_v0 - u0, 1.0)

        # 5. Dimensionless normalized parameters
        # Net cone resistance qnet (kPa) [CFEM Eq. 5.18]
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

        # 7. Geotechnical design parameter extractions
        # Peak effective friction angle phi'_peak for sands (Kulhawy & Mayne 1990) [CFEM Eq. 5.34]
        phi_term = (qt / p_atm) / np.sqrt(sigma_v0_eff / p_atm)
        phi_peak = 17.6 + 11.0 * np.log10(np.maximum(phi_term, 0.1))
        phi_peak = np.clip(phi_peak, 20.0, 48.0)

        # Undrained shear strength su for clays (CFEM Section 5.4.4.4.2)
        # Nkt bearing factor [CFEM Fig. 5.12]
        Nkt = 10.5 - 4.6 * np.log(np.maximum(Bq + 0.1, 0.05))
        Nkt = np.clip(Nkt, 8.0, 25.0)
        su_Nkt = np.maximum(q_net, 0.0) / Nkt

        # N_delta_u bearing factor [CFEM Eq. 5.30 & 5.31]
        N_delta_u = np.maximum(7.9 - 6.5 * np.log(np.maximum(Bq + 0.3, 0.05)), 4.0)
        su_Ndu = np.where(delta_u > 0.0, delta_u / N_delta_u, 0.0)

        # NkE bearing factor [CFEM Eq. 5.32 & 5.33]
        NkE = np.maximum(4.5 - 10.66 * np.log(np.maximum(Bq + 0.2, 0.05)), 4.0)
        su_NkE = np.maximum(qt - u2, 0.0) / NkE

        # Yield stress and OCR (Mayne 1991, Jefferies & Been 2015) [CFEM Eqs. 5.42 & 5.43]
        m_prime = 1.0 - (0.28 / (1.0 + (Ic / 2.7)**3.0))
        sigma_p = 0.33 * (np.maximum(q_net, 1.0) ** m_prime)
        ocr = np.clip(sigma_p / sigma_v0_eff, 0.8, 40.0)

        # 8. Seismic SCPTu Processing (if Vs provided)
        has_vs = "Vs_m_s" in df.columns
        if has_vs:
            vs = df["Vs_m_s"].values
            rho = (gamma_t * 1000.0) / 9.81  # Density kg/m3
            g0 = (rho * (vs ** 2)) / 1e6      # Small-strain shear modulus G0 (MPa) [CFEM Eq. 5.44]

            # Aging / Cementation Factor alpha (Schnaid 2005) [CFEM Eqs. 5.9 & 5.10]
            # Schnaid lower bound: G0 / (200 * (qc_norm)^(1/3))
            term_schnaid = (np.maximum(q_net, 1.0) / p_atm) ** (1.0 / 3.0)
            alpha_aging = np.where(term_schnaid > 0, (g0 * 1000.0) / (200.0 * term_schnaid * p_atm), 1.0)
        else:
            vs = np.full(n_pts, np.nan)
            g0 = np.full(n_pts, np.nan)
            alpha_aging = np.full(n_pts, np.nan)

        # Assemble processed DataFrame
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
        """Classifies normalized CPT data into Robertson (1990) 9-zone SBTn chart."""
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
# 2. SYNTHETIC BENCHMARK DATASET GENERATOR (CPTu / SCPTu)
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
            # High dynamic excess pore pressure in undrained penetration
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
# 3. ADVANCED VISUALIZATIONS (PLOTLY & MATPLOTLIB)
# -----------------------------------------------------------------------------

class CPTuVisualizer:
    """Multi-panel interactive Plotly visualization and 300-DPI Matplotlib publication graphics."""

    @staticmethod
    def create_continuous_profile_dashboard(df: pd.DataFrame) -> go.Figure:
        """Standard synchronized geotechnical log: qt, fs, u2, Ic, and Vs/G0."""
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

        # Panel 1: qt
        fig.add_trace(
            go.Scatter(x=df["qt_MPa"], y=df["Depth_m"], mode='lines', name='qt (MPa)',
                       line=dict(color='#1f77b4', width=2),
                       hovertemplate='Depth: %{y:.2f} m<br>qt: %{x:.2f} MPa'),
            row=1, col=1
        )

        # Panel 2: fs & Rf
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

        # Panel 3: u2 & u0
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

        # Panel 4: Vs & G0
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
    def create_robertson_sbtn_plot(df: pd.DataFrame) -> go.Figure:
        """Interactive 9-zone Robertson (1990) Q vs Fr soil classification chart."""
        fig = go.Figure()

        # Scatter data points colored by Depth
        fig.add_trace(
            go.Scatter(
                x=df["Fr_norm_pct"],
                y=df["Q_norm"],
                mode='markers',
                marker=dict(
                    size=6,
                    color=df["Depth_m"],
                    colorscale='Viridis',
                    colorbar=dict(title="Depth (m)", x=1.02),
                    showscale=True
                ),
                text=df["SBTn_Description"],
                hovertemplate='<b>%{text}</b><br>Fr: %{x:.2f}%<br>Q: %{y:.1f}<br>Depth: %{marker.color:.1f} m'
            )
        )

        # Soil type guide text labels
        zone_labels = [
            (0.3, 0.4, "1: Sensitive Fine-Grained"),
            (6.0, 1.5, "2: Organic / Peat"),
            (3.0, 5.0, "3: Clays"),
            (1.5, 18.0, "4: Silt Mixtures"),
            (0.8, 50.0, "5: Sand Mixtures"),
            (0.3, 150.0, "6: Sands"),
            (0.15, 400.0, "7: Gravelly Sand"),
            (2.5, 200.0, "8: Very Stiff Sand"),
            (5.0, 30.0, "9: Very Stiff Fine-Grained"),
        ]
        for x_pos, y_pos, lbl in zone_labels:
            fig.add_annotation(
                x=math.log10(x_pos), y=math.log10(y_pos),
                text=lbl, showarrow=False,
                font=dict(size=9, color="rgba(70,70,70,0.8)")
            )

        fig.update_xaxes(type="log", title="Normalized Friction Ratio Fr (%)", range=[-1.0, 1.2])
        fig.update_yaxes(type="log", title="Normalized Cone Resistance Q", range=[-0.3, 3.2])
        fig.update_layout(
            title="Robertson (1990) 9-Zone Normalized SBTn Soil Classification Chart",
            template="plotly_white",
            height=600,
            margin=dict(l=60, r=40, t=60, b=60)
        )
        return fig

    @staticmethod
    def generate_static_publication_figure(df: pd.DataFrame) -> io.BytesIO:
        """Produces a 300 DPI, multi-panel print-ready geotechnical log figure."""
        plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
        fig, axes = plt.subplots(1, 4, figsize=(16, 8), sharey=True, dpi=300)

        depths = df["Depth_m"]

        # Panel 1: qt
        axes[0].plot(df["qt_MPa"], depths, color='#1f77b4', lw=1.8)
        axes[0].set_xlabel(r'Corrected $q_t$ (MPa)', fontsize=10, fontweight='bold')
        axes[0].set_ylabel('Depth Below Ground (m)', fontsize=10, fontweight='bold')
        axes[0].set_title('(a) Tip Resistance', fontsize=11, fontweight='bold')
        axes[0].invert_yaxis()
        axes[0].grid(True, which='both', ls='--', alpha=0.6)

        # Panel 2: Unit Weight Comparison
        axes[1].plot(df["gamma_Robertson_kNm3"], depths, '--', color='#ff7f0e', lw=1.3, label='Robertson (2010)')
        axes[1].plot(df["gamma_Mayne_fs_kNm3"], depths, ':', color='#2ca02c', lw=1.3, label='Mayne $f_s$ (2012)')
        axes[1].plot(df["gamma_Mayne_qe_kNm3"], depths, '-.', color='#9467bd', lw=1.3, label=r'Mayne $q_E$ (2022)')
        axes[1].plot(df["gamma_t_mean_kNm3"], depths, '-', color='black', lw=2.0, label=r'Ensemble Mean $\gamma_t$')
        axes[1].set_xlabel(r'Unit Weight $\gamma_t$ (kN/m$^3$)', fontsize=10, fontweight='bold')
        axes[1].set_title(r'(b) In-Situ $\gamma_t$ Evaluation', fontsize=11, fontweight='bold')
        axes[1].legend(loc='lower left', frameon=True, fontsize=8)
        axes[1].grid(True, which='both', ls='--', alpha=0.6)

        # Panel 3: Shear Strengths (su and phi')
        axes[2].plot(df["su_Nkt_kPa"], depths, '-', color='#d62728', lw=1.8, label=r'$s_u$ ($N_{kt}$ Mode)')
        axes[2].plot(df["su_Ndu_kPa"], depths, '--', color='#8c564b', lw=1.4, label=r'$s_u$ ($N_{\Delta u}$ Mode)')
        axes[2].plot(df["su_NkE_kPa"], depths, ':', color='#e377c2', lw=1.4, label=r'$s_u$ ($N_{kE}$ Mode)')
        axes[2].set_xlabel(r'Undrained Shear Strength $s_u$ (kPa)', color='#d62728', fontsize=10, fontweight='bold')
        axes[2].set_title('(c) Clay Undrained Strength', fontsize=11, fontweight='bold')
        axes[2].legend(loc='lower right', frameon=True, fontsize=8)
        axes[2].grid(True, which='both', ls='--', alpha=0.6)

        # Panel 4: Small-Strain Stiffness G0 & Vs
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


# -----------------------------------------------------------------------------
# 4. EXCEL EXPORT ENGINE (OPENPYXL)
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
# 5. STREAMLIT APPLICATION (ENTRY POINT)
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
        **Automated Piezocone Penetrometer Analysis & Seismic Small-Strain Stiffness Characterization**  
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

    # Process sounding data
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
            "💡 **Interpretation Insights (CFEM Section 5.4.4):**\n"
            "- Coarse-grained sands typically exhibit $q_t > 5\\text{ MPa}$ and $R_f < 1\\%$.\n"
            "- Sensitive clays exhibit high penetration porewater pressures ($u_2 \\gg u_0$) and low friction ratios ($R_f < 1\\%$).\n"
            "- Small-strain shear modulus $G_0 = \\rho V_s^2$ provides the benchmark elastic threshold for dynamic and seismic analysis."
        )

    with tab_sbtn:
        st.subheader("Soil Behaviour Type (SBTn) Classification")
        fig_sbtn = CPTuVisualizer.create_robertson_sbtn_plot(df_processed)
        st.plotly_chart(fig_sbtn, use_container_width=True)

    with tab_data:
        st.subheader("Interpreted Continuous Geotechnical Parameters")
        st.dataframe(df_processed, use_container_width=True)

    with tab_export:
        st.subheader("Executive Deliverables for Geotechnical Reporting")
        st.markdown("Download high-resolution vector figures and fully formatted Excel interpretation sheets.")

        col_ex1, col_ex2 = st.columns(2)
        with col_ex1:
            st.markdown("##### 📄 Multi-Panel Publication Graphic (300 DPI)")
            fig_buf = CPTuVisualizer.generate_static_publication_figure(df_processed)
            st.image(fig_buf, caption="300 DPI High-Resolution Print Preview (CFEM Standards)", use_container_width=True)
            st.download_button(
                label="⬇️ Download Publication Figure (PNG)",
                data=fig_buf,
                file_name="CFEM_CPTu_SCPTu_Profile.png",
                mime="image/png"
            )

        with col_ex2:
            st.markdown("##### 📊 Full Geotechnical Data Schedule (.xlsx)")
            st.markdown(
                """
                Includes:
                - **Metadata:** Calibration parameters ($a_{net}$, water table, $p_{atm}$).
                - **Sounding Dataset:** Normalized parameters ($Q$, $F_r$, $B_q$, $I_c$), unit weights ($\gamma_t$), strength profiles ($s_u$, $\phi'_{peak}$), and dynamic properties ($V_s$, $G_0$, $\\alpha$).
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
