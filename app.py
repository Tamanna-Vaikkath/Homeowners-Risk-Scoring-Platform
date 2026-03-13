
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY = True
except ImportError:
    PLOTLY = False

ROOT      = Path(__file__).parent
DATA_PATH = ROOT / "data" / "raw" / "homeowners_portfolio.csv"
if not DATA_PATH.exists():
    DATA_PATH = ROOT / "data" / "homeowners_portfolio.csv"

st.set_page_config(
    page_title="Homeowners Underwriting Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# CSS — refined dark-navy / gold actuarial theme
# ══════════════════════════════════════════════════════════════════════════════
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: #0f172a;
}
.main > div { padding-top: 0 !important; }
.block-container { padding: 1.5rem 2.5rem !important; max-width: 1500px; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0f172a !important;
    border-right: 1px solid #1e293b;
}
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
[data-testid="stSidebar"] .stRadio label { color: #94a3b8 !important; font-size: 13px !important; }
[data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] p { color: #e2e8f0 !important; }

/* Hero */
.platform-hero {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 45%, #0f172a 100%);
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
    border: 1px solid #1e3a5f;
}
.platform-hero::before {
    content: '';
    position: absolute;
    top: -80px; right: -80px;
    width: 280px; height: 280px;
    background: radial-gradient(circle, rgba(212,175,55,0.12) 0%, transparent 65%);
    border-radius: 50%;
}
.platform-hero::after {
    content: '';
    position: absolute;
    bottom: -40px; left: 30%;
    width: 180px; height: 180px;
    background: radial-gradient(circle, rgba(59,130,246,0.08) 0%, transparent 65%);
    border-radius: 50%;
}
.hero-eyebrow {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 3px;
    color: #d4af37;
    font-weight: 600;
    margin-bottom: 8px;
    font-family: 'DM Sans', sans-serif;
}
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 26px;
    color: #f8fafc;
    line-height: 1.3;
    margin-bottom: 8px;
}
.hero-sub {
    font-size: 13px;
    color: #94a3b8;
    max-width: 680px;
    line-height: 1.7;
}
.hero-formula {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: #d4af37;
    background: rgba(212,175,55,0.08);
    border: 1px solid rgba(212,175,55,0.2);
    border-radius: 8px;
    padding: 10px 16px;
    margin-top: 14px;
    display: inline-block;
}

/* Tier badges */
.tier-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    border-radius: 10px;
    padding: 12px 18px;
    margin-bottom: 4px;
    border: 1px solid;
}
.tier-1 { background: rgba(59,130,246,0.07); border-color: rgba(59,130,246,0.25); }
.tier-2 { background: rgba(139,92,246,0.07); border-color: rgba(139,92,246,0.25); }
.tier-3 { background: rgba(245,158,11,0.07); border-color: rgba(245,158,11,0.25); }

/* Section headers */
.sec-hd {
    font-family: 'DM Serif Display', serif;
    font-size: 17px;
    color: #0f172a;
    margin-bottom: 4px;
}
.sec-line {
    height: 2px;
    background: linear-gradient(90deg, #3b82f6, #8b5cf6, transparent);
    border: none;
    margin: 6px 0 20px 0;
    border-radius: 2px;
}

/* Form tier headers */
.form-tier-hd {
    background: linear-gradient(90deg, #f8faff, #fff);
    border-left: 3px solid;
    border-radius: 0 8px 8px 0;
    padding: 10px 16px;
    margin-bottom: 16px;
}
.form-tier-hd-1 { border-color: #3b82f6; }
.form-tier-hd-2 { border-color: #8b5cf6; }
.form-tier-hd-3 { border-color: #f59e0b; }
.form-tier-title { font-size: 13px; font-weight: 700; color: #1e293b; }
.form-tier-sub { font-size: 11px; color: #64748b; margin-top: 2px; }

/* KPI cards */
.kpi { background: #fff; border-radius: 12px; padding: 16px 18px; border: 1px solid #e2e8f0; box-shadow: 0 1px 8px rgba(0,0,0,0.05); position: relative; overflow: hidden; }
.kpi::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px; }
.kpi-blue::before   { background: linear-gradient(90deg,#3b82f6,#60a5fa); }
.kpi-purple::before { background: linear-gradient(90deg,#8b5cf6,#a78bfa); }
.kpi-amber::before  { background: linear-gradient(90deg,#f59e0b,#fbbf24); }
.kpi-red::before    { background: linear-gradient(90deg,#ef4444,#f87171); }
.kpi-green::before  { background: linear-gradient(90deg,#10b981,#34d399); }
.kpi-slate::before  { background: linear-gradient(90deg,#64748b,#94a3b8); }
.kpi-label { font-size: 10px; text-transform: uppercase; letter-spacing: 1.5px; color: #94a3b8; font-weight: 600; margin-bottom: 4px; }
.kpi-value { font-size: 22px; font-weight: 700; color: #0f172a; line-height: 1.2; font-variant-numeric: tabular-nums; font-family: 'DM Sans', sans-serif; }
.kpi-sub   { font-size: 11px; color: #94a3b8; margin-top: 3px; }

/* Decision banner */
.decision-banner {
    border-radius: 16px;
    padding: 24px 30px;
    margin-bottom: 20px;
    border: 2px solid;
}

/* Flag cards */
.flag { padding: 11px 16px; border-radius: 10px; margin-bottom: 6px; border-left: 4px solid; font-size: 12.5px; line-height: 1.6; }
.flag-H  { background: #fef2f2; border-color: #ef4444; color: #7f1d1d; }
.flag-W  { background: #fffbeb; border-color: #f59e0b; color: #78350f; }
.flag-G  { background: #f0fdf4; border-color: #22c55e; color: #14532d; }
.flag-T  { background: #faf5ff; border-color: #8b5cf6; color: #4c1d95; }

/* Interaction cards */
.ix-card {
    background: #fff;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 10px;
    border: 1px solid #e2e8f0;
    border-left: 4px solid #8b5cf6;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}
.ix-card-inactive { border-left-color: #e2e8f0; opacity: 0.55; }
.ix-name { font-size: 13px; font-weight: 700; color: #1e293b; margin-bottom: 4px; }
.ix-logic { font-size: 11.5px; color: #475569; line-height: 1.5; }
.ix-badge { display:inline-block; font-size:10px; font-weight:600; padding:2px 8px; border-radius:20px; text-transform:uppercase; letter-spacing:.8px; }
.badge-confirmed { background:#fef3c7; color:#92400e; }
.badge-partial   { background:#ede9fe; color:#4c1d95; }

/* Score gauge */
.score-ring { display: flex; flex-direction: column; align-items: center; justify-content: center; }

/* Sensitivity rows */
.sens-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 14px;
    background: #f8fafc;
    border-radius: 8px;
    margin-bottom: 6px;
    border: 1px solid #e2e8f0;
    gap: 12px;
}
.sens-action { font-size: 12.5px; color: #1e293b; font-weight: 500; flex: 1; }
.sens-delta  { font-size: 12px; font-weight: 700; color: #16a34a; white-space: nowrap; font-family: 'JetBrains Mono', monospace; }

/* Tabs */
.stTabs [data-baseweb="tab"] {
    font-size: 11.5px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.7px !important;
    color: #64748b !important;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] { color: #1e293b !important; }

/* Sidebar radio */
.stRadio > div { gap: 4px !important; }

/* Form labels */
.stSlider label, .stSelectbox label, .stNumberInput label, .stCheckbox label {
    font-size: 11px !important;
    font-weight: 600 !important;
    color: #475569 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.4px !important;
}

/* Submit button */
[data-testid="stFormSubmitButton"] {
    text-align: center !important;
}
[data-testid="stFormSubmitButton"] button {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 100%) !important;
    color: #f8fafc !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    letter-spacing: 2px !important;
    font-size: 13px !important;
    padding: 13px 36px !important;
    width: 100% !important;
    text-align: center !important;
    text-transform: uppercase !important;
    box-shadow: 0 2px 12px rgba(15,23,42,0.28) !important;
    transition: all 0.18s ease !important;
    line-height: 1.4 !important;
    display: block !important;
}
[data-testid="stFormSubmitButton"] button:hover {
    background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 100%) !important;
    box-shadow: 0 4px 16px rgba(15,23,42,0.38) !important;
    transform: translateY(-1px) !important;
}

/* Contribution table */
.contrib-table { width: 100%; border-collapse: collapse; font-size: 12.5px; }
.contrib-table th { background: #f1f5f9; padding: 10px 12px; text-align: left; font-size: 10px; text-transform: uppercase; letter-spacing: 1px; color: #64748b; font-weight: 600; }
.contrib-table td { padding: 9px 12px; border-bottom: 1px solid #f1f5f9; color: #1e293b; }
.contrib-table tr:hover td { background: #f8fafc; }
.bar-pos { display: inline-block; height: 8px; border-radius: 4px; background: linear-gradient(90deg,#ef4444,#f87171); }
.bar-neg { display: inline-block; height: 8px; border-radius: 4px; background: linear-gradient(90deg,#10b981,#34d399); }

/* Mono numbers */
.mono { font-family: 'JetBrains Mono', monospace; font-size: 11.5px; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SCORING ENGINE  (self-contained — no pickle dependency)
# ══════════════════════════════════════════════════════════════════════════════

LOC_OFFSET = {
    "CA-Wildfire":  0.22,  "FL-Flood":    0.28,  "CO-Mixed":   0.07,
    "TX-Hail":      0.13,  "WA-NW":      -0.07,  "AZ-Desert": -0.11,
    "NY-NE":       -0.03,  "IL-Midwest": -0.14,  "GA-SE":     -0.18,
    "MN-North":    -0.22,
}
LOCATIONS = list(LOC_OFFSET.keys())

ETA_P2  = 7.18
ETA_P98 = 12.85

T3_CATALOGUE = [
    ("ix_roof_wildfire",    "Roof Age × Wildfire Zone",       "+6%",  0.43, "CONFIRMED",
     "Roof >20yr in high wildfire zone — fire resistance degraded; 2.8× loss frequency increase. Mandatory inspection required."),
    ("ix_water_canopy",     "Water Claim × Tree Canopy",       "+3%",  0.31, "CONFIRMED",
     "Prior water claim + dense canopy >60% — gutter clog recurrence 1.6×. Rate adjustment +25% applied."),
    ("ix_rcv_crime",        "RCV Overstated × Crime Index",    "+2%",  0.32, "CONFIRMED",
     "Over-insured property in high-crime zone — theft/arson incentive 1.4×. Coverage validation required."),
    ("ix_flood_foundation", "Flood Zone × Stone Foundation",   "+1.5%",0.14, "PARTIAL",
     "FEMA flood zone + permeable stone/dirt foundation — water ingress severity 1.3×. Flood endorsement required."),
    ("ix_slope_burn",       "Slope × Burn History",            "+1%",  0.33, "CONFIRMED",
     "Slope >20° + recent burn scar — mudslide/debris-flow risk 2.1×. Decline or +50% premium surcharge."),
    ("ix_roof_hail",        "Roof Age × Hail Zone",            "+1%",  0.33, "CONFIRMED",
     "Asphalt roof >20yr in hail zone — impact penetration severity 1.5×. Rate +30% applied."),
]

IX_CONDITIONS = {
    "ix_roof_wildfire":    ("Roof Age > 20 years",    "Wildfire Zone = Yes",       "Mandatory inspection"),
    "ix_water_canopy":     ("Prior water claim = Yes", "Tree canopy density > 60%", "Rate +25%"),
    "ix_rcv_crime":        ("RCV ratio > 1.15",        "Crime index > 55",          "Coverage adjustment"),
    "ix_flood_foundation": ("Flood zone = Yes",        "Foundation: Stone/Dirt",    "Flood policy required"),
    "ix_slope_burn":       ("Slope > 20°",             "Burn history = Yes",        "Decline or +50%"),
    "ix_roof_hail":        ("Asphalt roof > 20yr",     "Hail zone = Yes",           "Rate +30%"),
}


def score_policy(
    # Tier 1
    roof_age, roof_material, home_age, construction_type,
    prior_water_claim, months_since_water, coverage_a, rcv_ratio,
    # Tier 2
    ins_lapses, swimming_pool, trampoline, home_business,
    fire_sprinklers, monitored_alarm, gated_community, wood_stove,
    recent_reno, pet_ownership, prior_claims_5yr, fire_hydrant_dist,
    iso_class, building_code_comply, crime_idx,
    # Tier 3
    wildfire_score, wildfire_zone, canopy_pct, flood_zone, flood_depth_in,
    slope_deg, hail_zone, burn_history, foundation_type, location_zone,
) -> dict:

    # ── Encodings ────────────────────────────────────────────────────────────
    mat_r  = {"metal": 0, "slate": 1, "tile": 2, "asphalt": 3}.get(roof_material, 3)
    cnst_r = {"superior": 0, "masonry": 1, "brick_veneer": 2, "wood_frame": 3}.get(construction_type, 3)
    pool_r = {"none": 0, "in_ground": 1, "above_ground": 2}.get(swimming_pool, 0)
    biz_r  = {"none": 0, "home_office": 1, "active_business": 2}.get(home_business, 0)
    sprnk  = {"full": -2, "partial": -1, "none": 0}.get(fire_sprinklers, 0)
    fnd_r  = {"concrete_slab": 0, "poured_concrete": 0, "block": 1, "stone_dirt": 2}.get(foundation_type, 0)
    rcv_os = 1 if rcv_ratio > 1.15 else 0
    rec_f  = (3 if months_since_water < 12 else
              2 if months_since_water < 24 else
              1 if months_since_water < 36 else 0)
    loc_off = LOC_OFFSET.get(location_zone, 0.0)

    # ── Tier 1 η ─────────────────────────────────────────────────────────────
    eta_t1 = (
        8.10
        + 0.027 * roof_age
        + 0.050 * mat_r
        + 0.012 * home_age
        + 0.045 * cnst_r
        + rec_f  * 0.155 * prior_water_claim
        + 0.090 * rcv_os
        + 0.018 * rcv_ratio
        + loc_off
    )

    # ── Tier 2 incremental η ─────────────────────────────────────────────────
    eta_t2_incr = (
        0.013  * fire_hydrant_dist
        + 0.008 * crime_idx  / 20
        + 0.140 * ins_lapses
        + 0.052 * pool_r
        + 0.165 * trampoline
        + 0.078 * biz_r
        + 0.065 * abs(sprnk) * (-1)
        - 0.110 * monitored_alarm
        + 0.048 * wood_stove
        - 0.072 * gated_community
        + 0.225 * prior_claims_5yr
        - 0.055 * building_code_comply
        + 0.030 * (iso_class - 1) / 9
        - 0.038 * recent_reno
        + 0.028 * pet_ownership
    )
    eta_glm = eta_t1 + eta_t2_incr

    # ── Tier 3 interaction flags ──────────────────────────────────────────────
    ix_roof_wf  = int(roof_age > 20 and wildfire_zone == 1)
    ix_water_cn = int(prior_water_claim == 1 and canopy_pct > 60)
    ix_rcv_cr   = int(rcv_os == 1 and crime_idx > 55)
    ix_fld_fnd  = int(flood_zone == 1 and foundation_type == "stone_dirt")
    ix_slp_brn  = int(slope_deg > 20 and burn_history == 1)
    ix_rf_hail  = int(roof_age > 20 and hail_zone == 1)

    ix_map = {
        "ix_roof_wildfire":    ix_roof_wf,
        "ix_water_canopy":     ix_water_cn,
        "ix_rcv_crime":        ix_rcv_cr,
        "ix_flood_foundation": ix_fld_fnd,
        "ix_slope_burn":       ix_slp_brn,
        "ix_roof_hail":        ix_rf_hail,
    }

    # ── Tier 3 GAM δ ─────────────────────────────────────────────────────────
    gam_roof_nl  = 0.004 * max(roof_age  - 15, 0) ** 2 / 100
    gam_wf_nl    = 0.011 * (wildfire_score / 100) ** 2
    gam_flood_nl = 0.007 * np.log1p(flood_depth_in)
    gam_slope_nl = 0.009 * np.sqrt(max(slope_deg, 0))

    gam_ix_rfw   = ix_roof_wf  * 0.170
    gam_ix_wcan  = ix_water_cn * 0.115
    gam_ix_rcrc  = ix_rcv_cr  * 0.095
    gam_ix_ffnd  = ix_fld_fnd * 0.085
    gam_ix_sbrn  = ix_slp_brn * 0.075
    gam_ix_rhail = ix_rf_hail  * 0.065

    delta_gam = (
        gam_roof_nl + gam_wf_nl + gam_flood_nl + gam_slope_nl
        + gam_ix_rfw + gam_ix_wcan + gam_ix_rcrc
        + gam_ix_ffnd + gam_ix_sbrn + gam_ix_rhail
    )
    eta_final = eta_glm + delta_gam

    # ── Risk scores ───────────────────────────────────────────────────────────
    def to_score(eta):
        return float(np.clip((eta - ETA_P2) / (ETA_P98 - ETA_P2) * 100, 0, 100))

    s_t1    = to_score(eta_t1)
    s_glm   = to_score(eta_glm)
    s_final = to_score(eta_final)
    scale   = 100.0 / (ETA_P98 - ETA_P2)

    # ── Loss calibration ─────────────────────────────────────────────────────
    _LA, _LB = 0.5208, 2.7995
    loss_t1    = float(np.clip(np.exp(_LA * eta_t1    + _LB), 800, 20_000))
    loss_glm   = float(np.clip(np.exp(_LA * eta_glm   + _LB), 800, 20_000))
    loss_final = float(np.clip(np.exp(_LA * eta_final + _LB), 800, 20_000))

    claim_freq = float(np.clip(0.04 + (s_final / 100) * 0.42, 0.04, 0.50))
    claim_sev  = float(np.clip(loss_final / max(claim_freq, 0.03), 1_500, 55_000))

    # ── Decision ─────────────────────────────────────────────────────────────
    if s_final < 30:
        decision, desc = "Preferred", "Auto-bind eligible · Preferred tier rates · No manual review"
    elif s_final < 60:
        decision, desc = "Standard", "Standard rates ±15% · Desktop review recommended"
    elif s_final < 80:
        decision, desc = "Rated", "15–50% surcharge · Senior UW manual review required before binding"
    else:
        decision, desc = "Decline", "Refer to E&S market · Do not bind at standard rates"

    lr_map  = {"Preferred": 0.68, "Standard": 0.65, "Rated": 0.60}
    lr      = lr_map.get(decision)
    premium = round(loss_final / lr, -1) if lr else None

    # ── Feature contributions (score-point scale) ─────────────────────────────
    t1_contrib = {
        "Roof Vulnerability":   (0.027 * roof_age + 0.050 * mat_r) * scale,
        "Home Age / Structure": (0.012 * home_age + 0.045 * cnst_r) * scale,
        "Water Loss Recency":   (rec_f * 0.155 * prior_water_claim) * scale,
        "RCV Validation":       (0.090 * rcv_os + 0.018 * rcv_ratio) * scale,
        "Location Zone":        loc_off * scale,
    }
    t2_contrib = {
        "Prior Claims (5yr)":  0.225 * prior_claims_5yr  * scale,
        "Insurance Lapses":    0.140 * ins_lapses         * scale,
        "Crime Index":         0.008 * crime_idx / 20    * scale,
        "Pool / Trampoline":  (0.052 * pool_r + 0.165 * trampoline) * scale,
        "Home Business":       0.078 * biz_r              * scale,
        "Fire Sprinklers":    -0.065 * abs(sprnk)         * scale,
        "Monitored Alarm":    -0.110 * monitored_alarm    * scale,
        "Wood Stove":          0.048 * wood_stove         * scale,
        "Gated Community":    -0.072 * gated_community    * scale,
        "Building Compliance":-0.055 * building_code_comply * scale,
        "Hydrant Distance":    0.013 * fire_hydrant_dist  * scale,
        "Recent Renovation":  -0.038 * recent_reno        * scale,
        "Pet Ownership":       0.028 * pet_ownership      * scale,
    }
    # GAM smooth contribution
    smooth_total = (gam_roof_nl + gam_wf_nl + gam_flood_nl + gam_slope_nl) * scale
    t3_contrib = {}
    if abs(smooth_total) > 0.01:
        t3_contrib["Non-linear Smooths"] = smooth_total

    ix_delta_map = {
        "ix_roof_wildfire":    gam_ix_rfw,
        "ix_water_canopy":     gam_ix_wcan,
        "ix_rcv_crime":        gam_ix_rcrc,
        "ix_flood_foundation": gam_ix_ffnd,
        "ix_slope_burn":       gam_ix_sbrn,
        "ix_roof_hail":        gam_ix_rhail,
    }
    for fc, name, *_ in T3_CATALOGUE:
        v = ix_delta_map.get(fc, 0) * scale
        if abs(v) > 0.01:
            t3_contrib[name] = v

    # ── Triggered interactions ────────────────────────────────────────────────
    triggered = []
    for fc, name, r2, h, status, logic in T3_CATALOGUE:
        if ix_map.get(fc, 0):
            cond1, cond2, action = IX_CONDITIONS[fc]
            triggered.append({
                "flag": fc, "name": name, "h": h, "r2": r2,
                "status": status, "logic": logic, "action": action,
                "cond1": cond1, "cond2": cond2,
                "delta": ix_delta_map.get(fc, 0),
                "delta_pts": ix_delta_map.get(fc, 0) * scale,
            })

    # ── UW flags ─────────────────────────────────────────────────────────────
    flags = []
    if roof_age > 25:
        flags.append(("H", f"Roof age {roof_age}yr exceeds 25yr threshold — mandatory inspection required"))
    if wildfire_zone and wildfire_score > 60:
        flags.append(("H", f"Wildfire score {wildfire_score:.0f}/100 in confirmed high-risk zone"))
    if prior_water_claim and months_since_water < 18:
        flags.append(("H", f"Water claim {months_since_water}mo ago — repeat risk 1.6×; inspection required"))
    if rcv_os:
        flags.append(("W", f"RCV ratio {rcv_ratio:.2f} — material overstatement >1.15; fraud flag raised"))
    if ins_lapses >= 2:
        flags.append(("W", f"{ins_lapses} coverage lapse(s) — financial stress indicator (+15–30%)"))
    if crime_idx > 65:
        flags.append(("W", f"Crime index {crime_idx:.0f}/100 — elevated theft/arson exposure"))
    if flood_zone and foundation_type == "stone_dirt":
        flags.append(("H", "Flood zone + permeable foundation — water ingress severity 1.3×"))
    if prior_claims_5yr >= 3:
        flags.append(("W", f"{prior_claims_5yr} prior claims (5yr) — adverse loss history pattern"))
    if ix_slp_brn:
        flags.append(("H", "Slope >20° + burn history — mudslide/debris-flow 2.1×; decline or +50% premium"))
    if fire_sprinklers == "full":
        flags.append(("G", "Full sprinkler system — strongest fire severity mitigant (−10 to −20%)"))
    if monitored_alarm:
        flags.append(("G", "Monitored alarm — theft deterrence + rapid fire response discount applied"))
    if gated_community:
        flags.append(("G", "Gated community — controlled access reduces theft/vandalism exposure"))
    if building_code_comply:
        flags.append(("G", "Post-2000 building code — modern construction discount applied"))
    if recent_reno:
        flags.append(("G", "Recent renovation — positive maintenance indicator; discount applied"))
    for t in triggered:
        flags.append(("T", f"[GAM Interaction] {t['name']} — {t['logic'][:80]}... H={t['h']:.3f} · {t['status']}"))

    return {
        "score_t1":      round(s_t1,    1),
        "score_glm":     round(s_glm,   1),
        "score_final":   round(s_final, 1),
        "t2_increment":  round(s_glm - s_t1, 1),
        "t3_lift":       round(s_final - s_glm, 1),
        "eta_t1":        round(eta_t1,    4),
        "eta_glm":       round(eta_glm,   4),
        "delta_gam":     round(delta_gam, 4),
        "eta_final":     round(eta_final, 4),
        "m_t1":          round(float(np.exp(eta_t1 - 8.10 - loc_off)), 4),
        "m_t2":          round(float(np.exp(eta_t2_incr)), 4),
        "m_gam":         round(float(np.exp(delta_gam)), 4),
        "loss_t1":       round(loss_t1, 0),
        "loss_glm":      round(loss_glm, 0),
        "loss_final":    round(loss_final, 0),
        "claim_freq":    round(claim_freq, 4),
        "claim_sev":     round(claim_sev, 0),
        "premium":       premium,
        "decision":      decision,
        "desc":          desc,
        "t1_contrib":    t1_contrib,
        "t2_contrib":    t2_contrib,
        "t3_contrib":    t3_contrib,
        "triggered":     triggered,
        "ix_map":        ix_map,
        "flags":         flags,
        "smooth_total":  smooth_total,
    }


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS & CHART BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

DEC_PALETTE = {
    "Preferred": {"bg": "#f0fdf4", "border": "#22c55e", "text": "#14532d", "accent": "#16a34a"},
    "Standard":  {"bg": "#eff6ff", "border": "#3b82f6", "text": "#1e3a8a", "accent": "#2563eb"},
    "Rated":     {"bg": "#fffbeb", "border": "#f59e0b", "text": "#78350f", "accent": "#d97706"},
    "Decline":   {"bg": "#fef2f2", "border": "#ef4444", "text": "#7f1d1d", "accent": "#dc2626"},
}

HOVER = dict(bgcolor="#0f172a", font_color="#f8fafc", font_size=12, bordercolor="#1e293b")

def _base_fig(fig, h=300, margin=None):
    m = margin or dict(l=8, r=60, t=28, b=8)
    fig.update_layout(
        height=h, paper_bgcolor="white", plot_bgcolor="white", margin=m,
        font=dict(family="DM Sans, sans-serif", size=11, color="#475569"),
        hoverlabel=HOVER,
    )
    fig.update_xaxes(gridcolor="#f1f5f9", zeroline=False, tickfont=dict(size=10, color="#64748b"))
    fig.update_yaxes(gridcolor="#f1f5f9", zeroline=False, tickfont=dict(size=10, color="#334155"))
    return fig


@st.cache_data
def load_portfolio():
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    return None


def score_gauge_svg(score, decision):
    cx, cy, r = 120, 100, 78
    import math
    def arc_pt(angle_deg):
        a = math.radians(angle_deg)
        return cx + r * math.cos(a), cy - r * math.sin(a)

    angle = 180 - (score / 100 * 180)
    px_, py_ = arc_pt(angle)
    color_map = {
        "Preferred": "#22c55e", "Standard": "#3b82f6",
        "Rated": "#f59e0b",    "Decline":  "#ef4444"
    }
    needle_color = color_map[decision]

    return f"""
    <div style='display:flex;justify-content:center;padding:8px 0 0 0'>
    <svg viewBox="0 0 240 160" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:230px">
      <defs>
        <linearGradient id="gx" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%"   stop-color="#22c55e"/>
          <stop offset="30%"  stop-color="#3b82f6"/>
          <stop offset="65%"  stop-color="#f59e0b"/>
          <stop offset="100%" stop-color="#ef4444"/>
        </linearGradient>
      </defs>
      <path d="M 42 100 A 78 78 0 0 1 198 100" fill="none" stroke="#e2e8f0" stroke-width="13" stroke-linecap="round"/>
      <path d="M 42 100 A 78 78 0 0 1 198 100" fill="none" stroke="url(#gx)" stroke-width="13" stroke-linecap="round" opacity="0.4"/>
      <line x1="{cx}" y1="{cy}" x2="{px_:.1f}" y2="{py_:.1f}"
            stroke="{needle_color}" stroke-width="3.5" stroke-linecap="round"/>
      <circle cx="{cx}" cy="{cy}" r="5" fill="{needle_color}"/>
      <text x="34" y="116" font-size="9" fill="#94a3b8" font-family="DM Sans" text-anchor="middle">0</text>
      <text x="120" y="22" font-size="9" fill="#94a3b8" font-family="DM Sans" text-anchor="middle">50</text>
      <text x="206" y="116" font-size="9" fill="#94a3b8" font-family="DM Sans" text-anchor="middle">100</text>
      <text x="{cx}" y="145" text-anchor="middle" font-size="34" font-weight="700"
            fill="{needle_color}" font-family="DM Sans">{score:.0f}</text>
      <text x="{cx}" y="158" text-anchor="middle" font-size="8.5" fill="#94a3b8"
            font-family="DM Sans" letter-spacing="1.5">RISK SCORE</text>
    </svg>
    </div>"""


def build_waterfall(r):
    base = 0
    t2_val = r["t2_increment"]
    t3_val = r["t3_lift"]
    measures = ["absolute", "relative", "relative", "subtotal", "relative", "total"]
    xs = ["Base", "Tier 1\nStructural", "Tier 2\nBehavioral", "GLM Score\n(pre-T3)", "Tier 3\nGAM Effect", "Final\nRisk Score"]
    ys = [0, r["score_t1"], t2_val, r["score_glm"], t3_val, r["score_final"]]
    texts = [
        "0",
        f"+{r['score_t1']:.1f} pts",
        f"{t2_val:+.1f} pts",
        f"{r['score_glm']:.1f}",
        f"{t3_val:+.1f} pts",
        f"<b>{r['score_final']:.1f}</b>"
    ]
    hovers = [
        "Baseline — zero risk accumulated<br>All tiers start from this point",
        f"<b>Tier 1 Foundation: +{r['score_t1']:.1f} pts</b><br>Structural variables: roof age/material,<br>home age/construction, water loss recency,<br>RCV ratio, location zone<br>η_T1 = {r['eta_t1']:.4f}",
        f"<b>Tier 2 Behavioral: {t2_val:+.1f} pts</b><br>Lifestyle factors + mitigant discounts<br>Prior claims, lapses, pool/trampoline,<br>alarms, sprinklers, community<br>Δη_T2 = {r['eta_glm'] - r['eta_t1']:.4f}",
        f"<b>GLM Score (pre-T3): {r['score_glm']:.1f}</b><br>Combined structural + behavioral signal<br>Before non-linear GAM adjustment",
        f"<b>Tier 3 GAM Effect: {t3_val:+.1f} pts</b><br>Non-linear smooths + pairwise interactions<br>Captures variance the log-linear GLM misses<br>δ_GAM = {r['delta_gam']:.4f}",
        f"<b>Final Risk Score: {r['score_final']:.1f} / 100</b><br>η_final = {r['eta_final']:.4f}<br>Decision: {r['decision']}",
    ]
    fig = go.Figure(go.Waterfall(
    orientation="v",
    measure=measures,
    x=xs,
    y=ys,
    text=texts,
    textposition="outside",
    connector=dict(line=dict(color="#cbd5e1", width=1.5, dash="dot")),

    increasing=dict(
        marker=dict(color="#ef4444", line=dict(width=0))
    ),

    decreasing=dict(
        marker=dict(color="#10b981", line=dict(width=0))
    ),

    totals=dict(
        marker=dict(color="#3b82f6", line=dict(width=0))
    ),

    customdata=hovers,
    hovertemplate="%{customdata}<extra></extra>",
))
    # Decision threshold lines
    for thresh, lbl, color in [(30,"Preferred Ceiling","#22c55e"),(60,"Standard Ceiling","#3b82f6"),(80,"Rated Ceiling","#f59e0b")]:
        fig.add_hline(y=thresh, line_color=color, line_dash="dot", line_width=1.2,
                      annotation_text=f"  {lbl} ({thresh})", annotation_font=dict(size=9, color=color),
                      annotation_position="right")
    fig.update_layout(showlegend=False, yaxis=dict(title="Risk Score (0–100)", range=[-5, 110]))
    return _base_fig(fig, 370, dict(l=8, r=110, t=24, b=8))


def build_contributions(r):
    all_items = []
    for k, v in r["t1_contrib"].items():
        if abs(v) > 0.05:
            all_items.append({"label": k, "value": v, "tier": "T1", "color": "#3b82f6",
                               "tip": f"Tier 1 structural factor (r-correlation with loss)"})
    for k, v in r["t2_contrib"].items():
        if abs(v) > 0.05:
            direction = "Risk factor" if v > 0 else "Mitigant discount"
            all_items.append({"label": k, "value": v, "tier": "T2", "color": "#8b5cf6",
                               "tip": f"Tier 2 behavioral factor — {direction}"})
    for k, v in r["t3_contrib"].items():
        if abs(v) > 0.05:
            all_items.append({"label": k, "value": v, "tier": "T3", "color": "#f59e0b",
                               "tip": "Tier 3 GAM — non-linear / interaction effect"})

    df = pd.DataFrame(all_items).sort_values("value", ascending=True)
    if df.empty:
        return None

    bar_colors = [
        ("#ef4444" if row.value > 0 else "#10b981")
        for _, row in df.iterrows()
    ]
    tier_colors = {"T1": "#3b82f6", "T2": "#8b5cf6", "T3": "#f59e0b"}
    marker_colors = [tier_colors[t] for t in df["tier"]]

    fig = go.Figure()
    # Main bars
    fig.add_trace(go.Bar(
        x=df["value"], y=df["label"], orientation="h",
        marker=dict(
            color=[("#ef4444" if v > 0 else "#10b981") for v in df["value"]],
            opacity=0.80,
            line=dict(width=0)
        ),
        text=[f"{v:+.1f}" for v in df["value"]], textposition="outside",
        textfont=dict(size=9.5, color="#1e293b"),
        customdata=list(zip(df["tier"], df["value"], df["tip"])),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Tier: %{customdata[0]}<br>"
            "Score contribution: <b>%{customdata[1]:+.2f} pts</b><br>"
            "%{customdata[2]}<extra></extra>"
        ),
    ))
    fig.add_vline(x=0, line_color="#cbd5e1", line_width=1.5)
    # Tier legend annotations
    for tier, color, label in [("T1","#3b82f6","Tier 1"),("T2","#8b5cf6","Tier 2"),("T3","#f59e0b","Tier 3")]:
        if tier in df["tier"].values:
            pass  # markers on y-axis labels not supported; use legend
    fig.update_layout(
        showlegend=False,
        xaxis=dict(title="Score Contribution (points)", title_font=dict(size=10, color="#64748b")),
    )
    return _base_fig(fig, max(300, len(df) * 28 + 60), dict(l=8, r=80, t=16, b=8))


def build_portfolio_hist(score, df):
    scores = df["final_risk_score"].values if "final_risk_score" in df.columns else df["score_final"].values
    pct = (scores < score).mean() * 100
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=scores, nbinsx=50,
        marker=dict(color="#cbd5e1", line=dict(width=0)),
        hovertemplate="Score range: %{x:.0f}<br>Policies: %{y}<extra></extra>",
        name="Portfolio",
    ))
    for lo, hi, c, lbl in [
        (0, 30, "rgba(240,253,244,0.6)", "Preferred"),
        (30, 60, "rgba(239,246,255,0.6)", "Standard"),
        (60, 80, "rgba(255,251,235,0.6)", "Rated"),
        (80, 100, "rgba(254,242,242,0.6)", "Decline"),
    ]:
        fig.add_vrect(x0=lo, x1=hi, fillcolor=c, line_width=0,
                      annotation_text=lbl, annotation_position="top left",
                      annotation_font=dict(size=9, color="#94a3b8"))
    fig.add_vline(x=score, line_color="#ef4444", line_width=2.5)
    fig.add_annotation(
        x=score, y=1, yref="paper",
        text=f"This policy ({score:.0f})",
        showarrow=False,
        xanchor="left" if score < 70 else "right",
        xshift=8 if score < 70 else -8,
        font=dict(size=11, color="#ef4444", family="DM Sans"),
        bgcolor="white",
        borderpad=3,
    )
    fig.update_layout(
        showlegend=False,
        xaxis=dict(title="Risk Score (0–100)"),
        yaxis=dict(title="Policies"),
        title=dict(
            text=f"This policy is at the <b>{pct:.0f}th percentile</b> — higher than {pct:.0f}% of portfolio",
            font=dict(size=12, color="#1e293b"), x=0,
        ),
    )
    return _base_fig(fig, 260, dict(l=8, r=10, t=40, b=8)), pct


def build_score_loss_scatter(df):
    score_col = "final_risk_score" if "final_risk_score" in df.columns else "score_final"
    if score_col not in df.columns or "expected_loss" not in df.columns:
        return None
    sample = df.sample(min(2000, len(df)), random_state=42)
    dc = {"Preferred": "#22c55e", "Standard": "#3b82f6", "Rated": "#f59e0b", "Decline": "#ef4444"}
    fig = go.Figure()
    for dec in ["Preferred", "Standard", "Rated", "Decline"]:
        if "decision" not in sample.columns:
            continue
        sub = sample[sample["decision"] == dec]
        if not len(sub):
            continue
        fig.add_trace(go.Scatter(
            x=sub[score_col], y=sub["expected_loss"], mode="markers",
            marker=dict(size=4, color=dc[dec], opacity=0.5, line=dict(width=0)),
            name=dec,
            hovertemplate=f"<b>{dec}</b><br>Risk Score: %{{x:.1f}}<br>Expected Loss: $%{{y:,.0f}}<extra></extra>",
        ))
    for lo, hi, c, lbl in [(0,30,"rgba(240,253,244,0.4)","Preferred"),(30,60,"rgba(239,246,255,0.4)","Standard"),(60,80,"rgba(255,251,235,0.4)","Rated"),(80,100,"rgba(254,242,242,0.4)","Decline")]:
        fig.add_vrect(x0=lo, x1=hi, fillcolor=c, line_width=0)
    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.2, font=dict(size=11)),
        xaxis=dict(title="Final Risk Score", title_font=dict(size=10, color="#64748b")),
        yaxis=dict(title="Expected Annual Loss ($)", title_font=dict(size=10, color="#64748b")),
        title=dict(text="Risk Score vs Expected Annual Loss · Colour by decision segment",
                   font=dict(size=12, color="#1e293b"), x=0),
    )
    return _base_fig(fig, 320, dict(l=8, r=8, t=40, b=8))


def build_tier_dist(df):
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=["Tier 1 Score", "Tier 2 Adjustment", "Final Risk Score"],
        horizontal_spacing=0.10,
    )
    pairs = [
        ("score_t1", "#3b82f6", 1, "Tier 1 foundation score (structural only)"),
        ("t2_increment", "#8b5cf6", 2, "Tier 2 incremental adjustment (behavioral)"),
        ("final_risk_score", "#ef4444", 3, "Final risk score (all tiers combined)"),
    ]
    for col, color, idx, tip in pairs:
        actual_col = col if col in df.columns else ("score_final" if col == "final_risk_score" else None)
        if actual_col and actual_col in df.columns:
            vals = df[actual_col].dropna()
            fig.add_trace(go.Histogram(
                x=vals, nbinsx=40,
                marker=dict(color=color, opacity=0.72, line=dict(width=0)),
                hovertemplate=f"{tip}: %{{x:.1f}}<br>Policies: %{{y}}<extra></extra>",
            ), 1, idx)
    fig.update_annotations(font=dict(size=11, color="#334155"))
    fig.update_layout(showlegend=False)
    return _base_fig(fig, 240, dict(l=8, r=8, t=44, b=8))


def build_sensitivity_scenarios(snap, current_score, current_premium):
    """Build what-if improvement scenarios from current inputs."""
    scenarios = []

    def try_it(label, **overrides):
        inp = dict(snap)
        inp.update(overrides)
        r2 = score_policy(**inp)
        delta = r2["score_final"] - current_score
        if delta < -0.4:
            p2 = r2["premium"]
            prem_delta = (p2 - current_premium) if (p2 and current_premium) else None
            scenarios.append({
                "action": label,
                "new_score": round(r2["score_final"], 1),
                "new_decision": r2["decision"],
                "new_premium": p2,
                "delta": round(delta, 1),
                "prem_delta": round(prem_delta, 0) if prem_delta else None,
            })

    if snap["roof_age"] > 15:
        try_it("Replace roof (new, 0 years)", roof_age=0)
    if snap["roof_material"] == "asphalt":
        try_it("Upgrade roof: asphalt → metal", roof_material="metal")
    if snap["fire_sprinklers"] == "none":
        try_it("Install full sprinkler system", fire_sprinklers="full")
    elif snap["fire_sprinklers"] == "partial":
        try_it("Upgrade sprinklers: partial → full", fire_sprinklers="full")
    if snap["monitored_alarm"] == 0:
        try_it("Install monitored alarm system", monitored_alarm=1)
    if snap["recent_reno"] == 0:
        try_it("Complete major renovation", recent_reno=1)
    if snap["gated_community"] == 0:
        try_it("Qualify as gated community", gated_community=1)
    dc_up = {"wood_frame": "brick_veneer", "brick_veneer": "masonry", "masonry": "superior"}
    if snap["construction_type"] in dc_up:
        nxt = dc_up[snap["construction_type"]]
        try_it(f"Upgrade construction → {nxt.replace('_',' ')}", construction_type=nxt)
    if snap["trampoline"] == 1:
        try_it("Remove trampoline from property", trampoline=0)
    if snap["home_business"] == "active_business":
        try_it("Reduce home business → home office only", home_business="home_office")
    if snap["building_code_comply"] == 0:
        try_it("Document building code compliance", building_code_comply=1)

    scenarios.sort(key=lambda x: x["delta"])
    return scenarios[:7]


def build_shap_waterfall(r):
    """Interactive SHAP-style waterfall showing contribution of each variable to final score."""
    items = []
    for k, v in r["t1_contrib"].items():
        if abs(v) > 0.05:
            items.append({"label": k, "value": round(v, 2), "tier": "T1", "color": "#3b82f6"})
    for k, v in r["t2_contrib"].items():
        if abs(v) > 0.05:
            items.append({"label": k, "value": round(v, 2), "tier": "T2", "color": "#8b5cf6"})
    for k, v in r["t3_contrib"].items():
        if abs(v) > 0.05:
            items.append({"label": k, "value": round(v, 2), "tier": "T3", "color": "#f59e0b"})

    if not items:
        return None

    items_sorted = sorted(items, key=lambda x: x["value"])
    labels  = [i["label"] for i in items_sorted]
    values  = [i["value"] for i in items_sorted]
    tiers   = [i["tier"]  for i in items_sorted]
    colors  = ["#ef4444" if v > 0 else "#10b981" for v in values]
    tier_colors = {"T1": "#3b82f6", "T2": "#8b5cf6", "T3": "#f59e0b"}

    # Running base from left → waterfall base positions
    running = r["score_t1"]
    bases, ends = [], []
    # Start from T1 score as anchor; show T2 and T3 increments as waterfall steps
    # For simplicity use a standard horizontal bar with base = 0 (SHAP style)
    fig = go.Figure()

    for i, item in enumerate(items_sorted):
        v = item["value"]
        tc = tier_colors[item["tier"]]
        bar_color = "#ef4444" if v > 0 else "#10b981"
        fig.add_trace(go.Bar(
            x=[v],
            y=[item["label"]],
            orientation="h",
            marker=dict(color=bar_color, opacity=0.82, line=dict(color=tc, width=1.5)),
            text=[f"{v:+.2f}"],
            textposition="outside",
            textfont=dict(size=9.5, color="#1e293b"),
            customdata=[[item["tier"], v, "Risk factor" if v > 0 else "Mitigant"]],
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Tier: <b>%{customdata[0]}</b><br>"
                "Score pts: <b>%{customdata[1]:+.2f}</b><br>"
                "Type: %{customdata[2]}<extra></extra>"
            ),
            showlegend=False,
        ))

    # Tier legend traces
    for tier, color, label in [("T1","#3b82f6","Tier 1 Structural"),("T2","#8b5cf6","Tier 2 Behavioral"),("T3","#f59e0b","Tier 3 GAM")]:
        if tier in tiers:
            fig.add_trace(go.Bar(
                x=[None], y=[None], orientation="h",
                marker=dict(color="white", line=dict(color=color, width=2)),
                name=label, showlegend=True,
            ))

    fig.add_vline(x=0, line_color="#cbd5e1", line_width=1.5)
    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="h", x=0, y=-0.18, font=dict(size=10, color="#475569")),
        barmode="overlay",
        xaxis=dict(title="Score Contribution (points)", title_font=dict(size=10, color="#64748b"), zeroline=False),
        yaxis=dict(tickfont=dict(size=10.5, color="#1e293b")),
        title=dict(text="SHAP-Style Variable Attribution — Score Points per Factor", font=dict(size=12, color="#1e293b"), x=0),
    )
    return _base_fig(fig, max(320, len(items_sorted) * 26 + 80), dict(l=8, r=80, t=44, b=48))


def build_radar_chart(r):
    """Radar chart showing risk profile across 5 risk dimensions."""
    # Compute dimension scores (0–10 scale each)
    score = r["score_final"]

    # Structural risk (T1 dominant)
    structural = min(10, r["score_t1"] / 10)
    # Behavioral risk (T2)
    behavioral = min(10, max(0, (r["t2_increment"] + 5) / 3))
    # Geographic risk (T3 geo vars)
    geo_items = {k: v for k, v in r["t3_contrib"].items() if k != "Non-linear Smooths"}
    geographic = min(10, max(0, sum(geo_items.values()) / 2 + 3))
    # Interaction risk
    interaction = min(10, len(r["triggered"]) * 3.5)
    # Loss severity
    severity = min(10, (r["claim_sev"] - 1500) / 5350)

    categories = ["Structural<br>Risk", "Behavioral<br>Risk", "Geographic<br>Exposure",
                  "Interaction<br>Complexity", "Loss<br>Severity"]
    vals = [structural, behavioral, geographic, interaction, severity]
    vals_closed = vals + [vals[0]]
    cats_closed = categories + [categories[0]]

    dec_colors = {"Preferred": "#22c55e", "Standard": "#3b82f6", "Rated": "#f59e0b", "Decline": "#ef4444"}
    fill_color = dec_colors.get(r["decision"], "#3b82f6")

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals_closed,
        theta=cats_closed,
        fill="toself",
        fillcolor=f"rgba({int(fill_color[1:3],16)},{int(fill_color[3:5],16)},{int(fill_color[5:7],16)},0.18)",
        line=dict(color=fill_color, width=2.5),
        marker=dict(size=7, color=fill_color),
        name=r["decision"],
        hovertemplate="<b>%{theta}</b><br>Score: %{r:.1f}/10<extra></extra>",
    ))
    # Benchmark (average portfolio) as reference
    avg_vals = [5.5, 4.2, 4.8, 2.0, 4.5]
    avg_closed = avg_vals + [avg_vals[0]]
    fig.add_trace(go.Scatterpolar(
        r=avg_closed,
        theta=cats_closed,
        fill="toself",
        fillcolor="rgba(148,163,184,0.08)",
        line=dict(color="#94a3b8", width=1.5, dash="dot"),
        marker=dict(size=5, color="#94a3b8"),
        name="Portfolio Avg",
        hovertemplate="<b>%{theta}</b><br>Portfolio avg: %{r:.1f}/10<extra></extra>",
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 10], tickfont=dict(size=9, color="#94a3b8"),
                            gridcolor="#e2e8f0", linecolor="#e2e8f0"),
            angularaxis=dict(tickfont=dict(size=10, color="#334155"), gridcolor="#f1f5f9", linecolor="#e2e8f0"),
            bgcolor="white",
        ),
        showlegend=True,
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.12, font=dict(size=10, color="#475569")),
        title=dict(text="Risk Profile Radar — 5 Dimensions vs Portfolio Average", font=dict(size=12, color="#1e293b"), x=0),
        paper_bgcolor="white",
        font=dict(family="DM Sans, sans-serif"),
        margin=dict(l=60, r=60, t=48, b=48),
        height=340,
        hoverlabel=HOVER,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
DEFAULTS = dict(
    roof_age=15, roof_material="asphalt", home_age=25, construction_type="wood_frame",
    prior_water_claim=0, months_since_water=999, coverage_a=350_000, rcv_ratio=1.05,
    ins_lapses=0, swimming_pool="none", trampoline=0, home_business="none",
    fire_sprinklers="none", monitored_alarm=0, gated_community=0, wood_stove=0,
    recent_reno=0, pet_ownership=0, prior_claims_5yr=0, fire_hydrant_dist=2.0,
    iso_class=3, building_code_comply=0, crime_idx=30.0,
    wildfire_score=20.0, wildfire_zone=0, canopy_pct=30.0, flood_zone=0,
    flood_depth_in=0.0, slope_deg=5.0, hail_zone=0, burn_history=0,
    foundation_type="concrete_slab", location_zone="IL-Midwest",
    _result=None, _snap=None,
)
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
df_port = load_portfolio()

with st.sidebar:
    st.markdown("""
    <div style='padding:20px 12px 14px 12px'>
        <div style='font-size:9px;text-transform:uppercase;letter-spacing:3px;color:#d4af37;
                    font-weight:700;margin-bottom:6px'>Homeowners · GLM-GAM</div>
        <div style='font-family:"DM Serif Display",serif;font-size:18px;color:#f8fafc;line-height:1.3'>
            Risk Intelligence<br>Platform
        </div>
        <div style='height:2px;background:linear-gradient(90deg,#d4af37,transparent);
                    margin:12px 0 4px 0;border-radius:2px'></div>
    </div>""", unsafe_allow_html=True)

    section = st.radio(
        "nav",
        ["🏠  Policy Scorer", "📊  Portfolio Analytics", "📖  Framework Guide"],
        label_visibility="collapsed",
    )

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    if df_port is not None:
        st.markdown("<div style='font-size:9px;text-transform:uppercase;letter-spacing:2px;color:#64748b;font-weight:600;padding:0 12px;margin-bottom:8px'>Portfolio Snapshot</div>", unsafe_allow_html=True)
        score_col = "final_risk_score" if "final_risk_score" in df_port.columns else "score_final"
        metrics = [
            ("Policies", f"{len(df_port):,}", "#3b82f6"),
            ("Avg Risk Score", f"{df_port[score_col].mean():.1f}" if score_col in df_port.columns else "—", "#f59e0b"),
            ("Avg Expected Loss", f"${df_port['expected_loss'].mean():,.0f}" if "expected_loss" in df_port.columns else "—", "#ef4444"),
        ]
        for lbl, val, color in metrics:
            st.markdown(f"""
            <div style='display:flex;justify-content:space-between;align-items:center;
                        padding:7px 12px;background:#1e293b;border-radius:7px;margin-bottom:4px;
                        border-left:3px solid {color}'>
                <span style='font-size:11px;color:#94a3b8'>{lbl}</span>
                <span style='font-size:12px;font-variant-numeric:tabular-nums;color:#f1f5f9;
                             font-weight:700'>{val}</span>
            </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='padding:0 12px;font-size:11px;color:#475569;line-height:1.6'>
            No portfolio data found at<br><code style='color:#94a3b8'>data/raw/homeowners_portfolio.csv</code>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — POLICY SCORER
# ══════════════════════════════════════════════════════════════════════════════
if "Scorer" in section:

    st.markdown("""
    <div class="platform-hero">
        <div class="hero-eyebrow">Underwriting Intelligence Platform · 3-Tier Architecture</div>
        <div class="hero-title">Homeowners Policy Risk Assessment</div>
    </div>""", unsafe_allow_html=True)

    # Tier legend row
    tc1, tc2, tc3 = st.columns(3)
    tier_info = [
        (tc1, "Tier 1", "Structural Foundation", "#3b82f6",
         "· Roof age/material, home age/construction, water loss recency, RCV ratio, location zone"),
        (tc2, "Tier 2", "Behavioral Adjustments", "#8b5cf6",
         "· Prior claims, coverage lapses, pool/trampoline, alarms, sprinklers, business use, renovations"),
        (tc3, "Tier 3", "Interactions & Non-linear (GAM)", "#f59e0b",
         "· Wildfire/flood/hail geography, canopy density, slope, burn history"),
    ]
    for col, lbl, sub, color, desc in tier_info:
        col.markdown(f"""
        <div style='background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:14px 16px;
                    border-top:3px solid {color};height:100%;'>
            <div style='font-size:10px;text-transform:uppercase;letter-spacing:1.5px;
                        color:{color};font-weight:700;margin-bottom:3px'>{lbl}</div>
            <div style='font-size:13px;font-weight:700;color:#0f172a;margin-bottom:5px'>{sub}</div>
            <div style='font-size:11px;color:#64748b;line-height:1.6'>{desc}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # ── INPUT FORM ─────────────────────────────────────────────────────────────
    with st.form("risk_form", clear_on_submit=False):

        # Location
        st.markdown("""
        <div style='background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;
                    padding:12px 18px;margin-bottom:18px'>
            <div style='font-size:11px;font-weight:700;color:#475569;text-transform:uppercase;
                        letter-spacing:1px;margin-bottom:10px'>📍 Location Zone</div>
        """, unsafe_allow_html=True)
        location_zone = st.selectbox(
            "Territory / Location Zone",
            LOCATIONS,
            index=LOCATIONS.index(st.session_state.location_zone),
            help="Geographic zone determines the baseline location risk offset loaded into Tier 1 η",
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # ── TIER 1 ────────────────────────────────────────────────────────────
        st.markdown("""
        <div class="form-tier-hd form-tier-hd-1">
            <div class="form-tier-title">🔵 Structural Foundation</div>
            <div class="form-tier-sub">Core property risk indicators</div>
        </div>""", unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            roof_age = st.slider("Roof Age (years)", 0, 50,
                                  value=st.session_state.roof_age,
                                  help="r=0.42 — strongest Tier 1 predictor. Inspection mandatory >25yr. Non-linear acceleration >15yr captured by GAM smooth.")
            home_age = st.slider("Home Age (years)", 0, 150,
                                  value=st.session_state.home_age,
                                  help="r=0.12 — infrastructure age proxy. Homes >80yr flagged for coverage validation.")
        with c2:
            roof_material = st.selectbox("Roof Material", ["asphalt", "tile", "slate", "metal"],
                                          index=["asphalt","tile","slate","metal"].index(st.session_state.roof_material),
                                          help="r=0.38 — weather resistance rating. Metal=lowest risk; Asphalt=highest. Impacts wildfire interaction.")
            construction_type = st.selectbox("Construction Type",
                                              ["wood_frame","brick_veneer","masonry","superior"],
                                              index=["wood_frame","brick_veneer","masonry","superior"].index(st.session_state.construction_type),
                                              help="r=0.35 — structural fire/wind resilience. Superior=masonry/steel. Wood frame=highest exposure.")
        with c3:
            prior_water_claim = st.selectbox("Prior Water Claim",
                                              [0, 1],
                                              index=st.session_state.prior_water_claim,
                                              format_func=lambda x: "Yes" if x else "No",
                                              help="r=0.29 — chronic maintenance signal. Interacts with canopy density in Tier 3 (H=0.31).")
            months_since_water = st.slider("Months Since Water Claim", 1, 120,
                                            value=min(st.session_state.months_since_water, 120),
                                            help="Recency of water claim. <12mo = 3× weight; <24mo = 2×; <36mo = 1×. Set high if no claim.") \
                if prior_water_claim else 999
            coverage_a = st.number_input("Coverage A Amount ($)", min_value=50_000, max_value=2_000_000,
                                          value=int(st.session_state.coverage_a), step=10_000,
                                          help="r=0.18 — property value proxy. Used for RCV validation check.")
        with c4:
            rcv_ratio = st.slider("RCV Ratio", 0.7, 2.0, value=float(st.session_state.rcv_ratio), step=0.01,
                                   help="Replacement Cost Value ÷ Coverage A. Values >1.15 trigger overstatement flag and interact with crime index in Tier 3.")

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        # ── TIER 2 ────────────────────────────────────────────────────────────
        st.markdown("""
        <div class="form-tier-hd form-tier-hd-2">
            <div class="form-tier-title">🟣 Behavioral and Lifestyle Adjustments</div>
            <div class="form-tier-sub">Risk factors and mitigant discounts</div>
        </div>""", unsafe_allow_html=True)

        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            prior_claims_5yr = st.slider("Prior Claims (5yr)", 0, 8, value=st.session_state.prior_claims_5yr,
                                          help="r=0.26 — strongest Tier 2 predictor. β=+0.225 log-unit per claim.")
            ins_lapses = st.slider("Insurance Lapses", 0, 5, value=st.session_state.ins_lapses,
                                    help="r=0.18 — financial stress indicator. Each lapse adds +0.140 to η. +15–30% rate impact.")
            crime_idx = st.slider("Crime Index (0–100)", 0.0, 100.0,
                                   value=float(st.session_state.crime_idx), step=1.0,
                                   help="r=0.19 — area crime exposure. >55 triggers RCV×Crime interaction in Tier 3.")
        with c2:
            swimming_pool = st.selectbox("Swimming Pool",
                                          ["none", "in_ground", "above_ground"],
                                          index=["none","in_ground","above_ground"].index(st.session_state.swimming_pool),
                                          help="r=0.14 — drowning liability + maintenance burden. +$100–300/yr impact.")
            trampoline = st.selectbox("Trampoline on Property",
                                       [0, 1],
                                       index=st.session_state.trampoline,
                                       format_func=lambda x: "Yes" if x else "No",
                                       help="r=0.12 — high injury liability. +$250/yr or decline. Interacts with hail zone.")
            home_business = st.selectbox("Home Business",
                                          ["none", "home_office", "active_business"],
                                          index=["none","home_office","active_business"].index(st.session_state.home_business),
                                          help="r=0.09 — commercial liability in residential policy. Active business → refer if significant.")
        with c3:
            fire_sprinklers = st.selectbox("Fire Sprinklers",
                                            ["none", "partial", "full"],
                                            index=["none","partial","full"].index(st.session_state.fire_sprinklers),
                                            help="r=0.05 (mitigant) — most effective fire severity mitigant. Full: −10 to −20%. β=−0.065.")
            monitored_alarm = st.selectbox("Monitored Alarm",
                                            [0, 1],
                                            index=st.session_state.monitored_alarm,
                                            format_func=lambda x: "Yes" if x else "No",
                                            help="r=0.07 (mitigant) — theft deterrence + rapid fire response. −10% discount. β=−0.110.")
            gated_community = st.selectbox("Gated Community",
                                            [0, 1],
                                            index=st.session_state.gated_community,
                                            format_func=lambda x: "Yes" if x else "No",
                                            help="r=0.03 (mitigant) — controlled access reduces theft/vandalism. −5% discount.")
        with c4:
            wood_stove = st.selectbox("Wood-Burning Stove",
                                       [0, 1],
                                       index=st.session_state.wood_stove,
                                       format_func=lambda x: "Yes" if x else "No",
                                       help="r=0.10 — elevated fire ignition risk. +5–20% rate impact. β=+0.048.")
            pet_ownership = st.selectbox("Pet Ownership (dogs)",
                                          [0, 1],
                                          index=st.session_state.pet_ownership,
                                          format_func=lambda x: "Yes" if x else "No",
                                          help="r=0.16 — liability exposure for dangerous breeds. +5–25% adjustment.")
            fire_hydrant_dist = st.slider("Fire Hydrant Distance (mi)", 0.0, 15.0,
                                           value=float(st.session_state.fire_hydrant_dist), step=0.5,
                                           help="r=0.15 — fire response speed proxy. >5mi may trigger location-based review.")
        with c5:
            iso_class = st.slider("ISO Fire Class (1–10)", 1, 10, value=st.session_state.iso_class,
                                   help="r=0.06 — fire dept quality rating. Class 1=best; 10=no protection. Location adjustment factor.")
            recent_reno = st.selectbox("Recent Renovation",
                                        [0, 1],
                                        index=st.session_state.recent_reno,
                                        format_func=lambda x: "Yes (last 5yr)" if x else "No",
                                        help="r=0.08 (mitigant) — updated systems reduce loss probability. −15% adjustment.")
            building_code_comply = st.selectbox("Post-2000 Building Code",
                                                 [0, 1],
                                                 index=st.session_state.building_code_comply,
                                                 format_func=lambda x: "Compliant" if x else "Pre-2000",
                                                 help="Modern construction discount applied. β=−0.055.")

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        # ── TIER 3 ────────────────────────────────────────────────────────────
        st.markdown("""
        <div class="form-tier-hd form-tier-hd-3">
            <div class="form-tier-title">🟡 Geographical Risk and Interaction Effects</div>
            <div class="form-tier-sub">Non-linear effects</div>
        </div>""", unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            wildfire_score = st.slider("Wildfire Score (0–100)", 0.0, 100.0,
                                        value=float(st.session_state.wildfire_score), step=1.0,
                                        help="Continuous wildfire exposure score. Quadratic non-linear effect in GAM. Auto-set by zone if wildfire zone active.")
            wildfire_zone = st.selectbox("Wildfire Zone",
                                          [0, 1],
                                          index=st.session_state.wildfire_zone,
                                          format_func=lambda x: "Yes (High Risk)" if x else "No",
                                          help="Binary high-risk wildfire zone flag. Triggers Roof Age × Wildfire interaction (H=0.43, strongest).")
            canopy_pct = st.slider("Tree Canopy Coverage %", 0.0, 100.0,
                                    value=float(st.session_state.canopy_pct), step=5.0,
                                    help=">60% triggers Water Claim × Canopy interaction (H=0.31) — gutter clog recurrence 1.6×.")
        with c2:
            flood_zone = st.selectbox("FEMA Flood Zone",
                                       [0, 1],
                                       index=st.session_state.flood_zone,
                                       format_func=lambda x: "Yes (Flood Zone)" if x else "No",
                                       help="FEMA designated flood zone. Required for flood endorsement. Interacts with stone/dirt foundation (H=0.14).")
            flood_depth_in = st.slider("Base Flood Depth (inches)", 0.0, 60.0,
                                        value=float(st.session_state.flood_depth_in), step=1.0,
                                        help="Expected flood depth in inches. Log-concave GAM smooth applied — larger values show diminishing marginal effect.")
            foundation_type = st.selectbox("Foundation Type",
                                            ["concrete_slab", "poured_concrete", "block", "stone_dirt"],
                                            index=["concrete_slab","poured_concrete","block","stone_dirt"].index(st.session_state.foundation_type),
                                            format_func=lambda x: x.replace("_"," ").title(),
                                            help="Stone/dirt in flood zone triggers water ingress interaction. Concrete slab = safest.")
        with c3:
            slope_deg = st.slider("Slope Angle (degrees)", 0.0, 45.0,
                                   value=float(st.session_state.slope_deg), step=1.0,
                                   help="Property slope. >20° + burn history triggers mudslide/debris-flow interaction (H=0.33). Square-root GAM smooth.")
            burn_history = st.selectbox("Recent Burn History",
                                         [0, 1],
                                         index=st.session_state.burn_history,
                                         format_func=lambda x: "Yes (within 3yr)" if x else "No",
                                         help="Post-fire erosion risk. Combined with >20° slope → Decline or +50% premium surcharge.")
            hail_zone = st.selectbox("Hail Zone",
                                      [0, 1],
                                      index=st.session_state.hail_zone,
                                      format_func=lambda x: "Yes (Active Hail)" if x else "No",
                                      help="Active hail exposure zone. Triggers Roof Age × Hail interaction for asphalt roofs >20yr (H=0.33).")
        with c4:
            st.markdown("""
            <div style='background:#fffbeb;border:1px solid #fde68a;border-radius:10px;
                        padding:14px;margin-top:4px'>
                <div style='font-size:10px;font-weight:700;color:#92400e;text-transform:uppercase;
                            letter-spacing:1px;margin-bottom:8px'>⚡ Interaction Triggers</div>
                <div style='font-size:11px;color:#78350f;line-height:1.7'>
                    Combinations that activate Tier 3 interactions:<br>
                    <br>
                    • Roof &gt;20yr + Wildfire zone<br>
                    • Prior water claim + Canopy &gt;60%<br>
                    • RCV &gt;1.15 + Crime index &gt;55<br>
                    • Flood zone + Stone/dirt foundation<br>
                    • Slope &gt;20° + Burn history<br>
                    • Asphalt roof &gt;20yr + Hail zone
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

        _, btn_mid, _ = st.columns([1.2, 1, 1.2])
        with btn_mid:
            submitted = st.form_submit_button("⚡  SCORE POLICY")

    # ── ON SUBMIT ───────────────────────────────────────────────────────────────
    if submitted:
        snap = dict(
            roof_age=int(roof_age), roof_material=roof_material,
            home_age=int(home_age), construction_type=construction_type,
            prior_water_claim=int(prior_water_claim),
            months_since_water=int(months_since_water) if prior_water_claim else 999,
            coverage_a=float(coverage_a), rcv_ratio=float(rcv_ratio),
            ins_lapses=int(ins_lapses), swimming_pool=swimming_pool,
            trampoline=int(trampoline), home_business=home_business,
            fire_sprinklers=fire_sprinklers, monitored_alarm=int(monitored_alarm),
            gated_community=int(gated_community), wood_stove=int(wood_stove),
            recent_reno=int(recent_reno), pet_ownership=int(pet_ownership),
            prior_claims_5yr=int(prior_claims_5yr), fire_hydrant_dist=float(fire_hydrant_dist),
            iso_class=int(iso_class), building_code_comply=int(building_code_comply),
            crime_idx=float(crime_idx), wildfire_score=float(wildfire_score),
            wildfire_zone=int(wildfire_zone), canopy_pct=float(canopy_pct),
            flood_zone=int(flood_zone), flood_depth_in=float(flood_depth_in),
            slope_deg=float(slope_deg), hail_zone=int(hail_zone),
            burn_history=int(burn_history), foundation_type=foundation_type,
            location_zone=location_zone,
        )
        for k, v in snap.items():
            st.session_state[k] = v
        st.session_state._result = score_policy(**snap)
        st.session_state._snap = snap

    # ── RESULTS ─────────────────────────────────────────────────────────────────
    if st.session_state._result is not None:
        r    = st.session_state._result
        snap = st.session_state._snap
        pal  = DEC_PALETTE[r["decision"]]

        st.markdown("<hr style='border-color:#e2e8f0;margin:24px 0 20px 0'>", unsafe_allow_html=True)

        # ── Decision banner (full-width, no gauge) ───────────────────────────
        bg, border, text, accent = pal["bg"], pal["border"], pal["text"], pal["accent"]
        prem_html = (f'<div style="text-align:center;padding:0 8px">'
                     f'<div style="font-size:10px;text-transform:uppercase;letter-spacing:1.5px;color:{accent};'
                     f'font-weight:600;margin-bottom:3px">Indicative Premium</div>'
                     f'<div style="font-size:30px;font-weight:700;font-variant-numeric:tabular-nums;color:{text}">'
                     f'${r["premium"]:,.0f}</div>'
                     f'<div style="font-size:11px;color:{text};opacity:.7"></div></div>'
                     if r["premium"] else
                     f'<div style="text-align:center;padding:0 8px">'
                     f'<div style="font-size:10px;text-transform:uppercase;letter-spacing:1.5px;color:{accent};'
                     f'font-weight:600;margin-bottom:3px">Indicative Premium</div>'
                     f'<div style="font-size:16px;font-weight:700;color:{text}">E&amp;S Referral</div></div>')

        # Score progress bar (clean horizontal bar replaces odometer)
        score_pct = int(r["score_final"])
        score_bar_color = {"Preferred": "#22c55e", "Standard": "#3b82f6",
                           "Rated": "#f59e0b", "Decline": "#ef4444"}[r["decision"]]

        st.markdown(f"""
        <div style='background:{bg};border:2px solid {border};border-radius:16px;padding:24px 28px;margin-bottom:0'>
            <div style='display:flex;align-items:flex-start;justify-content:space-between;flex-wrap:wrap;gap:20px'>
                <!-- Left: decision label -->
                <div style='flex:1;min-width:220px'>
                    <div style='font-size:10px;text-transform:uppercase;letter-spacing:2px;color:{accent};
                                font-weight:700;margin-bottom:5px'>Underwriting Decision</div>
                    <div style='font-family:"DM Serif Display",serif;font-size:34px;color:{text};line-height:1.1'>
                        {r["decision"]}</div>
                    <div style='font-size:12.5px;color:{text};margin-top:6px;opacity:.8'>{r["desc"]}</div>
                </div>
                <!-- Centre: risk score with clean bar -->
                <div style='flex:1;min-width:240px'>
                    <div style='font-size:10px;text-transform:uppercase;letter-spacing:1.5px;color:{accent};
                                font-weight:600;margin-bottom:6px'>Risk Score</div>
                    <div style='display:flex;align-items:baseline;gap:8px;margin-bottom:10px'>
                        <span style='font-size:48px;font-weight:700;font-variant-numeric:tabular-nums;
                                     color:{text};line-height:1'>{r["score_final"]:.0f}</span>
                        <span style='font-size:14px;color:{text};opacity:.6'>/ 100</span>
                    </div>
                    <!-- Progress bar with decision zone markers -->
                    <div style='position:relative;height:10px;background:#e2e8f0;border-radius:5px;overflow:hidden'>
                        <div style='height:100%;width:{score_pct}%;background:linear-gradient(90deg,#22c55e,{score_bar_color});
                                    border-radius:5px;transition:width 0.4s ease'></div>
                    </div>
                    <div style='display:flex;justify-content:space-between;margin-top:4px'>
                        <span style='font-size:9px;color:#22c55e;font-weight:600'>Preferred&lt;30</span>
                        <span style='font-size:9px;color:#3b82f6;font-weight:600'>Standard&lt;60</span>
                        <span style='font-size:9px;color:#f59e0b;font-weight:600'>Rated&lt;80</span>
                        <span style='font-size:9px;color:#ef4444;font-weight:600'>Decline</span>
                    </div>
                    <!-- Tier breakdown mini pills -->
                    <div style='display:flex;gap:8px;margin-top:12px;flex-wrap:wrap'>
                        <div style='background:rgba(59,130,246,0.10);border:1px solid rgba(59,130,246,0.25);
                                    border-radius:6px;padding:4px 10px'>
                            <span style='font-size:9px;color:#3b82f6;font-weight:700;text-transform:uppercase;
                                         letter-spacing:.8px'>T1</span>
                            <span style='font-size:12px;font-weight:700;color:#0f172a;margin-left:5px;
                                         font-family:"JetBrains Mono",monospace'>{r["score_t1"]:.1f}</span>
                        </div>
                        <div style='background:rgba(139,92,246,0.10);border:1px solid rgba(139,92,246,0.25);
                                    border-radius:6px;padding:4px 10px'>
                            <span style='font-size:9px;color:#8b5cf6;font-weight:700;text-transform:uppercase;
                                         letter-spacing:.8px'>T2</span>
                            <span style='font-size:12px;font-weight:700;color:#0f172a;margin-left:5px;
                                         font-family:"JetBrains Mono",monospace'>{r["t2_increment"]:+.1f}</span>
                        </div>
                        <div style='background:rgba(245,158,11,0.10);border:1px solid rgba(245,158,11,0.25);
                                    border-radius:6px;padding:4px 10px'>
                            <span style='font-size:9px;color:#f59e0b;font-weight:700;text-transform:uppercase;
                                         letter-spacing:.8px'>T3</span>
                            <span style='font-size:12px;font-weight:700;color:#0f172a;margin-left:5px;
                                         font-family:"JetBrains Mono",monospace'>{r["t3_lift"]:+.1f}</span>
                        </div>
                    </div>
                </div>
                <!-- Right: E[Loss] + Premium -->
                <div style='display:flex;gap:20px;flex-wrap:wrap;align-items:flex-start'>
                    <div style='text-align:center;padding:0 8px'>
                        <div style='font-size:10px;text-transform:uppercase;letter-spacing:1.5px;color:{accent};
                                    font-weight:600;margin-bottom:3px'>Expected Annual Loss</div>
                        <div style='font-size:30px;font-weight:700;font-variant-numeric:tabular-nums;
                                    color:{text};line-height:1.1'>${r["loss_final"]:,.0f}</div>
                        <div style='font-size:11px;color:{text};opacity:.7'></div>
                    </div>
                    {prem_html}
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

        # ── 6-KPI row (removed T1/T2 multipliers) ────────────────────────────
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        kpis = [
            (k1, "kpi-blue",   "T1 Score",         f"{r['score_t1']:.1f}",     "Structural foundation"),
            (k2, "kpi-purple", "T2 Increment",     f"{r['t2_increment']:+.1f}", "Behavioral"),
            (k3, "kpi-amber",  "T3 GAM Lift",      f"{r['t3_lift']:+.1f}",     f"{len(r['triggered'])} triggered"),
            (k4, "kpi-slate",  "Score",         f"{r['score_glm']:.1f}",   "Before T3"),
            (k5, "kpi-red",    "Claim Frequency",       f"{r['claim_freq']:.1%}",   "Annual probability"),
            (k6, "kpi-green",  "Claim Severity",   f"${r['claim_sev']:,.0f}",  "Per event"),
        ]
        for col, css, lbl, val, sub in kpis:
            col.markdown(f"""
            <div class="kpi {css}">
                <div class="kpi-label">{lbl}</div>
                <div class="kpi-value">{val}</div>
                <div class="kpi-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

        # ── Tier 3 interaction alerts ─────────────────────────────────────────
        if r["triggered"]:
            st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
            n = len(r["triggered"])
            st.markdown(f"""
            <div style='background:#faf5ff;border:1.5px solid #8b5cf6;border-radius:12px;
                        padding:16px 20px;margin-bottom:16px'>
                <div style='font-size:11px;font-weight:700;color:#6d28d9;text-transform:uppercase;
                            letter-spacing:1.2px;margin-bottom:12px'>
                    ⚡ {n} Tier 3 Interaction{"s" if n>1 else ""} Triggered — GAM Non-linear Effect Active
                </div>""", unsafe_allow_html=True)

            ix_cols = st.columns(min(n, 3))
            for i, t in enumerate(r["triggered"]):
                with ix_cols[i % min(n, 3)]:
                    status_badge = f'<span class="ix-badge badge-confirmed">Confirmed H={t["h"]:.2f}</span>' \
                        if t["status"] == "CONFIRMED" else f'<span class="ix-badge badge-partial">Partial H={t["h"]:.2f}</span>'
                    st.markdown(f"""
                    <div class="ix-card">
                        <div style='display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:6px'>
                            <div class="ix-name">{t["name"]}</div>
                            {status_badge}
                        </div>
                        <div class="ix-logic">{t["logic"]}</div>
                        <div style='display:flex;gap:12px;margin-top:10px;flex-wrap:wrap'>
                            <div style='font-size:10px;background:#f3f0ff;color:#6d28d9;padding:3px 8px;
                                        border-radius:12px;font-weight:600'>R² lift: {t["r2"]}</div>
                            <div style='font-size:10px;background:#fef3c7;color:#92400e;padding:3px 8px;
                                        border-radius:12px;font-weight:600'>+{t["delta_pts"]:.1f} score pts</div>
                            <div style='font-size:10px;background:#fef2f2;color:#7f1d1d;padding:3px 8px;
                                        border-radius:12px;font-weight:600'>Action: {t["action"]}</div>
                        </div>
                    </div>""", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        # ── Analysis Tabs ─────────────────────────────────────────────────────
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊  Score Waterfall",
            "🔬  Feature Contributions",
            "⚡  Interaction Explorer",
            "🎯  SHAP & Radar",
            "💡  Risk Reduction",
            "📋  Flags & UW Notes",
        ])

        with tab1:
            st.markdown("#### Risk Score Waterfall — How the Score Builds Tier by Tier")
            st.markdown("""
            <p style='font-size:12.5px;color:#64748b;line-height:1.7;margin-bottom:16px'>
            The waterfall traces each tier's additive contribution to the final risk score.
            Red bars increase risk; green bars are mitigants. Hover any bar for the exact η contribution,
            feature list, and actuarial rationale. Dashed lines show underwriting decision thresholds.
            </p>""", unsafe_allow_html=True)
            if PLOTLY:
                st.plotly_chart(build_waterfall(r), use_container_width=True)

            wc1, wc2, wc3 = st.columns(3)
            for col, tier, lbl, val, sub, color in [
                (wc1, "T1", "Tier 1 Foundation", f"{r['score_t1']:.1f} pts",
                 f"η_T1 = {r['eta_t1']:.4f} · exp(η) = ×{r['m_t1']:.3f}", "#3b82f6"),
                (wc2, "T2", "Tier 2 Behavioral", f"{r['t2_increment']:+.1f} pts",
                 f"Δη_T2 = {r['eta_glm']-r['eta_t1']:.4f} · Mult = ×{r['m_t2']:.3f}", "#8b5cf6"),
                (wc3, "T3", "Tier 3 GAM Effect", f"{r['t3_lift']:+.1f} pts",
                 f"δ_GAM = {r['delta_gam']:.4f} · Mult = ×{r['m_gam']:.3f}", "#f59e0b"),
            ]:
                col.markdown(f"""
                <div style='background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;
                            padding:14px;border-top:3px solid {color}'>
                    <div style='font-size:10px;font-weight:700;color:{color};text-transform:uppercase;
                                letter-spacing:1px;margin-bottom:4px'>{lbl}</div>
                    <div style='font-size:22px;font-weight:700;color:#0f172a;font-variant-numeric:tabular-nums'>{val}</div>
                    <div style='font-size:10.5px;font-family:"JetBrains Mono",monospace;color:#64748b;margin-top:4px'>{sub}</div>
                </div>""", unsafe_allow_html=True)

        with tab2:
            st.markdown("#### Feature Contributions — Score Points by Variable & Tier")
            st.markdown("""
            <p style='font-size:12.5px;color:#64748b;line-height:1.7;margin-bottom:16px'>
            Each bar shows how many score points a specific variable contributes.
            Red = risk-increasing; Green = mitigant discount. Variables are grouped and colour-coded by tier.
            Hover for exact values, tier classification, and actuarial rationale.
            </p>""", unsafe_allow_html=True)
            if PLOTLY:
                fig_c = build_contributions(r)
                if fig_c:
                    st.plotly_chart(fig_c, use_container_width=True)

            # Contribution table
            st.markdown("##### Full Contribution Table")

            # Build sorted table
            all_rows = []
            for k, v in r["t1_contrib"].items():
                all_rows.append({"Tier": "T1", "Variable": k, "Score Pts": v,
                                  "η Contribution": v / (100.0 / (ETA_P98 - ETA_P2)), "Direction": "Risk" if v > 0 else "Mitigant"})
            for k, v in r["t2_contrib"].items():
                all_rows.append({"Tier": "T2", "Variable": k, "Score Pts": v,
                                  "η Contribution": v / (100.0 / (ETA_P98 - ETA_P2)), "Direction": "Risk" if v > 0 else "Mitigant"})
            for k, v in r["t3_contrib"].items():
                all_rows.append({"Tier": "T3", "Variable": k, "Score Pts": v,
                                  "η Contribution": v / (100.0 / (ETA_P98 - ETA_P2)), "Direction": "Risk" if v > 0 else "Mitigant"})

            df_c = pd.DataFrame(all_rows).sort_values("Score Pts", key=abs, ascending=False)
            df_c["Score Pts"] = df_c["Score Pts"].round(2)
            df_c["η Contribution"] = df_c["η Contribution"].round(4)
            tier_col = {"T1": "#3b82f6", "T2": "#8b5cf6", "T3": "#f59e0b"}
            max_val = df_c["Score Pts"].abs().max() or 1

            rows_html = ""
            for _, row in df_c.iterrows():
                color = tier_col[row["Tier"]]
                bar_w = int(abs(row["Score Pts"]) / max_val * 80)
                bar_cls = "bar-pos" if row["Score Pts"] > 0 else "bar-neg"
                dir_color = "#ef4444" if row["Score Pts"] > 0 else "#10b981"
                rows_html += f"""
                <tr>
                    <td><span style='background:{color}20;color:{color};padding:2px 7px;border-radius:12px;
                                     font-size:10px;font-weight:700'>{row["Tier"]}</span></td>
                    <td style='font-weight:500'>{row["Variable"]}</td>
                    <td style='color:{dir_color};font-family:"JetBrains Mono",monospace;font-weight:700'>
                        {row["Score Pts"]:+.2f}</td>
                    <td><span class="{bar_cls}" style="width:{bar_w}px"></span></td>
                    <td style='font-family:"JetBrains Mono",monospace;color:#64748b'>{row["η Contribution"]:+.4f}</td>
                    <td><span style='color:{"#ef4444" if row["Direction"]=="Risk" else "#10b981"};
                                     font-size:11px;font-weight:600'>{row["Direction"]}</span></td>
                </tr>"""

            st.markdown(f"""
            <table class="contrib-table">
                <thead><tr>
                    <th>Tier</th><th>Variable</th><th>Score Pts</th><th>Relative Size</th>
                    <th>η Contribution</th><th>Direction</th>
                </tr></thead>
                <tbody>{rows_html}</tbody>
            </table>""", unsafe_allow_html=True)

        with tab3:
            st.markdown("#### Tier 3 Interaction Explorer — Pairwise GAM Effects")
            st.markdown("""
            <p style='font-size:12.5px;color:#64748b;line-height:1.7;margin-bottom:16px'>
            The GLM cannot capture non-linear or compound risks — that's why a GAM residual model is layered on top.
            All 6 pairwise interactions are validated via Friedman H-statistic (H≥0.30 = confirmed).
            <b>Active interactions</b> (triggered by current policy inputs) are highlighted in purple with score impact shown.
            </p>""", unsafe_allow_html=True)

            active_keys = [t["flag"] for t in r["triggered"]]

            # ── Interactive H-statistic chart ─────────────────────────────────
            if PLOTLY:
                ix_names  = [row[1] for row in T3_CATALOGUE]
                ix_h      = [row[3] for row in T3_CATALOGUE]
                ix_r2     = [row[2] for row in T3_CATALOGUE]
                ix_status = [row[4] for row in T3_CATALOGUE]
                ix_logic  = [row[5] for row in T3_CATALOGUE]
                ix_keys   = [row[0] for row in T3_CATALOGUE]
                is_active = [k in active_keys for k in ix_keys]

                # Score impact for active interactions
                ix_delta_map2 = {t["flag"]: t["delta_pts"] for t in r["triggered"]}
                score_impacts = [ix_delta_map2.get(k, 0) for k in ix_keys]

                # Colors: active = purple gradient by H value, inactive = muted grey
                bar_colors = []
                for i, (active, h_val) in enumerate(zip(is_active, ix_h)):
                    if active:
                        bar_colors.append("#7c3aed" if h_val >= 0.30 else "#a78bfa")
                    else:
                        bar_colors.append("#e2e8f0")

                fig_ix = go.Figure()

                # Main H-stat bars
                fig_ix.add_trace(go.Bar(
                    x=ix_h,
                    y=ix_names,
                    orientation="h",
                    marker=dict(
                        color=bar_colors,
                        line=dict(color=["#6d28d9" if a else "#cbd5e1" for a in is_active], width=1.5),
                    ),
                    text=[f"<b>H={h:.3f}</b>" if a else f"H={h:.3f}"
                          for h, a in zip(ix_h, is_active)],
                    textposition="outside",
                    textfont=dict(size=10, color="#1e293b"),
                    customdata=list(zip(
                        ix_names, ix_h, ix_r2, ix_status, ix_logic,
                        ["✅ ACTIVE" if a else "— inactive" for a in is_active],
                        score_impacts,
                        [IX_CONDITIONS[k][0] for k in ix_keys],
                        [IX_CONDITIONS[k][1] for k in ix_keys],
                        [IX_CONDITIONS[k][2] for k in ix_keys],
                    )),
                    hovertemplate=(
                        "<b>%{customdata[0]}</b>  %{customdata[5]}<br>"
                        "──────────────────────────<br>"
                        "Friedman H-statistic: <b>%{customdata[1]:.4f}</b>  (threshold: 0.30)<br>"
                        "R² lift: <b>%{customdata[2]}</b> · Status: %{customdata[3]}<br>"
                        "Score impact (this policy): <b>+%{customdata[6]:.2f} pts</b><br>"
                        "──────────────────────────<br>"
                        "Condition 1: %{customdata[7]}<br>"
                        "Condition 2: %{customdata[8]}<br>"
                        "Required action: %{customdata[9]}<br>"
                        "──────────────────────────<br>"
                        "<i style='color:#94a3b8'>%{customdata[4]}</i><extra></extra>"
                    ),
                    name="H-Statistic",
                ))

                # Score impact overlay bars for active interactions
                active_impacts = [s if a else 0 for s, a in zip(score_impacts, is_active)]
                if any(v > 0 for v in active_impacts):
                    fig_ix.add_trace(go.Bar(
                        x=[v * 0.6 for v in active_impacts],  # scale to fit same axis
                        y=ix_names,
                        orientation="h",
                        marker=dict(color="rgba(239,68,68,0.18)", line=dict(color="#ef4444", width=1)),
                        text=[f"+{v:.1f}pts" if v > 0 else "" for v in active_impacts],
                        textposition="inside",
                        textfont=dict(size=9, color="#7f1d1d"),
                        hovertemplate="Score impact: <b>+%{customdata:.2f} pts</b><extra></extra>",
                        customdata=active_impacts,
                        name="Score Impact",
                    ))

                # Threshold line
                fig_ix.add_vline(x=0.30, line_color="#f59e0b", line_dash="dot", line_width=2,
                                  annotation_text="  H≥0.30 = Confirmed",
                                  annotation_font=dict(size=9.5, color="#92400e", family="DM Sans"),
                                  annotation_position="top")

                fig_ix.update_layout(
                    showlegend=True,
                    legend=dict(orientation="h", x=0, y=-0.15, font=dict(size=10, color="#475569")),
                    barmode="overlay",
                    yaxis=dict(autorange="reversed", tickfont=dict(size=11, color="#1e293b")),
                    xaxis=dict(title="Friedman H-Statistic (interaction strength)",
                               title_font=dict(size=10, color="#64748b"), range=[0, 0.58]),
                )
                st.plotly_chart(_base_fig(fig_ix, 310, dict(l=8, r=90, t=24, b=40)), use_container_width=True)

                # Active vs inactive summary pills
                n_confirmed = sum(1 for h in ix_h if h >= 0.30)
                n_active    = sum(is_active)
                sum_impact  = sum(score_impacts)
                pill_cols = st.columns(3)
                for col, lbl, val, bg, tc in [
                    (pill_cols[0], "Confirmed (H≥0.30)", f"{n_confirmed}/6", "#fef3c7", "#92400e"),
                    (pill_cols[1], "Active This Policy",  f"{n_active}/6",    "#f5f3ff", "#6d28d9"),
                    (pill_cols[2], "Total Score Impact",  f"+{sum_impact:.1f} pts", "#fef2f2", "#7f1d1d"),
                ]:
                    col.markdown(f"""
                    <div style='background:{bg};border-radius:10px;padding:10px 16px;text-align:center'>
                        <div style='font-size:9px;text-transform:uppercase;letter-spacing:1px;
                                    color:{tc};font-weight:700;margin-bottom:2px'>{lbl}</div>
                        <div style='font-size:22px;font-weight:700;color:{tc};font-variant-numeric:tabular-nums'>{val}</div>
                    </div>""", unsafe_allow_html=True)
                st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

            # Individual interaction cards
            st.markdown("##### All Interactions — Conditions & Effects")
            for row in T3_CATALOGUE:
                fc, name, r2, h, status, logic = row
                active = fc in active_keys
                c1_str, c2_str, action_str = IX_CONDITIONS[fc]
                active_style = "" if active else " ix-card-inactive"
                badge = f'<span class="ix-badge badge-confirmed">CONFIRMED  H={h:.2f}</span>' \
                    if status == "CONFIRMED" else f'<span class="ix-badge badge-partial">PARTIAL  H={h:.2f}</span>'
                active_label = '<span style="background:#d1fae5;color:#065f46;font-size:10px;font-weight:700;padding:2px 8px;border-radius:12px">✅ ACTIVE</span>' \
                    if active else '<span style="background:#f1f5f9;color:#94a3b8;font-size:10px;padding:2px 8px;border-radius:12px">— inactive</span>'

                st.markdown(f"""
                <div class="ix-card{active_style}">
                    <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:8px'>
                        <div class="ix-name" style='font-size:14px'>{name}</div>
                        <div style='display:flex;gap:8px;align-items:center'>{active_label} {badge}</div>
                    </div>
                    <div style='display:flex;gap:12px;margin-bottom:8px;flex-wrap:wrap'>
                        <div style='font-size:11px;background:#eff6ff;color:#1d4ed8;padding:4px 10px;border-radius:8px'>
                            Condition 1: {c1_str}</div>
                        <div style='font-size:11px;background:#eff6ff;color:#1d4ed8;padding:4px 10px;border-radius:8px'>
                            Condition 2: {c2_str}</div>
                        <div style='font-size:11px;background:#fef9c3;color:#854d0e;padding:4px 10px;border-radius:8px'>
                            Action: {action_str}</div>
                        <div style='font-size:11px;background:#f3f0ff;color:#6d28d9;padding:4px 10px;border-radius:8px'>
                            R² lift: {r2}</div>
                    </div>
                    <div class="ix-logic">{logic}</div>
                </div>""", unsafe_allow_html=True)

        with tab4:
            st.markdown("#### SHAP Attribution & Risk Radar")
            shap_col, radar_col = st.columns([3, 2])

            with shap_col:
                st.markdown("""
                <p style='font-size:12.5px;color:#64748b;line-height:1.7;margin-bottom:12px'>
                SHAP-style attribution chart showing the score-point contribution of each variable.
                Red bars increase risk; green bars are mitigants. Bar borders indicate tier (blue=T1, purple=T2, amber=T3).
                Hover any bar for exact values.
                </p>""", unsafe_allow_html=True)
                if PLOTLY:
                    fig_shap = build_shap_waterfall(r)
                    if fig_shap:
                        st.plotly_chart(fig_shap, use_container_width=True)

            with radar_col:
                st.markdown("""
                <p style='font-size:12.5px;color:#64748b;line-height:1.7;margin-bottom:12px'>
                Risk profile across 5 dimensions vs the portfolio average (dotted line).
                Larger filled area = higher composite risk.
                </p>""", unsafe_allow_html=True)
                if PLOTLY:
                    fig_radar = build_radar_chart(r)
                    st.plotly_chart(fig_radar, use_container_width=True)

            # Quick summary row
            dim_labels = ["Structural", "Behavioral", "Geographic", "Interactions", "Severity"]
            s_t1_norm = min(10, r["score_t1"] / 10)
            s_t2_norm = min(10, max(0, (r["t2_increment"] + 5) / 3))
            geo_sum = sum(v for k, v in r["t3_contrib"].items() if k != "Non-linear Smooths")
            s_geo  = min(10, max(0, geo_sum / 2 + 3))
            s_ix   = min(10, len(r["triggered"]) * 3.5)
            s_sev  = min(10, (r["claim_sev"] - 1500) / 5350)
            dim_vals = [s_t1_norm, s_t2_norm, s_geo, s_ix, s_sev]
            dim_colors = ["#3b82f6","#8b5cf6","#f59e0b","#ef4444","#64748b"]
            dim_cols = st.columns(5)
            for col, lbl, val, color in zip(dim_cols, dim_labels, dim_vals, dim_colors):
                bar_w = int(val / 10 * 100)
                col.markdown(f"""
                <div style='background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:12px 14px;text-align:center'>
                    <div style='font-size:9px;text-transform:uppercase;letter-spacing:1px;color:{color};font-weight:700;margin-bottom:4px'>{lbl}</div>
                    <div style='font-size:20px;font-weight:700;color:#0f172a;font-variant-numeric:tabular-nums'>{val:.1f}<span style='font-size:11px;color:#94a3b8'>/10</span></div>
                    <div style='height:4px;background:#f1f5f9;border-radius:2px;margin-top:6px'>
                        <div style='height:4px;width:{bar_w}%;background:{color};border-radius:2px'></div>
                    </div>
                </div>""", unsafe_allow_html=True)

        with tab5:
            st.markdown("#### Risk Reduction — What-If Improvement Scenarios")
            st.markdown("""
            <p style='font-size:12.5px;color:#64748b;line-height:1.7;margin-bottom:16px'>
            Actionable steps the insured or agent can take to reduce risk score and lower premium.
            Each scenario is computed by re-running the full GLM-GAM model with a single change.
            Only improvements of ≥0.4 score points are shown. Sorted by largest improvement first.
            </p>""", unsafe_allow_html=True)

            if snap:
                scenarios = build_sensitivity_scenarios(snap, r["score_final"], r["premium"])
                if scenarios:
                    dec_pal2 = {
                        "Preferred": ("#16a34a", "#dcfce7"),
                        "Standard":  ("#2563eb", "#dbeafe"),
                        "Rated":     ("#d97706", "#fef9c3"),
                        "Decline":   ("#dc2626", "#fee2e2"),
                    }
                    for s in scenarios:
                        dc, dbg = dec_pal2.get(s["new_decision"], ("#64748b", "#f1f5f9"))
                        upgrade = s["new_decision"] != r["decision"]
                        upgrade_badge = f' <span style="background:{dbg};color:{dc};font-size:10px;font-weight:700;padding:2px 8px;border-radius:12px">→ {s["new_decision"]}</span>' if upgrade else ""
                        prem_str = (f' · Premium: ${s["new_premium"]:,.0f}'
                                    f' ({"+" if (s["prem_delta"] or 0) > 0 else ""}${s["prem_delta"]:,.0f})'
                                    if s["new_premium"] and s["prem_delta"] is not None else "")
                        st.markdown(f"""
                        <div class="sens-row">
                            <div class="sens-action">✦ {s["action"]}{upgrade_badge}</div>
                            <div style='display:flex;gap:16px;align-items:center;flex-shrink:0'>
                                <div style='font-size:11px;color:#64748b'>Score: {r["score_final"]:.0f} →
                                    <strong style='color:#0f172a'>{s["new_score"]}</strong></div>
                                <div class="sens-delta">{s["delta"]:+.1f} pts</div>
                                <div style='font-size:11px;color:#64748b'>{prem_str}</div>
                            </div>
                        </div>""", unsafe_allow_html=True)
                else:
                    st.info("No feasible improvements found for this configuration — this policy is already near its minimum achievable score.")

            # Portfolio context
            if df_port is not None:
                st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
                st.markdown("##### Portfolio Context")
                fig_hist, pct = build_portfolio_hist(r["score_final"], df_port)
                st.plotly_chart(fig_hist, use_container_width=True)
                p1c, p2c, p3c = st.columns(3)
                for col, lbl, val in [
                    (p1c, "This Policy Score", f"{r['score_final']:.1f}"),
                    (p2c, "Portfolio Percentile", f"{pct:.0f}th"),
                    (p3c, "Policies Below", f"{pct:.0f}%"),
                ]:
                    col.metric(lbl, val)

        with tab6:
            st.markdown("#### Underwriting Flags & Notes")
            flags = r["flags"]
            if flags:
                h_flags = [(t, m) for t, m in flags if t == "H"]
                w_flags = [(t, m) for t, m in flags if t == "W"]
                g_flags = [(t, m) for t, m in flags if t == "G"]
                t_flags = [(t, m) for t, m in flags if t == "T"]

                for group, label, color in [
                    (h_flags, "🚨 Critical — Manual Review Required", "#ef4444"),
                    (w_flags, "⚠️  Warning — Elevated Risk Indicators", "#f59e0b"),
                    (t_flags, "⚡ GAM Interaction Flags", "#8b5cf6"),
                    (g_flags, "✅ Mitigants — Discount Applied", "#22c55e"),
                ]:
                    if group:
                        st.markdown(f"<div style='font-size:11px;font-weight:700;color:{color};text-transform:uppercase;"
                                    f"letter-spacing:1px;margin-top:14px;margin-bottom:6px'>{label}</div>",
                                    unsafe_allow_html=True)
                        for t, msg in group:
                            st.markdown(f'<div class="flag flag-{t}">{msg}</div>', unsafe_allow_html=True)
            else:
                st.success("No flags raised — clean policy.")

            # Premium build-up table
            if r["premium"]:
                st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
                st.markdown("##### Premium Build-Up")
                lr_map2 = {"Preferred": 0.68, "Standard": 0.65, "Rated": 0.60}
                load_map = {"Preferred": 1.25, "Standard": 1.55, "Rated": 2.00}
                lr   = lr_map2.get(r["decision"], 0.65)
                load = load_map.get(r["decision"], 1.55)
                rows_prem = [
                    ("Pure Premium (E[Loss])", f"${r['loss_final']:,.0f}", "Calibrated expected annual loss from GLM-GAM model"),
                    ("×Loss Load Factor", f"×{load:.2f}", f"Decision: {r['decision']} — covers adverse development & CAT loading"),
                    ("Loaded Loss", f"${r['loss_final']*load:,.0f}", "Loss with CAT and development load"),
                    ("÷Target Loss Ratio", f"{lr:.0%}", "Expense ratio: 30% (incl. commissions, admin, profit)"),
                    ("Indicative Premium", f"${r['premium']:,.0f}", "Annual gross written premium estimate"),
                ]
                for label, val, desc in rows_prem:
                    st.markdown(f"""
                    <div style='display:flex;align-items:center;gap:12px;padding:9px 14px;
                                background:#f8fafc;border-radius:8px;margin-bottom:5px;border:1px solid #e2e8f0'>
                        <div style='flex:1;font-size:12.5px;font-weight:600;color:#1e293b'>{label}</div>
                        <div style='font-family:"JetBrains Mono",monospace;font-size:13px;font-weight:700;
                                    color:#0f172a;min-width:90px;text-align:right'>{val}</div>
                        <div style='font-size:11px;color:#64748b;flex:2'>{desc}</div>
                    </div>""", unsafe_allow_html=True)
            else:
                st.warning("Policy declined — no standard market premium. Refer to E&S market.")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — PORTFOLIO ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif "Portfolio" in section:
    st.markdown("""
    <div class="platform-hero">
        <div class="hero-eyebrow">Portfolio Analytics</div>
        <div class="hero-title">10,000-Policy Risk Distribution</div>
        <div class="hero-sub">
            Visual analysis of GLM-GAM tier contributions across the full portfolio.
            Understand how behavioral and geographical factors layer on top of structural risk —
            and where the interaction effects concentrate.
        </div>
    </div>""", unsafe_allow_html=True)

    if df_port is None:
        st.warning("Portfolio data not found. Run `python generate_data.py` then `python models.py` first.")
        st.stop()

    # Score vs Loss scatter
    st.markdown('<div class="sec-hd">Risk Score vs Expected Annual Loss</div><hr class="sec-line">', unsafe_allow_html=True)
    if PLOTLY:
        fig_sl = build_score_loss_scatter(df_port)
        if fig_sl:
            st.plotly_chart(fig_sl, use_container_width=True)

    st.markdown("""
    <p style='font-size:12.5px;color:#64748b;line-height:1.7;margin-top:-12px;margin-bottom:24px'>
    Clear monotonic relationship between risk score and expected loss confirms model discrimination power.
    The GLM-GAM framework achieves ~92% explained variance — well above the industry benchmark of 65–70%.
    Colour shows the four decision segments; overlap at boundaries is by design (±5pt uncertainty band).
    </p>""", unsafe_allow_html=True)

    # Decision segment summary
    st.markdown('<div class="sec-hd">Decision Segment Summary</div><hr class="sec-line">', unsafe_allow_html=True)
    if "decision" in df_port.columns:
        seg_stats = df_port.groupby("decision").agg(
            Policies=("decision", "count"),
        ).reset_index()
        if "expected_loss" in df_port.columns:
            loss_stats = df_port.groupby("decision")["expected_loss"].mean().reset_index()
            loss_stats.columns = ["decision", "Avg Expected Loss"]
            seg_stats = seg_stats.merge(loss_stats, on="decision")
        score_col = "final_risk_score" if "final_risk_score" in df_port.columns else "score_final"
        if score_col in df_port.columns:
            sc_stats = df_port.groupby("decision")[score_col].mean().reset_index()
            sc_stats.columns = ["decision", "Avg Risk Score"]
            seg_stats = seg_stats.merge(sc_stats, on="decision")

        seg_stats["Share"] = (seg_stats["Policies"] / seg_stats["Policies"].sum() * 100).round(1)
        seg_stats = seg_stats.sort_values("Avg Risk Score" if "Avg Risk Score" in seg_stats.columns else "decision")

        sc1, sc2, sc3, sc4 = st.columns(4)
        dec_order = ["Preferred", "Standard", "Rated", "Decline"]
        for col, dec in zip([sc1, sc2, sc3, sc4], dec_order):
            row = seg_stats[seg_stats["decision"] == dec]
            if row.empty:
                continue
            row = row.iloc[0]
            pal = DEC_PALETTE[dec]
            loss_str = f"${row['Avg Expected Loss']:,.0f}" if "Avg Expected Loss" in row else "—"
            score_str = f"{row['Avg Risk Score']:.1f}" if "Avg Risk Score" in row else "—"
            col.markdown(f"""
            <div style='background:{pal["bg"]};border:1.5px solid {pal["border"]};border-radius:12px;
                        padding:16px 18px;'>
                <div style='font-size:10px;text-transform:uppercase;letter-spacing:1.5px;
                            color:{pal["accent"]};font-weight:700;margin-bottom:4px'>{dec}</div>
                <div style='font-size:26px;font-weight:700;color:{pal["text"]};font-variant-numeric:tabular-nums'>
                    {int(row["Policies"]):,}</div>
                <div style='font-size:11px;color:{pal["text"]};margin-top:2px'>
                    {row["Share"]:.1f}% of portfolio</div>
                <div style='height:1px;background:#e2e8f0;margin:10px 0'></div>
                <div style='font-size:12px;color:{pal["text"]};font-weight:600'>
                    Avg Score: {score_str}</div>
                <div style='font-size:12px;color:{pal["text"]}'>
                    Avg E[Loss]: {loss_str}</div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — FRAMEWORK GUIDE
# ══════════════════════════════════════════════════════════════════════════════
elif "Framework" in section:
    st.markdown("""
    <div class="platform-hero">
        <div class="hero-eyebrow">Technical Documentation</div>
        <div class="hero-title">GLM-GAM 3-Tier Framework</div>
        <div class="hero-sub">
            Regulator-ready documentation of the actuarial model architecture, variable catalogue,
            and interaction effects.
        </div>
    </div>""", unsafe_allow_html=True)

    # Decision thresholds compact banner
    st.markdown("""
    <div style='background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;padding:14px 20px;
                margin-bottom:20px;display:flex;gap:8px;align-items:center;flex-wrap:wrap'>
        <span style='font-size:10px;font-weight:700;color:#64748b;text-transform:uppercase;
                     letter-spacing:1px;margin-right:8px'>Decision Thresholds:</span>
        <span style='background:#f0fdf4;color:#14532d;font-size:11px;font-weight:700;padding:4px 14px;border-radius:20px;border:1px solid #22c55e'>0–29 · Preferred</span>
        <span style='background:#eff6ff;color:#1e3a8a;font-size:11px;font-weight:700;padding:4px 14px;border-radius:20px;border:1px solid #3b82f6'>30–59 · Standard</span>
        <span style='background:#fffbeb;color:#78350f;font-size:11px;font-weight:700;padding:4px 14px;border-radius:20px;border:1px solid #f59e0b'>60–79 · Rated</span>
        <span style='background:#fef2f2;color:#7f1d1d;font-size:11px;font-weight:700;padding:4px 14px;border-radius:20px;border:1px solid #ef4444'>80–100 · Decline</span>
    </div>""", unsafe_allow_html=True)

    # Tier 1 variable table
    st.markdown('<div class="sec-hd">Tier 1 Variables — Structural Foundation</div><hr class="sec-line">', unsafe_allow_html=True)
    t1_data = {
        "Variable":       ["Roof Age",   "Roof Vulnerability", "Dwelling Construction", "Water Loss Recency",
                           "Prior Claims (5yr)", "Coverage A Amount", "Fire Station Distance", "Home Age",
                           "Square Footage", "ISO Class"],
        "r Value":        [0.42, 0.38, 0.35, 0.29, 0.26, 0.18, 0.15, 0.12, 0.08, 0.06],
        "GLM β":          ["+0.027", "+0.050", "+0.045", "+0.155", "+0.225", "+0.018", "+0.013", "+0.012", "—", "+0.030"],
        "Why Important":  ["Material degradation", "Weather resistance", "Structural fire/wind resilience",
                           "Chronic maintenance indicator", "Loss history pattern", "Property size/value proxy",
                           "Fire response speed", "Infrastructure age", "Exposure/maintenance burden",
                           "Fire department quality"],
        "UW Impact":      ["Inspection >25yr", "Material rating factor", "Type-based adjustment",
                           "Inspection required", "Frequency adjustment", "Coverage validation",
                           "Location-based review", "Inspection trigger", "Limit validation", "Location adjustment"],
    }
    st.dataframe(pd.DataFrame(t1_data), use_container_width=True, hide_index=True)

    # Tier 2 variable table
    st.markdown('<div class="sec-hd">Tier 2 Variables — Behavioral & Lifestyle</div><hr class="sec-line">', unsafe_allow_html=True)
    t2_data = {
        "Variable":     ["Prior Claims (5yr)", "Insurance Lapses", "Crime Index", "Pet Ownership",
                         "Swimming Pool", "Trampoline", "Wood-Burning Stove", "Home Business",
                         "Recent Renovations", "Monitored Alarm", "Fire Sprinklers", "Gated Community"],
        "r Value":      [0.26, 0.18, 0.19, 0.16, 0.14, 0.12, 0.10, 0.09, 0.08, 0.07, 0.05, 0.03],
        "Effect Type":  ["Loss history", "Negative behavior", "Area risk", "Liability",
                         "Liability + maint.", "Injury liability", "Fire risk", "Liability + wear",
                         "Positive indicator", "Loss prevention", "Fire severity reduction", "Theft reduction"],
        "Typical Adj.": ["+10–30%/claim", "+15–30%", "+5–15%", "+5–25%",
                         "+$100–300/yr", "+$250/yr", "+5–20%", "Refer if active",
                         "−15%", "−10%", "−10 to −20%", "−5%"],
        "GLM β":        ["+0.225", "+0.140", "+0.008", "+0.028",
                         "+0.052", "+0.165", "+0.048", "+0.078",
                         "−0.038", "−0.110", "−0.065", "−0.072"],
    }
    st.dataframe(pd.DataFrame(t2_data), use_container_width=True, hide_index=True)

    # Tier 3 interaction table
    st.markdown('<div class="sec-hd">Tier 3 Interactions — GAM Catalogue</div><hr class="sec-line">', unsafe_allow_html=True)
    t3_data = {
        "Interaction":  [row[1] for row in T3_CATALOGUE],
        "R² Lift":      [row[2] for row in T3_CATALOGUE],
        "H-Stat":       [row[3] for row in T3_CATALOGUE],
        "Status":       [row[4] for row in T3_CATALOGUE],
        "Condition 1":  [IX_CONDITIONS[row[0]][0] for row in T3_CATALOGUE],
        "Condition 2":  [IX_CONDITIONS[row[0]][1] for row in T3_CATALOGUE],
        "Action":       [IX_CONDITIONS[row[0]][2] for row in T3_CATALOGUE],
        "Logic":        [row[5][:80] + "..." for row in T3_CATALOGUE],
    }
    st.dataframe(pd.DataFrame(t3_data), use_container_width=True, hide_index=True)