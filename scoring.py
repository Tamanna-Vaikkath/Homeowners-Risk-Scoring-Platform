"""
scoring.py
==========
Backend scoring engine — all GLM-GAM computation lives here.
app.py only calls score_policy() and renders the returned dict.

Tier 1 — Structural Foundation (GLM, ~60% variance):
    Roof Age, Roof Material, Home Age, Construction Type,
    Prior Water Claim, RCV Ratio, Coverage A, Location Zone

Tier 2 — Lifestyle & Secondary (GLM, +~20% incremental):
    Prior Claims, Insurance Lapses, Pool, Trampoline, Home Business,
    Sprinklers, Alarm, Gated, Wood Stove, Recent Reno, Pet Ownership,
    Building Code, Hydrant Distance, ISO Class, Crime Index

Tier 3 — Geographical + GAM Residual (+~12% incremental):
    Wildfire Score/Zone, Canopy%, Flood Zone/Depth, Slope,
    Hail Zone, Burn History, Foundation + 6 pairwise interactions
"""

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.special import expit  # noqa: F401  (kept for future freq model)

ROOT    = Path(__file__).parent
MDL_DIR = ROOT / "data" / "models"

# ── Calibrated score normalisation ───────────────────────────────────────────
# Calibrated from a 5,000-policy sweep of score_policy() outputs so that:
#   Preferred  0-30  |  Standard 31-60  |  Rated 61-80  |  Decline 81-100
ETA_P2_DEFAULT  = 7.18
ETA_P98_DEFAULT = 12.85

# ── Location zones & offsets ──────────────────────────────────────────────────
LOCATIONS = [
    "CA-Wildfire", "TX-Hail",   "FL-Flood",  "CO-Mixed",
    "IL-Midwest",  "NY-NE",     "AZ-Desert", "WA-NW",
    "GA-SE",       "MN-North",
]
LOC_OFFSET = {
    "CA-Wildfire":  0.22,  "FL-Flood":    0.28,  "CO-Mixed":   0.07,
    "TX-Hail":      0.13,  "WA-NW":      -0.07,  "AZ-Desert": -0.11,
    "NY-NE":       -0.03,  "IL-Midwest": -0.14,  "GA-SE":     -0.18,
    "MN-North":    -0.22,
}

# ── Tier 3 Interaction catalogue ──────────────────────────────────────────────
# (flag_col, display_name, friedman_h, status, r2_lift, logic)
T3_CATALOGUE = [
    ("ix_roof_wildfire",    "Roof Age × Wildfire Zone",      0.43, "CONFIRMED", "+6%",
     "Roof >20yr in wildfire zone — fire resistance degraded; 2.8× loss frequency"),
    ("ix_water_canopy",     "Water Claim × Tree Canopy",     0.31, "CONFIRMED", "+3%",
     "Prior water claim + dense canopy >60% — gutter clog recurrence 1.6×"),
    ("ix_rcv_crime",        "RCV Overstate × Crime Index",   0.32, "CONFIRMED", "+2%",
     "Over-insured + high crime zone — theft/arson incentive 1.4×"),
    ("ix_flood_foundation", "Flood Zone × Stone Foundation", 0.14, "PARTIAL",   "+1.5%",
     "FEMA flood zone + permeable foundation — water ingress severity 1.3×"),
    ("ix_slope_burn",       "Slope × Burn History",          0.33, "CONFIRMED", "+1%",
     "Slope >20° + recent burn scar — mudslide/debris-flow risk 2.1×"),
    ("ix_roof_hail",        "Roof Age × Hail Zone",          0.33, "CONFIRMED", "+1%",
     "Asphalt roof >20yr in hail zone — impact penetration severity 1.5×"),
]

# ── Interaction condition descriptions (for UI display) ───────────────────────
IX_CONDITIONS = {
    "ix_roof_wildfire":    ("Roof >20yr",        "High wildfire zone",  "Mandatory inspection"),
    "ix_water_canopy":     ("Prior water claim", "Dense canopy >60%",   "Rate +25%"),
    "ix_rcv_crime":        ("RCV ratio >1.15",   "Crime index >55",     "Coverage adjustment"),
    "ix_flood_foundation": ("FEMA flood zone",   "Stone foundation",    "Flood policy required"),
    "ix_slope_burn":       ("Slope >20°",        "Recent burn scar",    "Decline or +50% premium"),
    "ix_roof_hail":        ("Asphalt roof >20yr","Hail zone",           "Rate +30%"),
}

# ── Score normalisation ───────────────────────────────────────────────────────
_params: Optional[dict] = None

def _get_norm():
    """Load calibrated P2/P98 from disk, fall back to defaults."""
    global _params
    if _params is None:
        p = MDL_DIR / "score_params.pkl"
        if p.exists():
            with open(p, "rb") as f:
                loaded = pickle.load(f)
            # Only use saved params if they reflect the wider calibrated range
            if loaded.get("eta_p98", 0) - loaded.get("eta_p2", 0) > 4.0:
                _params = loaded
    if _params:
        return _params["eta_p2"], _params["eta_p98"]
    return ETA_P2_DEFAULT, ETA_P98_DEFAULT


def _eta_to_score(eta: float) -> float:
    p2, p98 = _get_norm()
    return float(np.clip((eta - p2) / (p98 - p2) * 100, 0, 100))


# ── Main public function ──────────────────────────────────────────────────────
def score_policy(
    # Tier 1 — Structural Foundation
    roof_age:             int,
    roof_material:        str,   # asphalt | tile | slate | metal
    home_age:             int,
    construction_type:    str,   # wood_frame | brick_veneer | masonry | superior
    prior_water_claim:    int,   # 0 / 1
    months_since_water:   int,   # 1-36 or 999
    coverage_a:           float,
    rcv_ratio:            float,
    # Tier 2 — Lifestyle & Secondary
    ins_lapses:           int,
    swimming_pool:        str,   # none | in_ground | above_ground
    trampoline:           int,
    home_business:        str,   # none | home_office | active_business
    fire_sprinklers:      str,   # none | partial | full
    monitored_alarm:      int,
    gated_community:      int,
    wood_stove:           int,
    recent_reno:          int,
    pet_ownership:        int,
    prior_claims_5yr:     int,
    fire_hydrant_dist:    float,
    iso_class:            int,
    building_code_comply: int,
    crime_idx:            float,
    # Tier 3 — Geographical + GAM context
    wildfire_score:       float,
    wildfire_zone:        int,
    canopy_pct:           float,
    flood_zone:           int,
    flood_depth_in:       float,
    slope_deg:            float,
    hail_zone:            int,
    burn_history:         int,
    foundation_type:      str,   # concrete_slab | poured_concrete | block | stone_dirt
    location_zone:        str,
) -> dict:
    """
    Score a single homeowners policy using the GLM-GAM 3-tier framework.
    Returns a complete result dict — no print statements, no UI logic.
    """

    # ── Categorical encodings ─────────────────────────────────────────────────
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

    # ── Tier 1 η  (Structural Foundation) ────────────────────────────────────
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

    # ── Tier 2 η  (Lifestyle & Secondary — incremental) ───────────────────────
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

    # ── Tier 3: Interaction flags ─────────────────────────────────────────────
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

    # ── Tier 3: GAM δ  (non-linear smooths + interactions) ────────────────────
    gam_roof_nl  = 0.004 * max(roof_age  - 15, 0) ** 2 / 100   # spline: accelerates >15yr
    gam_wf_nl    = 0.011 * (wildfire_score / 100) ** 2           # quadratic wildfire
    gam_flood_nl = 0.007 * np.log1p(flood_depth_in)              # log-concave flood depth
    gam_slope_nl = 0.009 * np.sqrt(max(slope_deg, 0))            # sqrt slope

    gam_ix_rfw   = ix_roof_wf  * 0.170   # H=0.43 strongest
    gam_ix_wcan  = ix_water_cn * 0.115   # H=0.31
    gam_ix_rcrc  = ix_rcv_cr  * 0.095   # H=0.32
    gam_ix_ffnd  = ix_fld_fnd * 0.085   # H=0.14 partial
    gam_ix_sbrn  = ix_slp_brn * 0.075   # H=0.33
    gam_ix_rhail = ix_rf_hail * 0.065   # H=0.33

    delta_gam = (
        gam_roof_nl + gam_wf_nl + gam_flood_nl + gam_slope_nl
        + gam_ix_rfw + gam_ix_wcan + gam_ix_rcrc
        + gam_ix_ffnd + gam_ix_sbrn + gam_ix_rhail
    )
    eta_final = eta_glm + delta_gam

    # ── Risk scores ───────────────────────────────────────────────────────────
    p2, p98 = _get_norm()
    s_t1    = _eta_to_score(eta_t1)
    s_glm   = _eta_to_score(eta_glm)
    s_final = _eta_to_score(eta_final)

    # ── Loss multipliers ──────────────────────────────────────────────────────
    m_t1  = float(np.exp(eta_t1      - 8.10 - loc_off))
    m_t2  = float(np.exp(eta_t2_incr))
    m_gam = float(np.exp(delta_gam))

    # ── Expected loss  (log-linear calibration → realistic HO dollar range) ──
    # η≈8.2→$1,200 (Preferred) | η≈9.5→$2,500 (Standard)
    # η≈11.4→$7,000 (Rated)    | η≈13.1→$15,000 (Decline/E&S)
    _LA, _LB = 0.5208, 2.7995
    loss_t1    = float(np.clip(np.exp(_LA * eta_t1    + _LB), 800, 20_000))
    loss_glm   = float(np.clip(np.exp(_LA * eta_glm   + _LB), 800, 20_000))
    loss_final = float(np.clip(np.exp(_LA * eta_final + _LB), 800, 20_000))

    # ── Claim frequency & severity ────────────────────────────────────────────
    claim_freq = float(np.clip(0.04 + (s_final / 100) * 0.42, 0.04, 0.50))
    claim_sev  = float(np.clip(loss_final / max(claim_freq, 0.03), 1_500, 55_000))

    # ── Underwriting decision ─────────────────────────────────────────────────
    if s_final < 30:
        decision = "Preferred"
        desc     = "Auto-bind eligible · Preferred tier rates · No manual review required"
    elif s_final < 60:
        decision = "Standard"
        desc     = "Standard rates ±15% · Desktop review recommended"
    elif s_final < 80:
        decision = "Rated"
        desc     = "15–50% surcharge · Senior UW manual review required before binding"
    else:
        decision = "Decline"
        desc     = "Refer to E&S market · Do not bind at standard rates"

    # ── Indicative premium  (E[Loss] / target loss ratio) ─────────────────────
    lr_map  = {"Preferred": 0.68, "Standard": 0.65, "Rated": 0.60}
    lr      = lr_map.get(decision)
    premium = round(loss_final / lr, -1) if lr else None

    # ── Feature contributions  (score-point scale) ────────────────────────────
    scale = 100.0 / (p98 - p2)

    tier1_contrib = {
        "Roof Vulnerability":   (0.027 * roof_age + 0.050 * mat_r)          * scale,
        "Home Age / Structure": (0.012 * home_age + 0.045 * cnst_r)         * scale,
        "Water Loss Recency":   (rec_f * 0.155 * prior_water_claim)          * scale,
        "RCV Validation":       (0.090 * rcv_os + 0.018 * rcv_ratio)        * scale,
        "Location Zone":        loc_off                                      * scale,
    }
    tier2_contrib = {
        "Prior Claims (5yr)":   0.225 * prior_claims_5yr   * scale,
        "Insurance Lapses":     0.140 * ins_lapses          * scale,
        "Crime Index":          0.008 * crime_idx / 20     * scale,
        "Pool / Trampoline":   (0.052 * pool_r + 0.165 * trampoline) * scale,
        "Home Business":        0.078 * biz_r               * scale,
        "Fire Sprinklers":     -0.065 * abs(sprnk)          * scale,
        "Monitored Alarm":     -0.110 * monitored_alarm     * scale,
        "Wood Stove":           0.048 * wood_stove          * scale,
        "Gated Community":     -0.072 * gated_community     * scale,
        "Building Compliance": -0.055 * building_code_comply * scale,
        "Hydrant Distance":     0.013 * fire_hydrant_dist   * scale,
        "Recent Renovation":   -0.038 * recent_reno         * scale,
        "Pet Ownership":        0.028 * pet_ownership       * scale,
    }
    tier3_contrib = {}
    smooth_total = (gam_roof_nl + gam_wf_nl + gam_flood_nl + gam_slope_nl) * scale
    if abs(smooth_total) > 0.01:
        tier3_contrib["Non-linear Smooths"] = smooth_total

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
            tier3_contrib[name] = v

    # ── Triggered interactions ────────────────────────────────────────────────
    triggered = []
    for fc, name, h, status, r2, logic in T3_CATALOGUE:
        if ix_map.get(fc, 0):
            triggered.append({
                "flag": fc, "name": name, "h": h,
                "status": status, "r2": r2, "logic": logic,
                "delta": ix_delta_map.get(fc, 0),
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
        flags.append(("W", f"{ins_lapses} coverage lapse(s) in last 3yr — financial stress indicator (+15–30%)"))
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
        flags.append(("T", f"[GAM Interaction] {t['name']} — {t['logic']} · H={t['h']:.3f} · {t['status']}"))

    return {
        # Scores
        "score_t1":      round(s_t1,    2),
        "score_glm":     round(s_glm,   2),
        "score_final":   round(s_final, 2),
        "t2_increment":  round(s_glm   - s_t1,   2),
        "t3_lift":       round(s_final - s_glm,   2),
        # η values
        "eta_t1":        round(eta_t1,    4),
        "eta_glm":       round(eta_glm,   4),
        "delta_gam":     round(delta_gam, 4),
        "eta_final":     round(eta_final, 4),
        # Multipliers
        "m_t1":          round(m_t1,  4),
        "m_t2":          round(m_t2,  4),
        "m_gam":         round(m_gam, 4),
        # Loss & pricing
        "loss_t1":       round(loss_t1,    0),
        "loss_glm":      round(loss_glm,   0),
        "loss_final":    round(loss_final, 0),
        "claim_freq":    round(claim_freq, 4),
        "claim_sev":     round(claim_sev,  0),
        "premium":       premium,
        # Decision
        "decision":      decision,
        "desc":          desc,
        # Contributions
        "tier1_contrib": tier1_contrib,
        "tier2_contrib": tier2_contrib,
        "tier3_contrib": tier3_contrib,
        # Interactions
        "triggered":     triggered,
        "ix_map":        ix_map,
        # Flags
        "flags":         flags,
    }