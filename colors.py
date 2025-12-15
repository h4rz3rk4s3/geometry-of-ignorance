PALETTES = {}

# ============================================================================
# COLOR PALETTES FOR YOUR PAPER
# ============================================================================

# 1. PROFESSIONAL NEUTRAL (No bias, clean, academic)
# Works well in print, avoids emotional associations
PALETTES["professional_neutral"] = {
    "Knowledge": "#2E86AB",      # Calm blue
    "Neutral": "#A23B72",         # Muted purple
    "Non-Knowledge": "#F18F01"    # Warm orange
}

PALETTES["professional_shap"] = {
    "Knowledge": "#2E86AB",      # Calm blue
    "Neutral": "#FFFFFF",         # Muted purple
    "Non-Knowledge": "#F18F01"    # Warm orange
}

# PALETTES["professional_neutral"] = {
#     "Knowledge": "#2E86AB",      # Calm blue
#     "Neutral": "#A23B72",         # Muted purple
#     "Non-Knowledge": "#F18F01"    # Warm orange
# }

# 2. EARTH TONES (Warm, natural, no negative connotations)
# Good for avoiding red/green "good/bad" associations
PALETTES["earth_tones"] = {
    "Knowledge": "#06A77D",       # Teal/green
    "Neutral": "#D5A021",         # Gold
    "Non-Knowledge": "#C73E1D"    # Terracotta
}

# 3. COOL ACADEMIC (Modern, scientific)
# Blues and teals - professional and neutral
PALETTES["cool_academic"] = {
    "Knowledge": "#005F73",       # Deep teal
    "Neutral": "#94D2BD",         # Light teal
    "Non-Knowledge": "#EE9B00"    # Amber
}

PALETTES["cool_academic_shap"] = {
    "Knowledge": "#005F73",       # Deep teal
    "Neutral": "#FFFFFF",         # white
    "Non-Knowledge": "#EE9B00"    # Amber
}

# 4. COLORBLIND OPTIMIZED (Deuteranopia/Protanopia friendly)
# Based on Wong's palette for color blindness
# Avoids red-green confusion, high contrast
PALETTES["colorblind_safe"] = {
    "Knowledge": "#0173B2",       # Blue (distinct)
    "Neutral": "#ECE133",         # Yellow (high visibility)
    "Non-Knowledge": "#DE8F05"    # Orange (distinct from blue)
}

# 5. GRAYSCALE OPTIMIZED (Works perfectly in B&W printing)
# Maximum contrast, clearly distinguishable in grayscale
PALETTES["grayscale_optimized"] = {
    "Knowledge": "#000000",       # Black (darkest)
    "Neutral": "#808080",         # Medium gray
    "Non-Knowledge": "#D3D3D3"    # Light gray
}

# 6. VIRIDIS-INSPIRED (Perceptually uniform, colorblind friendly)
# Popular in scientific visualization, naturally ordered
PALETTES["viridis_inspired"] = {
    "Knowledge": "#440154",       # Deep purple
    "Neutral": "#21908C",         # Teal
    "Non-Knowledge": "#FDE724"    # Yellow
}

# 7. PASTEL PROFESSIONAL (Soft, modern, non-threatening)
# Reduces visual fatigue, good for presentations
PALETTES["pastel_professional"] = {
    "Knowledge": "#6A9FB5",       # Soft blue
    "Neutral": "#AA96B0",         # Lavender
    "Non-Knowledge": "#E8A87C"    # Peach
}

# 8. HIGH CONTRAST (Bold, clear, excellent for projectors)
# Maximum differentiation for presentations
PALETTES["high_contrast"] = {
    "Knowledge": "#003F5C",       # Navy
    "Neutral": "#BC5090",         # Magenta
    "Non-Knowledge": "#FFA600"    # Bright orange
}