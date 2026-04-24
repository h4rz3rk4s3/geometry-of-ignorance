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

PALETTES["cool_academic_binary"] = {
    0: "#005F73",       # Deep teal
    1: "#EE9B00"    # Amber
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

# 1. BRIGHT PROFESSIONAL (Vibrant but balanced, excellent energy)
# Clear differentiation with punchy colors
PALETTES["bright_professional"] = {
    "synthetic": "#4A90E2",      # Bright blue
    "GT": "#E8B339",             # Golden yellow
    "OOD": "#E85D75"             # Coral pink
}

# 2. MODERN VIBRANT (Bold, contemporary, high visibility)
# Strong saturation while maintaining professionalism
PALETTES["modern_vibrant"] = {
    "synthetic": "#2E86AB",      # Ocean blue
    "GT": "#A23B72",             # Magenta
    "OOD": "#F18F01"             # Bright orange
}

# 3. JEWEL TONES (Rich, saturated, sophisticated)
# Luxurious feel with excellent contrast
PALETTES["jewel_tones"] = {
    "synthetic": "#0077B6",      # Sapphire blue
    "GT": "#7209B7",             # Amethyst
    "OOD": "#F77F00"             # Amber
}

# 4. CHEERFUL SCIENTIFIC (Bright yet neutral associations)
# Energetic without being overwhelming
PALETTES["cheerful_scientific"] = {
    "synthetic": "#06AED5",      # Cyan
    "GT": "#DD1C77",             # Raspberry
    "OOD": "#F2A900"             # Marigold
}

# 5. BOLD CLARITY (Maximum pop, presentation-ready)
# Excellent for grabbing attention while staying clear
PALETTES["bold_clarity"] = {
    "synthetic": "#1E88E5",      # Vivid blue
    "GT": "#D81B60",             # Pink-red
    "OOD": "#FFB300"             # Sunny yellow
}

# 6. TROPICAL PROFESSIONAL (Energetic, fresh, distinctive)
# Uncommon in academic papers, memorable
PALETTES["tropical_professional"] = {
    "synthetic": "#0096C7",      # Bright teal 0096C7
    "GT": "#9D4EDD",            # Bright purple 9D4EDD
    "OOD": "#E9A41D"             # Tangerine FF9E00
}

PALETTES["architectural_triad"] = {
    "synthetic": "#2D3436",    # Deep Charcoal
    "GT": "#0984E3",           # Strong Blue (Standard reference)
    "OOD": "#D63031"          # Muted Crimson (Distinct, not "warning" red)
}

PALETTES["earth_and_steel"] = {
    "synthetic": "#4A4E69",    # Slate Blue/Grey
    "GT": "#9A8C98",           # Muted Mauve (Neutral middle)
    "OOD": "#C9ADA7"          # Desert Sand (Warm but professional)
}

PALETTES["quantitative_contrast"] = {
    "synthetic": "#1B263B",    # Inky Navy
    "GT": "#415A77",           # Storm Blue
    "OOD": "#E0E1DD"          # Platinum/Grey-White (Use with a thin black edge)
}

PALETTES["blue_green"] = {
    "synthetic": "#0056C7",    # Inky Navy
    "GT": "#0096C7",           # Storm Blue
    "OOD": "#00C7B9"          # Platinum/Grey-White (Use with a thin black edge)
}