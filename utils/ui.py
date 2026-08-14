"""
ui.py - SVG icons, illustration and shared HTML fragments used by app.py.

These are pure strings - no project logic lives here.
"""

# ------------------------- Icon helpers -------------------------
STROKE = 'fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"'


def icon(name, size=24, sw="1.9"):
    """Return an inline SVG for a named icon (stroke follows text color)."""
    paths = {
        "chip": (
            '<rect x="6" y="6" width="12" height="12" rx="2.5"/>'
            '<rect x="10" y="10" width="4" height="4" rx="1"/>'
            '<path d="M9 6V3 M15 6V3 M9 21v-3 M15 21v-3 M6 9H3 M6 15H3 M21 9h-3 M21 15h-3"/>'
        ),
        "bolt": '<path d="M13 2 4.5 13.5H11L10 22l8.5-11.5H12L13 2Z" fill="currentColor" stroke="none"/>',
        "leaf": (
            '<path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 10-10 10Z" fill="currentColor" stroke="none"/>'
            '<path d="M2 21c0-3 1.85-5.36 5.08-6C9.5 14.52 12 13 13 12" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round"/>'
        ),
        "check": '<path d="M20 6 9 17l-5-5"/>',
        "info": (
            '<circle cx="12" cy="12" r="10"/>'
            '<path d="M12 16v-4 M12 8h.01"/>'
        ),
        "activity": '<path d="M22 12h-4l-3 9L9 3l-3 9H2"/>',
        "shield": (
            '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10Z"/>'
            '<path d="M9 12l2 2 4-4"/>'
        ),
        "wrench": (
            '<path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76Z"/>'
        ),
        "code": '<path d="m16 18 6-6-6-6M8 6l-6 6 6 6"/>',
        "braces": (
            '<path d="M8 3H7a2 2 0 0 0-2 2v5a2 2 0 0 1-2 2 2 2 0 0 1 2 2v5c0 1.1.9 2 2 2h1"/>'
            '<path d="M16 21h1a2 2 0 0 0 2-2v-5c0-1.1.9-2 2-2a2 2 0 0 1-2-2V5a2 2 0 0 0-2-2h-1"/>'
        ),
        "layers": (
            '<path d="M12 2 2 7l10 5 10-5-10-5Z"/>'
            '<path d="m2 17 10 5 10-5"/>'
            '<path d="m2 12 10 5 10-5"/>'
        ),
        "network": (
            '<circle cx="18" cy="5" r="3"/><circle cx="6" cy="12" r="3"/><circle cx="18" cy="19" r="3"/>'
            '<path d="M8.6 13.5l6.8 3M15.4 7.5l-6.8 3"/>'
        ),
        "eye": (
            '<path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z"/>'
            '<circle cx="12" cy="12" r="3"/>'
        ),
        "waves": (
            '<path d="M2 6c.6.5 1.2 1 2.5 1C7 7 7 5 9.5 5s2.5 2 5 2 2.5-2 5-2c1.3 0 1.9.5 2.5 1"/>'
            '<path d="M2 12c.6.5 1.2 1 2.5 1 2.5 0 2.5-2 5-2s2.5 2 5 2 2.5-2 5-2c1.3 0 1.9.5 2.5 1"/>'
            '<path d="M2 18c.6.5 1.2 1 2.5 1 2.5 0 2.5-2 5-2s2.5 2 5 2 2.5-2 5-2c1.3 0 1.9.5 2.5 1"/>'
        ),
        "arrow": '<path d="M5 12h14M13 6l6 6-6 6"/>',
    }
    inner = paths.get(name, paths["info"])
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" {STROKE} '
        f'stroke-width="{sw}" aria-hidden="true">{inner}</svg>'
    )


def logo(size=26, color="#ffffff"):
    """Small leaf logo mark used in the sidebar brand and footer."""
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
        f'stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
        '<path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 10-10 10Z" '
        f'fill="{color}" stroke="none"/>'
        '<path d="M2 21c0-3 1.85-5.36 5.08-6C9.5 14.52 12 13 13 12"/>'
        "</svg>"
    )


# ------------------------- Hero illustration -------------------------
def hero_art():
    """Modern AI-agriculture illustration (pure inline SVG, no assets).

    Blank lines are stripped so the markup stays one contiguous HTML block
    when injected - otherwise Markdown could parse it as a code block with
    a copy button (which breaks ClipboardJS in Streamlit).
    """
    art = """
    <svg viewBox="0 0 560 420" width="560" height="420" role="img" aria-label="AI crop analysis illustration" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <linearGradient id="leafGrad" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stop-color="#166534"/>
          <stop offset="55%" stop-color="#22a352"/>
          <stop offset="100%" stop-color="#3ec46e"/>
        </linearGradient>
        <linearGradient id="chipGrad" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stop-color="#ffffff"/>
          <stop offset="100%" stop-color="#eef9f2"/>
        </linearGradient>
        <filter id="softShadow" x="-30%" y="-30%" width="160%" height="160%">
          <feDropShadow dx="0" dy="10" stdDeviation="12" flood-color="#0f3d24" flood-opacity="0.12"/>
        </filter>
      </defs>

      <circle cx="468" cy="86" r="128" fill="#e8f6ee" opacity="0.85"/>
      <circle cx="92" cy="352" r="150" fill="#eef9f2" opacity="0.9"/>
      <rect x="380" y="250" width="120" height="120" rx="24" fill="#f2faf4" opacity="0.7"/>

      <circle cx="96" cy="96" r="4.5" fill="#9adbb2"/>
      <circle cx="470" cy="182" r="5" fill="#9adbb2"/>
      <circle cx="150" cy="88" r="3.5" fill="#9adbb2"/>
      <circle cx="432" cy="332" r="4" fill="#9adbb2"/>

      <path d="M70 232 L490 232" stroke="#3ec46e" stroke-dasharray="2 10" stroke-width="3" stroke-linecap="round" opacity="0.45"/>

      <g filter="url(#softShadow)">
        <path d="M280 64 C 398 78, 482 168, 482 236 C 482 336, 398 404, 280 404 C 162 404, 78 336, 78 236 C 78 168, 162 78, 280 64 Z" fill="url(#leafGrad)"/>
        <path d="M280 92 L280 376" stroke="#ffffff" stroke-opacity="0.5" stroke-width="3.5" stroke-linecap="round"/>
        <path d="M280 198 C 246 192, 216 180, 186 152 M280 198 C 314 192, 344 180, 374 152"
              stroke="#ffffff" stroke-opacity="0.4" stroke-width="3" stroke-linecap="round" fill="none"/>
        <path d="M280 250 C 240 248, 204 240, 172 224 M280 250 C 320 248, 356 240, 388 224"
              stroke="#ffffff" stroke-opacity="0.4" stroke-width="3" stroke-linecap="round" fill="none"/>
        <path d="M280 302 C 246 300, 214 292, 184 278 M280 302 C 314 300, 346 292, 376 278"
              stroke="#ffffff" stroke-opacity="0.4" stroke-width="3" stroke-linecap="round" fill="none"/>
      </g>

      <path d="M120 96 C 160 104, 184 132, 184 152 C 184 188, 152 212, 120 212 C 88 212, 56 188, 56 152 C 56 132, 80 104, 120 96 Z" fill="#a7e3bd" opacity="0.95"/>
      <path d="M120 120 L120 188 M104 156 C 112 152, 116 150, 120 148 M136 156 C 128 152, 124 150, 120 148"
            stroke="#ffffff" stroke-opacity="0.65" stroke-width="2.6" stroke-linecap="round" fill="none"/>

      <path d="M400 296 C 430 300, 448 322, 448 336 C 448 362, 430 380, 400 380 C 376 380, 352 362, 352 336 C 352 322, 370 300, 400 296 Z" fill="#a7e3bd"/>
      <path d="M400 312 L400 364 M386 338 C 392 336, 396 334, 400 334 M414 338 C 408 336, 404 334, 400 334"
            stroke="#ffffff" stroke-opacity="0.65" stroke-width="2.4" stroke-linecap="round" fill="none"/>

      <g filter="url(#softShadow)" transform="translate(408 56) rotate(-4)">
        <rect width="66" height="46" rx="12" fill="url(#chipGrad)" stroke="#d9efdf"/>
        <rect x="12" y="12" width="42" height="22" rx="6" fill="#0f3d24" opacity="0.10"/>
        <circle cx="33" cy="23" r="6.5" fill="#1b7a3d"/>
        <rect x="-8" y="10" width="8" height="4.5" rx="1.6" fill="#9adbb2"/>
        <rect x="13" y="-7" width="4.5" height="7" rx="1.6" fill="#9adbb2"/>
        <rect x="28" y="-7" width="4.5" height="7" rx="1.6" fill="#9adbb2"/>
        <rect x="43" y="-7" width="4.5" height="7" rx="1.6" fill="#9adbb2"/>
        <rect x="13" y="46" width="4.5" height="7" rx="1.6" fill="#9adbb2"/>
        <rect x="28" y="46" width="4.5" height="7" rx="1.6" fill="#9adbb2"/>
        <rect x="43" y="46" width="4.5" height="7" rx="1.6" fill="#9adbb2"/>
        <rect x="66" y="14" width="8" height="4.5" rx="1.6" fill="#9adbb2"/>
        <rect x="66" y="27" width="8" height="4.5" rx="1.6" fill="#9adbb2"/>
      </g>

      <g filter="url(#softShadow)" transform="translate(96 268)">
        <rect width="120" height="74" rx="16" fill="#ffffff" stroke="#e3ece5"/>
        <circle cx="34" cy="37" r="24" stroke="#1b7a3d" stroke-width="2.5" fill="#f0faf4"/>
        <path d="M34 15 V20 M34 54 V59 M12 37 H17 M51 37 H56" stroke="#1b7a3d" stroke-width="2" stroke-linecap="round"/>
        <path d="M34 24 C 42 25, 46 31, 46 32 C 46 37, 42 40, 35 41 C 33 38, 33 36, 34 33 Z" fill="#3ec46e"/>
        <circle cx="34" cy="26" r="2.4" fill="#166534"/>
        <rect x="70" y="20" width="38" height="7" rx="3.5" fill="#d9efdf"/>
        <rect x="70" y="34" width="30" height="7" rx="3.5" fill="#e7f5ec"/>
        <rect x="70" y="48" width="24" height="7" rx="3.5" fill="#22a352"/>
      </g>

      <path d="M494 106 h12 M500 100 v12" stroke="#22a352" stroke-width="3.4" stroke-linecap="round"/>
      <path d="M56 128 h10 M61 123 v10" stroke="#3ec46e" stroke-width="3" stroke-linecap="round"/>
      <path d="M512 300 h10 M517 295 v10" stroke="#9adbb2" stroke-width="3" stroke-linecap="round"/>
    </svg>
    """
    return "\n".join(line for line in art.splitlines() if line.strip()) + "\n"


# ------------------------- Brand / footer fragments -------------------------
def sidebar_brand():
    """Brand header block for the sidebar."""
    return f"""
    <div class="sidebar-brand">
      <div class="logo">{logo(26)}</div>
      <div>
        <div class="brand-name">CropAI</div>
        <div class="brand-sub">Smart Crop Health Detection</div>
      </div>
    </div>
    <div class="sidebar-divider"></div>
    """


def model_status_card(loaded):
    """Sidebar model status card. loaded: bool."""
    if loaded:
        return (
            '<div class="model-status ok"><span class="dot"></span>'
            "CNN model ready &amp; loaded</div>"
        )
    return (
        '<div class="model-status warn"><span class="dot"></span>'
        "Model not trained yet &mdash; run <code>python train.py</code></div>"
    )


def footer_section():
    """Professional footer shown at the bottom of every page."""
    bolt = icon("bolt", 14)
    return f"""
    <div class="footer">
      <div class="f-brand">{logo(22, "#1b7a3d")} CropAI</div>
      <div class="f-tag">AI-Based Crop Disease Detection</div>
      <div class="f-power">{bolt} Powered by Deep Learning &amp; Computer Vision</div>
      <div class="f-disclaimer">
        Educational project for research &amp; demonstration. Always confirm AI
        diagnoses with a local agricultural expert before applying treatments.
      </div>
    </div>
    """