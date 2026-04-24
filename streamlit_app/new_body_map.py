# ═══ REALISTIC ANATOMICAL BODY MAP ═══
def render_body_map(organ_data, selected_organ):
    def clr(o): return risk_color_hex(organ_data[o]['risk']) if o==selected_organ else '#c8daea'
    def opa(o): return '1.0' if o==selected_organ else '0.28'
    def glow(o): return 'animation:organGlow 1.8s ease-in-out infinite;' if o==selected_organ else ''
    def strk(o): return '#ffffff' if o==selected_organ else '#7799bb'
    def sw(o): return '2.2' if o==selected_organ else '0.6'
    def flt(o): return "filter='url(#og)'" if o==selected_organ else ''
    svg=f"""
<svg viewBox="0 0 360 700" xmlns="http://www.w3.org/2000/svg" width="100%" style="max-width:290px;display:block;margin:auto;background:linear-gradient(160deg,#f0f4ff 0%,#e8f0fa 100%);border-radius:16px">
<defs>
  <radialGradient id="hG" cx="40%" cy="35%" r="65%"><stop offset="0%" stop-color="#FDDBB4"/><stop offset="60%" stop-color="#E8A876"/><stop offset="100%" stop-color="#C97840"/></radialGradient>
  <linearGradient id="bG" x1="0" y1="0" x2="1" y2="0"><stop offset="0%" stop-color="#C88048"/><stop offset="25%" stop-color="#EDB880"/><stop offset="50%" stop-color="#F5CC98"/><stop offset="75%" stop-color="#EDB880"/><stop offset="100%" stop-color="#C88048"/></linearGradient>
  <linearGradient id="lG" x1="0" y1="0" x2="1" y2="0"><stop offset="0%" stop-color="#C07840"/><stop offset="35%" stop-color="#E8B070"/><stop offset="65%" stop-color="#F2C890"/><stop offset="100%" stop-color="#C88050"/></linearGradient>
  <radialGradient id="sh" cx="50%" cy="30%" r="70%"><stop offset="0%" stop-color="rgba(255,255,255,0.22)"/><stop offset="100%" stop-color="rgba(0,0,0,0)"/></radialGradient>
  <filter id="og" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="6" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
  <filter id="ds"><feDropShadow dx="0" dy="5" stdDeviation="8" flood-color="#8060A0" flood-opacity="0.15"/></filter>
  <linearGradient id="rL" x1="0" y1="0" x2="1" y2="0"><stop offset="0%" stop-color="#10b981"/><stop offset="50%" stop-color="#f59e0b"/><stop offset="100%" stop-color="#ef4444"/></linearGradient>
  <clipPath id="torsoClip"><path d="M128,148 Q108,156 104,180 Q100,215 100,255 Q100,290 106,318 Q114,348 128,362 L232,362 Q246,348 254,318 Q260,290 260,255 Q260,215 256,180 Q252,156 232,148 Z"/></clipPath>
</defs>

<!-- shadow -->
<ellipse cx="180" cy="676" rx="66" ry="9" fill="rgba(80,60,120,0.13)"/>

<!-- FEET -->
<path d="M138,645 Q128,648 122,650 Q118,656 122,660 Q130,664 152,664 Q168,662 170,657 Q170,648 162,645 Z" fill="#C07840" stroke="#A06030" stroke-width="0.7"/>
<path d="M198,645 Q208,648 214,650 Q232,656 238,660 Q242,664 208,664 Q192,662 190,657 Q190,648 198,645 Z" fill="#C07840" stroke="#A06030" stroke-width="0.7"/>

<!-- LEFT LEG - single smooth path, no joints -->
<path d="M130,358 Q124,380 120,415 Q116,455 116,490 Q116,525 118,555 Q120,585 122,615 Q124,635 130,645 Q142,650 154,648 Q164,645 166,635 Q168,615 168,585 Q170,555 170,525 Q170,490 168,455 Q166,415 162,375 Q158,358 148,356 Z" fill="url(#lG)" stroke="#A86838" stroke-width="0.8"/>
<!-- knee shading L -->
<ellipse cx="143" cy="492" rx="18" ry="11" fill="rgba(160,100,50,0.22)"/>

<!-- RIGHT LEG - single smooth path -->
<path d="M230,358 Q236,375 238,400 Q242,438 242,472 Q242,508 240,540 Q238,568 236,595 Q234,620 232,638 Q230,648 218,650 Q206,652 196,648 Q190,640 190,625 Q190,600 190,572 Q190,540 190,508 Q190,472 190,438 Q190,400 190,372 Q190,356 212,356 Z" fill="url(#lG)" stroke="#A86838" stroke-width="0.8"/>
<!-- knee shading R -->
<ellipse cx="217" cy="492" rx="18" ry="11" fill="rgba(160,100,50,0.22)"/>

<!-- LEFT ARM - smooth taper, no shoulder ball -->
<path d="M116,152 Q102,162 94,185 Q86,215 83,248 Q80,278 80,305 Q80,325 84,338 Q90,348 100,348 Q110,348 116,338 Q122,325 122,305 Q122,280 124,248 Q126,215 126,185 Q124,162 116,152 Z" fill="url(#lG)" stroke="#A86838" stroke-width="0.8"/>
<!-- left forearm -->
<path d="M100,348 Q92,365 88,390 Q84,418 84,440 Q84,458 88,468 Q94,478 104,478 Q114,478 119,468 Q123,458 122,440 Q122,418 120,390 Q118,365 116,348 Z" fill="url(#lG)" stroke="#A86838" stroke-width="0.8"/>
<!-- left hand -->
<ellipse cx="103" cy="485" rx="16" ry="12" fill="#D8A870" stroke="#A86838" stroke-width="0.7"/>

<!-- RIGHT ARM -->
<path d="M244,152 Q258,162 266,185 Q274,215 277,248 Q280,278 280,305 Q280,325 276,338 Q270,348 260,348 Q250,348 244,338 Q238,325 238,305 Q238,280 236,248 Q234,215 234,185 Q236,162 244,152 Z" fill="url(#lG)" stroke="#A86838" stroke-width="0.8"/>
<!-- right forearm -->
<path d="M260,348 Q268,365 272,390 Q276,418 276,440 Q276,458 272,468 Q266,478 256,478 Q246,478 241,468 Q237,458 238,440 Q238,418 240,390 Q242,365 244,348 Z" fill="url(#lG)" stroke="#A86838" stroke-width="0.8"/>
<!-- right hand -->
<ellipse cx="257" cy="485" rx="16" ry="12" fill="#D8A870" stroke="#A86838" stroke-width="0.7"/>

<!-- TORSO - smooth hourglass, no ball joints -->
<path d="M128,148 Q108,156 104,180 Q100,215 100,255 Q100,290 106,318 Q114,348 128,362 L232,362 Q246,348 254,318 Q260,290 260,255 Q260,215 256,180 Q252,156 232,148 Q210,138 180,136 Q150,136 128,148 Z" fill="url(#bG)" stroke="#A86838" stroke-width="0.9" filter="url(#ds)"/>
<!-- torso highlight -->
<path d="M128,148 Q108,156 104,180 Q100,215 100,255 Q100,290 106,318 Q114,348 128,362 L232,362 Q246,348 254,318 Q260,290 260,255 Q260,215 256,180 Q252,156 232,148 Q210,138 180,136 Q150,136 128,148 Z" fill="url(#sh)"/>
<!-- collarbone line -->
<path d="M136,152 Q180,158 224,152" stroke="rgba(120,70,30,0.18)" stroke-width="1.2" fill="none"/>
<!-- sternum -->
<line x1="180" y1="162" x2="180" y2="310" stroke="rgba(120,70,30,0.12)" stroke-width="1"/>
<!-- ribs -->
<path d="M142,195 Q180,208 218,195" stroke="rgba(120,70,30,0.08)" stroke-width="1" fill="none"/>
<path d="M136,220 Q180,234 224,220" stroke="rgba(120,70,30,0.07)" stroke-width="1" fill="none"/>
<path d="M132,246 Q180,260 228,246" stroke="rgba(120,70,30,0.06)" stroke-width="1" fill="none"/>
<!-- abs -->
<path d="M166,320 Q180,326 194,320" stroke="rgba(120,70,30,0.1)" stroke-width="1" fill="none"/>
<path d="M164,340 Q180,346 196,340" stroke="rgba(120,70,30,0.08)" stroke-width="1" fill="none"/>
<ellipse cx="180" cy="305" rx="5" ry="3.5" fill="rgba(120,70,30,0.18)"/>

<!-- PELVIS -->
<path d="M128,362 Q120,370 118,385 Q116,400 122,410 Q132,420 156,422 Q168,423 180,423 Q192,423 204,422 Q228,420 238,410 Q244,400 242,385 Q240,370 232,362 Z" fill="url(#bG)" stroke="#A86838" stroke-width="0.8"/>

<!-- NECK -->
<path d="M160,118 Q158,128 158,140 Q158,152 163,157 Q172,162 180,162 Q188,162 197,157 Q202,152 202,140 Q202,128 200,118 Z" fill="#DDAA74" stroke="#A86838" stroke-width="0.8"/>

<!-- HEAD -->
<ellipse cx="180" cy="70" rx="55" ry="62" fill="url(#hG)" stroke="#C09060" stroke-width="1" filter="url(#ds)"/>
<ellipse cx="172" cy="48" rx="38" ry="30" fill="rgba(255,255,255,0.14)"/>

<!-- Hair -->
<path d="M127,40 Q132,8 162,2 Q180,-3 198,2 Q228,8 233,40 Q235,55 230,62 Q208,40 180,38 Q152,40 130,62 Z" fill="#2C1A08" stroke="#1A0E04" stroke-width="0.6"/>
<path d="M127,46 Q122,60 124,76" stroke="#2C1A08" stroke-width="5" fill="none" stroke-linecap="round"/>
<path d="M233,46 Q238,60 236,76" stroke="#2C1A08" stroke-width="5" fill="none" stroke-linecap="round"/>

<!-- Ears -->
<path d="M125,66 Q118,74 118,85 Q118,96 125,101 Q132,105 136,99 Q132,90 132,82 Q132,74 136,68Z" fill="#DDAA74" stroke="#A86838" stroke-width="0.8"/>
<path d="M235,66 Q242,74 242,85 Q242,96 235,101 Q228,105 224,99 Q228,90 228,82 Q228,74 224,68Z" fill="#DDAA74" stroke="#A86838" stroke-width="0.8"/>

<!-- Eyebrows -->
<path d="M152,70 Q166,63 180,66" stroke="rgba(60,35,12,0.65)" stroke-width="2" fill="none" stroke-linecap="round"/>
<path d="M180,66 Q194,63 208,70" stroke="rgba(60,35,12,0.65)" stroke-width="2" fill="none" stroke-linecap="round"/>
<!-- Eyes -->
<ellipse cx="165" cy="78" rx="10" ry="6.5" fill="#fff" stroke="rgba(100,60,25,0.25)" stroke-width="0.5"/>
<ellipse cx="195" cy="78" rx="10" ry="6.5" fill="#fff" stroke="rgba(100,60,25,0.25)" stroke-width="0.5"/>
<ellipse cx="165" cy="78" rx="5.5" ry="5.5" fill="#2A1808"/>
<ellipse cx="195" cy="78" rx="5.5" ry="5.5" fill="#2A1808"/>
<ellipse cx="163" cy="76" rx="2" ry="2" fill="rgba(255,255,255,0.8)"/>
<ellipse cx="193" cy="76" rx="2" ry="2" fill="rgba(255,255,255,0.8)"/>
<!-- Nose -->
<path d="M180,86 L177,101 Q180,104 183,101 Z" fill="rgba(150,85,40,0.18)"/>
<path d="M174,101 Q180,106 186,101" stroke="rgba(140,75,35,0.4)" stroke-width="1.3" fill="none"/>
<!-- Mouth -->
<path d="M167,113 Q180,121 193,113" stroke="rgba(130,60,45,0.65)" stroke-width="1.8" fill="none" stroke-linecap="round"/>

<!-- ══ ORGANS (rendered inside torso area) ══ -->

<!-- BRAIN -->
<ellipse cx="180" cy="58" rx="46" ry="52" fill="{clr('Brain')}" stroke="{strk('Brain')}" stroke-width="{sw('Brain')}" opacity="{opa('Brain')}" style="{glow('Brain')}" {flt('Brain')}/>
<path d="M148,42 Q157,33 166,42 Q173,33 180,42" stroke="rgba(255,255,255,0.45)" stroke-width="1.3" fill="none" opacity="{opa('Brain')}"/>
<path d="M180,42 Q187,33 196,42 Q203,33 212,42" stroke="rgba(255,255,255,0.45)" stroke-width="1.3" fill="none" opacity="{opa('Brain')}"/>
<line x1="180" y1="12" x2="180" y2="100" stroke="rgba(255,255,255,0.2)" stroke-width="1.2" opacity="{opa('Brain')}"/>
<text x="180" y="62" text-anchor="middle" font-family="JetBrains Mono,monospace" font-size="9" fill="#fff" font-weight="700" opacity="{opa('Brain')}">{organ_data['Brain']['risk']}%</text>

<!-- HEART -->
<path d="M170,185 Q157,170 148,173 Q136,178 136,192 Q136,204 150,217 Q162,228 170,234 L180,242 L190,234 Q198,228 210,217 Q224,204 224,192 Q224,178 212,173 Q203,170 190,185 Q186,178 170,185Z" fill="{clr('Heart')}" stroke="{strk('Heart')}" stroke-width="{sw('Heart')}" opacity="{opa('Heart')}" style="{glow('Heart')}" {flt('Heart')}/>
<text x="180" y="218" text-anchor="middle" font-family="JetBrains Mono,monospace" font-size="8" fill="#fff" font-weight="700" opacity="{opa('Heart')}">{organ_data['Heart']['risk']}%</text>

<!-- LUNGS -->
<path d="M118,180 Q106,187 102,206 Q98,228 102,252 Q106,272 118,283 Q130,290 138,279 Q146,266 146,244 Q146,218 142,200 Q138,184 128,178Z" fill="{clr('Lungs')}" stroke="{strk('Lungs')}" stroke-width="{sw('Lungs')}" opacity="{opa('Lungs')}" style="{glow('Lungs')}" {flt('Lungs')}/>
<path d="M242,180 Q254,187 258,206 Q262,228 258,252 Q254,272 242,283 Q230,290 222,279 Q214,266 214,244 Q214,218 218,200 Q222,184 232,178Z" fill="{clr('Lungs')}" stroke="{strk('Lungs')}" stroke-width="{sw('Lungs')}" opacity="{opa('Lungs')}" style="{glow('Lungs')}" {flt('Lungs')}/>
<text x="180" y="265" text-anchor="middle" font-family="JetBrains Mono,monospace" font-size="8" fill="#fff" font-weight="700" opacity="{opa('Lungs')}">{organ_data['Lungs']['risk']}%</text>

<!-- LIVER -->
<path d="M182,280 Q208,275 228,285 Q240,295 238,312 Q236,325 220,330 Q204,334 188,327 Q174,320 172,306 Q172,290 182,280Z" fill="{clr('Liver')}" stroke="{strk('Liver')}" stroke-width="{sw('Liver')}" opacity="{opa('Liver')}" style="{glow('Liver')}" {flt('Liver')}/>
<text x="204" y="309" text-anchor="middle" font-family="JetBrains Mono,monospace" font-size="8" fill="#fff" font-weight="700" opacity="{opa('Liver')}">{organ_data['Liver']['risk']}%</text>

<!-- KIDNEYS -->
<path d="M138,296 Q128,293 124,303 Q120,315 124,328 Q128,340 138,342 Q148,342 152,332 Q156,320 154,308 Q150,296 138,296Z" fill="{clr('Kidneys')}" stroke="{strk('Kidneys')}" stroke-width="{sw('Kidneys')}" opacity="{opa('Kidneys')}" style="{glow('Kidneys')}" {flt('Kidneys')}/>
<path d="M222,296 Q232,293 236,303 Q240,315 236,328 Q232,340 222,342 Q212,342 208,332 Q204,320 206,308 Q210,296 222,296Z" fill="{clr('Kidneys')}" stroke="{strk('Kidneys')}" stroke-width="{sw('Kidneys')}" opacity="{opa('Kidneys')}" style="{glow('Kidneys')}" {flt('Kidneys')}/>
<text x="180" y="326" text-anchor="middle" font-family="JetBrains Mono,monospace" font-size="8" fill="#fff" font-weight="700" opacity="{opa('Kidneys')}">{organ_data['Kidneys']['risk']}%</text>

<!-- BLOOD/AORTA -->
<path d="M172,235 Q170,262 168,292 Q166,322 166,352" stroke="{clr('Blood')}" stroke-width="10" fill="none" opacity="{opa('Blood')}" stroke-linecap="round" style="{glow('Blood')}"/>
<path d="M188,235 Q190,262 192,292 Q194,322 194,352" stroke="{clr('Blood')}" stroke-width="10" fill="none" opacity="{opa('Blood')}" stroke-linecap="round"/>
<text x="180" y="366" text-anchor="middle" font-family="JetBrains Mono,monospace" font-size="8" fill="{clr('Blood')}" font-weight="700" opacity="{opa('Blood')}">{organ_data['Blood']['risk']}%</text>

<!-- LABELS -->
<text x="40" y="58" font-family="JetBrains Mono,monospace" font-size="8.5" fill="#334e68" font-weight="700" text-anchor="end">Brain</text>
<line x1="42" y1="56" x2="134" y2="54" stroke="#90aac8" stroke-width="0.9" stroke-dasharray="3,2"/>
<text x="40" y="238" font-family="JetBrains Mono,monospace" font-size="8.5" fill="#334e68" font-weight="700" text-anchor="end">Lungs</text>
<line x1="42" y1="236" x2="102" y2="230" stroke="#90aac8" stroke-width="0.9" stroke-dasharray="3,2"/>
<text x="320" y="210" font-family="JetBrains Mono,monospace" font-size="8.5" fill="#334e68" font-weight="700">Heart</text>
<line x1="318" y1="208" x2="224" y2="206" stroke="#90aac8" stroke-width="0.9" stroke-dasharray="3,2"/>
<text x="320" y="308" font-family="JetBrains Mono,monospace" font-size="8.5" fill="#334e68" font-weight="700">Liver</text>
<line x1="318" y1="306" x2="238" y2="302" stroke="#90aac8" stroke-width="0.9" stroke-dasharray="3,2"/>
<text x="40" y="322" font-family="JetBrains Mono,monospace" font-size="8.5" fill="#334e68" font-weight="700" text-anchor="end">Kidneys</text>
<line x1="42" y1="320" x2="124" y2="316" stroke="#90aac8" stroke-width="0.9" stroke-dasharray="3,2"/>
<text x="320" y="354" font-family="JetBrains Mono,monospace" font-size="8.5" fill="#334e68" font-weight="700">Blood</text>
<line x1="318" y1="352" x2="194" y2="348" stroke="#90aac8" stroke-width="0.9" stroke-dasharray="3,2"/>

<!-- LEGEND -->
<rect x="70" y="686" width="220" height="7" rx="3.5" fill="url(#rL)" opacity="0.9"/>
<text x="70" y="700" font-family="JetBrains Mono,monospace" font-size="7.5" fill="#4a6080">Normal</text>
<text x="290" y="700" font-family="JetBrains Mono,monospace" font-size="7.5" fill="#4a6080" text-anchor="end">Critical</text>
</svg>"""
    return svg

