import base64
import textwrap

image_path = r'C:\Users\ZAID-PC\.gemini\antigravity\brain\3db3e785-ef2b-48e5-acd1-53383bfeafb0\medical_body_3d_1776985802011.png'
with open(image_path, 'rb') as f:
    b64_bytes = base64.b64encode(f.read())
    b64_string = b64_bytes.decode('utf-8')

# Wrap the base64 string to 64 chars per line so Streamlit markdown doesn't choke on a massive single line
wrapped_b64 = '\n'.join(textwrap.wrap(b64_string, 64))

b64_data_uri = f"data:image/png;base64,\n{wrapped_b64}"

svg_template = f'''# ═══ REALISTIC ANATOMICAL BODY MAP ═══
def render_body_map(organ_data, selected_organ):
    def clr(o): return risk_color_hex(organ_data[o]['risk']) if o==selected_organ else '#c8daea'
    def opa(o): return '0.85' if o==selected_organ else '0.0'
    def glow(o): return 'filter="url(#glow)"' if o==selected_organ else ''
    def strk(o): return '#ffffff' if o==selected_organ else 'none'
    
    bg_img = """{b64_data_uri}"""
    
    svg = f"""
<div style="background: white; border-radius: 16px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); overflow: hidden; margin: 0 auto; max-width: 400px; padding: 10px;">
<svg viewBox="0 0 500 900" xmlns="http://www.w3.org/2000/svg" style="width:100%; display:block; margin:auto;">
    <defs>
        <filter id="glow" x="-30%" y="-30%" width="160%" height="160%">
            <feGaussianBlur stdDeviation="8" result="blur" />
            <feComposite in="SourceGraphic" in2="blur" operator="over"/>
        </filter>
        <linearGradient id="legendGrad" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stop-color="#10b981"/>
            <stop offset="50%" stop-color="#f59e0b"/>
            <stop offset="100%" stop-color="#ef4444"/>
        </linearGradient>
    </defs>

    <!-- 3D Background Image -->
    <image href="{{bg_img}}" x="0" y="0" width="500" height="900" preserveAspectRatio="xMidYMid slice" opacity="1.0" />

    <!-- ORGANS LAYER -->
    <!-- BRAIN -->
    <g {{glow('Brain')}}>
        <path d="M 250, 50 C 200, 50 210, 110 250, 130 C 290, 110 300, 50 250, 50 Z" 
              fill="{{clr('Brain')}}" stroke="{{strk('Brain')}}" stroke-width="2" opacity="{{opa('Brain')}}"/>
        <text x="250" y="100" text-anchor="middle" font-family="Arial" font-size="16" font-weight="bold" fill="#fff" opacity="{{opa('Brain')}}">{{organ_data['Brain']['risk']}}%</text>
    </g>

    <!-- LUNGS -->
    <g {{glow('Lungs')}}>
        <path d="M 180, 220 C 130, 250 140, 360 170, 400 C 200, 430 220, 390 220, 350 C 220, 300 200, 250 180, 220 Z" 
              fill="{{clr('Lungs')}}" stroke="{{strk('Lungs')}}" stroke-width="2" opacity="{{opa('Lungs')}}"/>
        <path d="M 320, 220 C 370, 250 360, 360 330, 400 C 300, 430 280, 390 280, 350 C 280, 300 300, 250 320, 220 Z" 
              fill="{{clr('Lungs')}}" stroke="{{strk('Lungs')}}" stroke-width="2" opacity="{{opa('Lungs')}}"/>
        <text x="250" y="320" text-anchor="middle" font-family="Arial" font-size="16" font-weight="bold" fill="#fff" opacity="{{opa('Lungs')}}">{{organ_data['Lungs']['risk']}}%</text>
    </g>

    <!-- HEART -->
    <g {{glow('Heart')}}>
        <path d="M 270, 280 C 220, 250 210, 310 250, 350 C 280, 370 290, 400 320, 350 C 350, 310 310, 250 270, 280 Z" 
              fill="{{clr('Heart')}}" stroke="{{strk('Heart')}}" stroke-width="2" opacity="{{opa('Heart')}}"/>
        <text x="275" y="330" text-anchor="middle" font-family="Arial" font-size="14" font-weight="bold" fill="#fff" opacity="{{opa('Heart')}}">{{organ_data['Heart']['risk']}}%</text>
    </g>

    <!-- LIVER -->
    <g {{glow('Liver')}}>
        <path d="M 160, 410 C 130, 470 170, 520 230, 520 C 300, 520 340, 460 340, 420 C 280, 390 200, 390 160, 410 Z" 
              fill="{{clr('Liver')}}" stroke="{{strk('Liver')}}" stroke-width="2" opacity="{{opa('Liver')}}"/>
        <text x="240" y="470" text-anchor="middle" font-family="Arial" font-size="16" font-weight="bold" fill="#fff" opacity="{{opa('Liver')}}">{{organ_data['Liver']['risk']}}%</text>
    </g>

    <!-- KIDNEYS -->
    <g {{glow('Kidneys')}}>
        <path d="M 180, 520 C 150, 510 140, 560 160, 600 C 190, 610 200, 560 180, 520 Z" 
              fill="{{clr('Kidneys')}}" stroke="{{strk('Kidneys')}}" stroke-width="2" opacity="{{opa('Kidneys')}}"/>
        <path d="M 320, 520 C 350, 510 360, 560 340, 600 C 310, 610 300, 560 320, 520 Z" 
              fill="{{clr('Kidneys')}}" stroke="{{strk('Kidneys')}}" stroke-width="2" opacity="{{opa('Kidneys')}}"/>
        <text x="250" y="570" text-anchor="middle" font-family="Arial" font-size="16" font-weight="bold" fill="#444" opacity="{{opa('Kidneys')}}">{{organ_data['Kidneys']['risk']}}%</text>
    </g>

    <!-- BLOOD/AORTA -->
    <g {{glow('Blood')}}>
        <path d="M 250, 350 L 250, 620 M 250, 620 L 190, 850 M 250, 620 L 310, 850" 
              fill="none" stroke="{{clr('Blood')}}" stroke-width="12" opacity="{{opa('Blood')}}" stroke-linecap="round"/>
        <text x="250" y="650" text-anchor="middle" font-family="Arial" font-size="16" font-weight="bold" fill="{{clr('Blood')}}" opacity="{{opa('Blood')}}">{{organ_data['Blood']['risk']}}%</text>
    </g>

    <!-- LEGEND -->
    <rect x="100" y="860" width="300" height="10" rx="5" fill="url(#legendGrad)" opacity="0.9"/>
    <text x="100" y="890" font-family="Arial" font-size="14" fill="#64748b">Normal</text>
    <text x="400" y="890" font-family="Arial" font-size="14" fill="#64748b" text-anchor="end">Critical</text>

</svg>
</div>
"""
    return svg
'''

c = open('streamlit_app/app.py', encoding='utf-8').read()
s = c.find('# ═══ REALISTIC ANATOMICAL BODY MAP ═══')
e = c.find('\n\nwith st.sidebar:', s)
out = c[:s] + svg_template + c[e:]
with open('streamlit_app/app.py', 'w', encoding='utf-8') as f:
    f.write(out)

print('Updated app.py with wrapped base64')
