import base64

def get_base64_image():
    with open('streamlit_app/stitch_body.png', 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

# The current app.py has double brackets for organ data rendering, we will fix this.
