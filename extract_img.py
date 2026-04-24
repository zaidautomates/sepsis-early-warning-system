import base64
with open('streamlit_app/app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()
b64_str = ''
capture = False
for line in lines:
    if 'bg_img = """data:image' in line:
        capture = True
        continue
    if capture and '"""' in line:
        b64_str += line.replace('"""', '').strip()
        break
    if capture:
        b64_str += line.strip()

with open('streamlit_app/body_bg.jpg', 'wb') as f:
    f.write(base64.b64decode(b64_str.replace('\n','').replace('\r','')))
print('Image saved successfully to streamlit_app/body_bg.jpg')
