from multiapp import MultiApp
from apps import app_eda, app_train, app_inference, app_reference, app_tools, app_projs, app_intro

app = MultiApp()

# Add all your application here
app.add_app("๐จโ๐ซ ็ถฒ็ซไป็ดน", app_intro.app)
app.add_app("๐ตโโ ่ณๆๆขๅ", app_eda.app)
app.add_app("๐โโ ๆจกๅ่จ็ทด", app_train.app)
app.add_app("๐ง ๆจกๅๆจ่ซ", app_inference.app)
app.add_app("๐ ๅ่่ณๆ", app_reference.app)
app.add_app("๐งฐ ้็ผๅทฅๅท", app_tools.app)
app.add_app("๐๏ธ ๅถๅฎๅฐๆก", app_projs.app)
# The main app
app.run()
