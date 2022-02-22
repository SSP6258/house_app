from multiapp import MultiApp
from apps import app_eda, app_train, app_inference, app_reference, app_tools, app_projs

app = MultiApp()

# Add all your application here
app.add_app("🕵‍♂ 資料探勘", app_eda.app)
app.add_app("🏋‍♂ 模型訓練", app_train.app)
app.add_app("🧙 模型推論", app_inference.app)
app.add_app("📚 參考資料", app_reference.app)
app.add_app("🧰 開發工具", app_tools.app)
app.add_app("🗃️ 其它專案", app_projs.app)
# The main app
app.run()
