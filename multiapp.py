"""Frameworks for running multiple Streamlit applications as a single app.
"""
import streamlit as st
from house_app import fn_chrome_96_workaround


class MultiApp:
    """Framework for combining multiple streamlit applications.
    Usage:
        def foo():
            st.title("Hello Foo")
        def bar():
            st.title("Hello Bar")
        app = MultiApp()
        app.add_app("Foo", foo)
        app.add_app("Bar", bar)
        app.run()
    It is also possible keep each application in a separate file.
        import foo
        import bar
        app = MultiApp()
        app.add_app("Foo", foo.app)
        app.add_app("Bar", bar.app)
        app.run()
    """

    def __init__(self):
        self.apps = []

    def add_app(self, title, func):
        """Adds a new application.
        Parameters
        ----------
        func:
            the python function to render this app.
        title:
            title of the app. Appears in the dropdown in the sidebar.
        """
        self.apps.append({
            "title": title,
            "function": func
        })

    def run(self):
        fn_chrome_96_workaround()
        st.set_page_config(page_title="尋找夢想家", page_icon="🏠")
        st.sidebar.title("👨‍🏫 [Jack.Pan's](https://www.facebook.com/jack.pan.96/) 房市看板 ")
        st.sidebar.write('- ✉️ ssp6258@yahoo.com.tw')
        st.sidebar.write('- 🚧 [故障報修](https://github.com/SSP6258/house_app/issues/new)')
        st.sidebar.header('🧭 功能導航')
        app = st.sidebar.selectbox(
            '應用選單',
            self.apps,
            format_func=lambda app: app['title'],
            index=0)

        app['function']()
