"""Frameworks for running multiple Streamlit applications as a single app.
"""
import streamlit as st
from house_app import fn_chrome_96_workaround, fn_show_img
from house_utils import dic_of_path
try:
    from streamlit_player import st_player
except:
    pass


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

        try:
            with st.sidebar:
                play = st.checkbox('▶️', value=True)
                music_url="https://soundcloud.com/audio-library-478708792/leaning-on-the-everlasting-arms-zachariah-hickman-audio-library-free-music?utm_source=clipboard&utm_medium=text&utm_campaign=social_sharing"
                st_player(music_url, playing=play, loop=True, volume=0.5, height=220)
        except:
            imgs = ['house_dora.JPG', 'house_dora.jpg', 'house_dora_2.jpg']
            fn_show_img(dic_of_path['database'], imgs[2], is_sidebar=True, width=None, caption="~ by Dora ~")

        st.sidebar.title("👨‍🏫 [Jack.Pan's](https://www.facebook.com/jack.pan.96/) 房市看板 ")
        st.sidebar.write('- ✉️ ssp6258@yahoo.com.tw')
        st.sidebar.write('- 🚧 [故障報修、意見反饋](https://github.com/SSP6258/house_app/issues/new)')
        st.sidebar.header('🧭 功能導航')
        app = st.sidebar.selectbox(
            '應用選單',
            self.apps,
            format_func=lambda app: app['title'],
            index=0)

        app['function']()
