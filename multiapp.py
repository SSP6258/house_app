"""Frameworks for running multiple Streamlit applications as a single app.
"""
import random
import datetime
import streamlit as st
from random import randint
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

        dic_imgs = {
            'house_dora.JPG': '',
            'house_dora.jpg': '',
            'house_dora_2.jpg': '',
            'house_dora_88.jpg': '~ ❤️女兒小一時送的父親節禮物 一棟別墅 ~',
            'house_sunset.jpg': '~ 🌅 八里夕陽 ~',
            'house_peace_island_s.jpg': '~ 🏝️ 和平島 ~',
            'house_0919.jpg': '~ 🙏 天佑台灣 ~',
            'house_ocean_1.jpg': '~ 🌊 海闊天空 ~',
            'house_view.JPG': '~ 🎑 家的視野 ~',
            'house_green.JPG': '~ 綠光 ~',
            'house_cloud.JPG': '~ 5 AM Club ~',
            'house_0108.JPG': '~ 女兒小二時的勞作 ~',
            'me.jpg': '~ 小奇萊 3152 ~',
            'tree.JPG': '~ 合歡北峰名樹 ~',
            'bird.jpg': '~ 合歡山日出 ~',
        }

        img = 'me.jpg'
        fn_show_img(dic_of_path['database'], img, is_sidebar=True, width=None, caption=dic_imgs[img])

        # try:
        #     with st.sidebar:
        #         music = {
        #             1: "https://soundcloud.com/audio-library-478708792/leaning-on-the-everlasting-arms-zachariah-hickman-audio-library-free-music?utm_source=clipboard&utm_medium=text&utm_campaign=social_sharing",
        #             2: "https://soundcloud.com/user-443256645/esther-abrami-no-9-esthers?utm_source=clipboard&utm_medium=text&utm_campaign=social_sharing",
        #             3: "https://soundcloud.com/audio_lava/hulu-ukulele?utm_source=clipboard&utm_medium=text&utm_campaign=social_sharing"
        #         }
        #         i = random.randint(1, len(music))
        #         st_player(music[i], key=str(datetime.datetime.now()), playing=False, loop=True, volume=0.1, height=220)
        # except:
        #     imgs = ['house_dora.JPG', 'house_dora.jpg', 'house_dora_2.jpg', 'house_dora_88.jpg']
        #     fn_show_img(dic_of_path['database'], imgs[3], is_sidebar=True, width=None, caption="~ 女兒蓋了棟別墅送我 💖 ~")

        st.sidebar.markdown("## 👨‍🏫 [$Jack.Pan's$](https://www.facebook.com/jack.pan.96/) $房市看板$ ")
        # st.sidebar.write('- ✉️ ssp6258@yahoo.com.tw')
        # st.sidebar.write('- 🚧 [故障報修、意見反饋](https://github.com/SSP6258/house_app/issues/new)')
        st.sidebar.header('🧭 $功能導航$')
        app = st.sidebar.selectbox(
            '應用選單',
            self.apps,
            format_func=lambda app: app['title'],
            index=1,
            label_visibility='collapsed'
            )

        app['function']()


