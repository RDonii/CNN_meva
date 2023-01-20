import streamlit as st
from fastai.vision.all import *
from plotly import express as px

st.set_page_config(
    page_title="Mevalar",
    page_icon="🍏",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# Sidebar
with st.sidebar:
    st.header('Modellar')
    st.write('''
        - <a href="https://drive.google.com/file/d/1hMqUjQGT_aJan4XL1Ari9z-mvYOx3DlS/view?usp=share_link" style="color:green;" target="_blank">fruits_filter.pkl</a>
        - <a href="https://drive.google.com/file/d/1_wfwQNWlERAXWKur-nnWpZi13KLu5nRB/view?usp=share_link" style="color:green;" target="_blank">fruits.pkl</a>
    ''', unsafe_allow_html=True)

    st.header('Train qilingan colablar:')
    st.write('''
        - <a href="https://colab.research.google.com/drive/12hZ9ZhEMYVovVDYwuDyoyAc1jwqYyZE7?usp=sharing" style="color:green;" target="_blank">fruits</a>
        - <a href="https://colab.research.google.com/drive/1jeTKtDIbsKRsQxLglRSDyAgXF5LLkpJq?usp=sharing" style="color:green;" target="_blank">fruits_filter</a>
    ''', unsafe_allow_html=True)

    st.header('Foydalanilgan datasetlar:')
    st.write('''
        - <a href="https://www.kaggle.com/datasets/itsahmad/indoor-scenes-cvpr-2019" style="color:green;" target="_blank">MIT Indoor Scenes</a>
        - <a href="https://storage.googleapis.com/openimages/web/factsfigures_v4.html" style="color:green;" target="_blank">Open Images Dataset V4</a>
    ''', unsafe_allow_html=True)

    st.header('Github:')
    st.write('''   <a href="https://github.com/RDonii/CNN_meva" style="color:green;" target="_blank">CNN_Mevalar</a>''', unsafe_allow_html=True)

    st.write("""
    ---
    *Streamlitni tanishtirganingiz uchun alohida raxmat aytmoqchiman.* 😊
    """)


# loading
lbs = {
    'Apple': 'Olma',
    'Pear': 'Nok',
    'Banana': 'Banan',
    'Lemon': 'Limon', 
    'Tomato': 'Pomidor',
    'Grape': 'Uzum'
}
main_ml = load_learner('fruits.pkl')
filter_ml = load_learner('fruits_filter.pkl')


# Main page
st.title('Mevalar')

with st.expander("Qo'llanma"):
    st.write('''
        Loyihada ikkita DL modeli train va deploy qilindi.
        Yuklangan rasim foydalanuvchi ixtiyoriga qarab filter qilinadi va quyidagi obyektlardan birini rasimdan topishga xarakat qiladi:
        - Olma
        - Nok
        - Banan
        - Limon
        - Pomidor
        - Uzum

        *Resurslarni chap yuqoridagi tugmani bosish orqali saydbardan topishingiz mumkin*.
    ''')
    
    st.warning("Filter modelni train qilishda manfiy klass uchun ishlatilingan dataset uy ichkarisida olingan rasimlar bolgani uchun **xozicha** ko'cha rasimlari bilan ishlashda xatoliklar kuzatilinishi mumkin.", icon='⚠️')

st.write('---')

uploaded = st.file_uploader('Iltimos rasim yuklang:', type=['jpg', 'jpeg', 'png'])

with st.expander("Sozlamalar"):
    filter_available = st.checkbox("FILTER", value=True, help="Mevani tanishdan avval, rasimda meva mavjud yoki yo'qligini aniqlash filteri.")
    facc = st.slider('Filter uchun minimal ehtimollik', 1, 99, value=85)
    macc = st.slider('Meva turi uchun minimal ehtimollik', 1, 99, value=85)


st.write('---')

if uploaded:
    with st.spinner("Loading..."):
        img = PILImage.create(uploaded)

        # filter on
        if filter_available:
            fpred, fprob_id, fprobs = filter_ml.predict(img)

            res, main_res, filt_res = st.tabs(["Natija", "Ko'rsatgichlar", "Filter ko'rsatgichlari"])
            with filt_res:
                st.header("`fruits_filter` modeli natijalari")
                st.write(f'Eng yuqori ehtimollik {fprobs[fprob_id].item()*100:.2f}% {"musbat" if int(fpred)==1 else "manfiy"}')

                ffig = px.bar(y=fprobs*100, x=filter_ml.dls.vocab, orientation='v',
                            labels={
                                "x": "Natijalar",
                                "y": "Ehtimollik %",
                            }
                        )
                st.plotly_chart(ffig)
            # filter posive
            if int(fpred)==1 and fprobs[fprob_id].item()*100>facc:
                mpred, mprob_id, mprobs = main_ml.predict(img)

                with res:
                    if mprobs[mprob_id].item()*100>macc:
                        st.header(lbs[mpred])
                        st.image(img, width=600)
                    else:
                        st.error("Meva turi uchun ehtimollik so'ralganidan past chiqdi. Yetarlicha ishonchimiz komil emas.", icon='❌')

                with main_res:
                    st.header("`fruits` modeli natijalari")
                    st.write(f'Eng yuqori ehtimollik {mprobs[mprob_id].item()*100:.2f}% {lbs[mpred]}')
                    mfig = px.bar(y=mprobs*100, x=main_ml.dls.vocab, orientation='v',
                                labels={
                                    "x": "Mevalar",
                                    "y": "Ehtimollik %",
                                }
                            )
                    st.plotly_chart(mfig)

            else:
                with res:
                    st.error("Rasmda meva aniqlanmadi", icon='❌')
                    st.image(img, width=600)
                with main_res:
                    st.error("Rasmda meva aniqlanmadi. Filter natijalarini ko'ring", icon='❌')
        else:
            res, main_res = st.tabs(["Natija", "Ko'rsatgichlar"])
            mpred, mprob_id, mprobs = main_ml.predict(img)

            with res:
                if mprobs[mprob_id].item()*100>macc:
                    st.header(lbs[mpred])
                    st.image(img, width=600)
                else:
                    st.error("Meva turi uchun ehtimollik so'ralganidan past chiqdi. Yetarlicha ishonchimiz komil emas.", icon='❌')

            with main_res:
                st.header("`fruits modeli` natijalari")
                st.write(f'Eng yuqori ehtimollik {mprobs[mprob_id].item()*100:.2f}% {lbs[mpred]}')
                mfig = px.bar(y=mprobs*100, x=main_ml.dls.vocab, orientation='v',
                            labels={
                                "x": "Mevalar",
                                "y": "Ehtimollik %",
                            }
                        )
                st.plotly_chart(mfig)