# CNN_meva
[Hosted on streamlit](https://cnn-meva.streamlit.app/)

## Umumiy
Deap learning asosida o'qitilgan sodda modellarning streamlitda deployi uchun na'muna. Loyiha shunchaki to'liqligicha o'quv amaliyoti(selfstudy) maqsadida qilingan.

## Qo'llanma
Loyihada ikkita DL modeli train va deploy qilindi.
Yuklangan rasim foydalanuvchi ixtiyoriga qarab filter qilinadi va quyidagi obyektlardan birini rasimdan topishga xarakat qiladi:
- Olma
- Nok
- Banan
- Limon
- Pomidor
- Uzum

## Resurslar

#### Modellar
- [fruits_filter.pkl](https://drive.google.com/file/d/1hMqUjQGT_aJan4XL1Ari9z-mvYOx3DlS/view?usp=share_link)
- [fruits.pkl](https://drive.google.com/file/d/1_wfwQNWlERAXWKur-nnWpZi13KLu5nRB/view?usp=share_link)

#### Train qilingan colablar
- [fruits](https://colab.research.google.com/drive/12hZ9ZhEMYVovVDYwuDyoyAc1jwqYyZE7?usp=sharing)
- [fruits_filter](https://colab.research.google.com/drive/1jeTKtDIbsKRsQxLglRSDyAgXF5LLkpJq?usp=sharing)
    > Colablar faqatgina ishlash jarayonidagina o'zgartirilgan. Yaqin kelajakda Colabdagi shlar ketma ketligini o'zgartirmagan holda tushinarli ko'rinishda qayta yoziladi.

#### oydalanilgan datasetlar
- [MIT Indoor Scenes](https://www.kaggle.com/datasets/itsahmad/indoor-scenes-cvpr-2019)
- [Open Images Dataset V4](https://storage.googleapis.com/openimages/web/factsfigures_v4.html)


## Kutilayotgan o'zgarishlar (TODO)
Filter modelni train qilishda manfiy klass uchun ishlatilingan dataset uy ichkarisida olingan rasimlar bolgani uchun **xozicha** uydan tashqarida olingan rasimlar bilan ishlashda xatoliklar kuzatilinishi mumkin. `fruits_filter.pkl` modelini boshqa turdagi rasimlar bilan qayta train qilish lizim.