import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Đọc dữ liệu từ tệp CSV
train_df = pd.read_csv("/train.csv")

# Hàm xử lý dữ liệu
def manipulate_df(df):
    df['Sex'] = df['Sex'].map(lambda x: 0 if x == 'male' else 1)
    df['Age'].fillna(value=df['Age'].mean(), inplace=True)
    df['FirstClass'] = df['Pclass'].map(lambda x: 1 if x == 1 else 0)
    df['SecondClass'] = df['Pclass'].map(lambda x: 1 if x == 2 else 0)
    df['ThirdClass'] = df['Pclass'].map(lambda x: 1 if x == 3 else 0)
    df = df[['Sex', 'Age', 'FirstClass', 'SecondClass', 'ThirdClass', 'Survived']]
    return df

# Xử lý dữ liệu
train_df = manipulate_df(train_df)
features = train_df[['Sex', 'Age', 'FirstClass', 'SecondClass', 'ThirdClass']]
survival = train_df['Survived']
X_train, X_test, y_train, y_test = train_test_split(features, survival, test_size=0.3)

scaler = StandardScaler()
train_features = scaler.fit_transform(X_train)
test_features = scaler.transform(X_test)
model = LogisticRegression()
model.fit(train_features, y_train)
train_score = model.score(train_features, y_train)
test_score = model.score(test_features, y_test)
y_predict = model.predict(test_features)

def center_text(text):
    return f"<h1 style='text-align: center;'>{text}</h1>"

# Your Streamlit app code
st.markdown(center_text("Dự đoán tỉ lệ % sống sót trong tàu Titanic"), unsafe_allow_html=True)

# Tạo một container để chứa các button
col1, col2, col3 = st.columns(3)

# Button 1
with col1:
    button_1 = st.button("Hiển thị dữ liệu")

# Button 2
with col2:
    button_2 = st.button("Mô hình")

# Button 3
with col3:
    button_3 = st.button("Hiển thị tỉ lệ sống sót")



# Button 1
if button_1:
    # Đặt biến trạng thái thành True khi nút được nhấn
    st.title("Bạn có sống sót sau thảm họa Titannic hay không?")
    st.subheader("Mô hình này sẽ dự đoán hành khách có sống sót sau thảm họa Titannic hay không")
    st.table(train_df.head(10))


if button_2:
    st.title("Hiệu xuất mô hình")
    # Tính toán confusion matrix
    confusion = confusion_matrix(y_test, y_predict)
    FN = confusion[1][0]
    TN = confusion[0][0]
    TP = confusion[1][1]
    FP = confusion[0][1]

    # Hiển thị thông tin và biểu đồ cột
    st.subheader("Train Set Score: {}".format(round(train_score, 3)))
    st.subheader("Test Set Score: {}".format(round(test_score, 3)))

    # Biểu đồ cột
    st.bar_chart({'False Negative': FN, 'True Negative': TN, 'True Positive': TP, 'False Positive': FP})




name = st.text_input("Name of Passenger ")
sex = st.selectbox("Sex", options=['Male', 'Female'])
age = st.slider("Age", 1, 100, 1)
p_class = st.selectbox("Passenger Class", options=['First Class', 'Second Class', 'Third Class'])

# Button 3
if button_3:
    sex = 0 if sex == 'Male' else 1
    f_class, s_class, t_class = 0, 0, 0
    if p_class == 'First Class':
        f_class = 1
    elif p_class == 'Second Class':
        s_class = 1
    else:
        t_class = 1
    input_data = scaler.transform([[sex, age, f_class, s_class, t_class]])

    # Thực hiện dự đoán
    prediction = model.predict(input_data)
    predict_probability = model.predict_proba(input_data)

    # Hiển thị kết quả dự đoán
    if prediction[0] == 1:
        st.subheader('Passenger {} sẽ sống sót với xác xuất là {}%'.format(name, round(
            predict_probability[0][1] * 100, 3)))
    else:
        st.subheader('Passenger {} sẽ không sống sót với tỉ lệ là {}%'.format(name, round(
            predict_probability[0][0] * 100, 3)))



