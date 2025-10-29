import streamlit as st
import pandas as pd
import pickle

st.set_page_config(
    page_title="企鹅分类器", # 页面标题
    page_icon="🐧", # 页面图标
    layout="wide", # 布局方式
)

with st.sidebar:
    st.image('images/rigth_logo.png', width=100)
    st.title('请选择页面')
page = st.sidebar.selectbox("请选择页面", ["简介页面", "预测分类页面"], label_visibility='collapsed')

if page == "简介页面":
    st.title("企鹅分类器🐧")
    st.header("数据集介绍")
    st.text('帕尔默群岛企鹅数据集是用于数据探索和数据可视化的一个出色的数据集，也可以作为机器学习入门练习。')
    st.text('该数据集是由 Gorman 等收集，并发布在一个名为 palmerpenguins 的 R 语言包，以对南极企鹅种类进行分析和研究。')
    st.text('该数据集记录了 344 行观测数据，包含 3 个不同物种的企鹅：阿德利企鹅、巴布亚企鹅和帽带企鹅的各种信息。')
    st.title("三种企鹅的卡通形象🐧")
    st.image('images/penguins.png')
               
elif page == "预测分类页面":
    st.header("预测企鹅分类")
    st.markdown("这个Web应用是基于帕尔默群岛企鹅数据集构建的模型。只需输入6个信息就可以预测企鹅的物种，使用下面的表单开始预测吧！")

    #该页面是 3:1:2 的列布局
    col_form, col1, col_logo = st.columns([3, 1, 2])
    with col_form:
        #运用表单和表单提交按钮
        with st.form('user_inputs'):
            island = st.selectbox('企鹅栖息的岛屿', options=['德里姆岛','德雷姆岛'])
            sex = st.selectbox('性别', options=['雄性','雌性'])
            bill_length = st.number_input('喙的长度（毫米）', min_value=0.0)
            bill_depth = st.number_input('喙的深度（毫米）', min_value=0.0)
            flipper_length = st.number_input('翅膀的长度（毫米）', min_value=0.0)
            body_mass = st.number_input('身体质量（克）', min_value=0.0)
            submitted = st.form_submit_button('预测分类')

    #初始化数据预处理格式中与岛屿相关的变量
    island_biscoe, island_dream, island_torgerson = 0, 0, 0
    #根据用户输入的岛屿数据更改对应的值
    if island == '比斯科群岛':
        island_biscoe = 1
    elif island == '德里姆岛':
        island_dream = 1
    elif island == '托尔森岛':
        island_torgerson = 1

    #初始化数据预处理格式中与性别相关的变量
    sex_male, sex_female = 0, 0
    #根据用户输入的性别数据更改对应的值
    if sex == '雄性':
        sex_male = 1
    elif sex == '雌性':
        sex_female = 1

    format_data = [bill_length, bill_depth, flipper_length, body_mass,
                   island_dream, island_torgerson, island_biscoe, sex_male,
                   sex_female]

    #使用 pickle 的 load 方法从磁盘文件反序列化加载一个之前保存的随机森林模型对象
    with open('rfc_model.pkl', 'rb') as f:
        rfc_model = pickle.load(f)
        
    with open('output_uniques.pkl', 'rb') as f:
        output_uniques_map = pickle.load(f)

    if submitted:
        format_data_df = pd.DataFrame(data=[format_data], columns=rfc_model.feature_names_in_)
        #使用模型对格式化后的数据 format_data 进行预测，返回预测的类别代码
        predict_result_code = rfc_model.predict(format_data_df)
        #将类别代码映射到具体的类别名称
        predict_result_species = output_uniques_map[predict_result_code][0]
        
        st.write('用户输入的数据是：')
        st.text([island, sex, bill_length, bill_depth, flipper_length, body_mass])

      
        st.subheader('随机森林模型')
        st.write(rfc_model)

        st.subheader('映射关系示例')
        # "1"应该对应“巴布亚企鹅”
        st.write(output_uniques_map[1])

        st.write('转换为数据预处理的格式：')
        
        format_data = [bill_length, bill_depth, flipper_length, body_mass,
                   island_dream, island_torgerson, island_biscoe, sex_male,
                   sex_female]
        st.text(format_data)

        







        st.write(f'根据您输入的数据，预测该企鹅的物种名称是：**{predict_result_species}**')

    with col_logo:
        if not submitted:
            st.image('images/rigth_logo.png', width=300)
        else:
            st.image(f'images/{predict_result_species}.png', width=300)
            
