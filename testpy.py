import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from scipy import stats

df = pd.read_csv('dataframe.csv')

# page configuration
st.set_page_config(
    page_title="Student Performance Dashboard",
    page_icon="🏫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Button to hide/show sidebar
if 'sidebar_visible' not in st.session_state:
    st.session_state.sidebar_visible = True

def toggle_sidebar():
    st.session_state.sidebar_visible = not st.session_state.sidebar_visible

# Display sidebar if visible
if st.session_state.sidebar_visible:
    with st.sidebar:
        st.title('🏫 Student Performance Dashboard')

        # Sankey Diagram Section
        st.header("Sankey Diagram")
        
        # Identify categorical columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        # Create a selectbox widget "Number of categories"
        num_categories = st.selectbox(
            label="Number of categories to display",
            options=list(range(2, len(categorical_cols) + 1))
        )

        # Store selected columns
        selected_columns = []

        # Create dynamic selectboxes for each category
        for i in range(num_categories):
            available_cols = [col for col in categorical_cols if col not in selected_columns]
            selected_col = st.selectbox(
                label=f"Category {i+1}",
                options=available_cols
            )
            selected_columns.append(selected_col)

        # Radar Chart Section
        st.header("Radar Chart")
        
        # Choose two student IDs to plot radar chart
        student1 = st.selectbox('Select the first student ID', df['studentID'])
        student2 = st.selectbox('Select the second student ID', df['studentID'])

        # Histogram Chart Section
        st.header("Top 10 Highest Subject")
        subject_list = ['math score', 'reading score', 'writing score', 'science score', 'physical education score']
        selected_subject = st.selectbox('Select a subject to find top 10:', subject_list)

def groupby_df(selected_columns, df):
    dfs = [] #list to store subdataframes
    for i in range(len(selected_columns) - 1):
        grouped_df = df.groupby([selected_columns[i], selected_columns[i+1]])['studentID'].count().reset_index()
        grouped_df.columns = ['source', 'target', 'value']
        dfs.append(grouped_df)
        if i == (len(selected_columns)-2):
            final_df = grouped_df
    
    #concat all the dataframes
    overall_df = pd.concat(dfs, axis=0)
    return overall_df, final_df

grouped_df, final_df = groupby_df(selected_columns, df)

def find_unique_mapping(df1, df2):
    # Find unique values from 'source' and 'target' columns
    unique_source_target = list(pd.unique(df1[['source', 'target']].values.ravel('K')))

    # Create a mapping dictionary to map each unique value to a unique integer
    mapping_dict = {value: idx for idx, value in enumerate(unique_source_target)}

    # Map the 'source' and 'target' columns to the integers
    df1['source'] = df1['source'].map(mapping_dict)
    df1['target'] = df1['target'].map(mapping_dict)
    df2['source'] = df2['source'].map(mapping_dict)
    df2['target'] = df2['target'].map(mapping_dict)
   
    # Convert the DataFrame to a dictionary with lists of values
    df_dict = df1.to_dict(orient='list')
    final_dict = df2.to_dict(orient='list')

    return unique_source_target, df_dict, final_dict

unique_source_target, df_dict, final_dict = find_unique_mapping(grouped_df, final_df)

# Color list for the nodes (sources)
color_list = [
     "rgba(31, 119, 180, 0.8)",
     "rgba(255, 127, 14, 0.8)",
     "rgba(44, 160, 44, 0.8)",
     "rgba(214, 39, 40, 0.8)",
     "rgba(148, 103, 189, 0.8)",
     "rgba(140, 86, 75, 0.8)",
     "rgba(227, 119, 194, 0.8)",
     "rgba(127, 127, 127, 0.8)",
     "rgba(188, 189, 34, 0.8)",
     "rgba(23, 190, 207, 0.8)",
     "rgba(31, 119, 180, 0.8)",
     "rgba(255, 127, 14, 0.8)",
     "rgba(44, 160, 44, 0.8)",
     "rgba(214, 39, 40, 0.8)",
     "rgba(148, 103, 189, 0.8)",
     "rgba(140, 86, 75, 0.8)",
     "rgba(227, 119, 194, 0.8)",
     "rgba(127, 127, 127, 0.8)",
     "rgba(188, 189, 34, 0.8)",
     "rgba(23, 190, 207, 0.8)",
     "rgba(31, 119, 180, 0.8)",
     "rgba(255, 127, 14, 0.8)",
     "rgba(44, 160, 44, 0.8)",
     "rgba(214, 39, 40, 0.8)",
     "rgba(148, 103, 189, 0.8)",
     "rgba(140, 86, 75, 0.8)",
     "rgba(227, 119, 194, 0.8)",
     "rgba(127, 127, 127, 0.8)",
     "rgba(188, 189, 34, 0.8)",
     "rgba(23, 190, 207, 0.8)",
     "rgba(31, 119, 180, 0.8)",
     "rgba(255, 127, 14, 0.8)",
     "rgba(44, 160, 44, 0.8)",
     "rgba(214, 39, 40, 0.8)",
     "rgba(148, 103, 189, 0.8)"]

def lighten_color(color_str, alpha=0.2):
    rgba_values = color_str.strip("rgba()").split(",")
    return f"rgba({rgba_values[0]}, {rgba_values[1]}, {rgba_values[2]}, {alpha})"

def calculate_total_values_by_source(df_dict):
    # Convert df_dict into a Pandas DataFrame for easy grouping
    df = pd.DataFrame(df_dict)
    
    # Group by 'source' and sum the 'value' for each unique source
    total_values_by_source = df.groupby('source')['value'].sum().to_dict()
    
    return total_values_by_source

def calculate_total_values_by_target(final_df):
    df = pd.DataFrame(final_df)
    total_values_by_target = df.groupby('target')['value'].sum().to_dict()
    return total_values_by_target

def draw_sankey(df_dict, final_dict, unique_source_target, selected_columns, color_list):
    # Calculate total values for each source node using Pandas
    total_values_by_source = calculate_total_values_by_source(df_dict)
    total_values_by_target = calculate_total_values_by_target(final_dict)
    labeled_nodes = []

    for i, label in enumerate(unique_source_target):
        source_value = total_values_by_source.get(i)
        target_value = total_values_by_target.get(i)
        
        if source_value is not None:
            #use the source value if it exists
            labeled_nodes.append(f"{label}: {source_value}")
        else:
            labeled_nodes.append(f"{label}: {target_value}")
    
    # Assign lighter colors to links based on the source node color
    link_colors = [lighten_color(color_list[src]) for src in df_dict['source']]
    
    # Creating the Sankey diagram with node and link colors
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labeled_nodes,  # Labels with total values for source nodes
            color=color_list[:len(unique_source_target)]  # Assign colors to nodes
        ),
        textfont=dict(
            color='black',  # Set label text color to black
            size=14,  # Font size for the labels
            weight='bold'
        ),
        link=dict(
            source=df_dict['source'],
            target=df_dict['target'],
            value=df_dict['value'],
            color=link_colors  # Lighter colors for links
        ))])

    # Dynamic title for the Sankey diagram
    title_text = "Student Performance due to " + " and ".join(selected_columns)
    fig.update_layout(title_text=title_text, font_size=24, width=3000, height=450)

    # Show the figure in Streamlit
    st.plotly_chart(fig)

def radar_chart(student1, student2):
    # Lấy dữ liệu cho học sinh 1 và học sinh 2 theo tên cột
    data_student1 = df[df['studentID'] == student1][['math score', 'reading score', 'writing score', 'science score', 'physical education score']].values.flatten()
    data_student2 = df[df['studentID'] == student2][['math score', 'reading score', 'writing score', 'science score', 'physical education score']].values.flatten()
    average_scores = df[['math score', 'reading score', 'writing score', 'science score', 'physical education score']].mean().values  # Lấy giá trị trung bình

    fig1 = go.Figure()

    # Dữ liệu học sinh 1
    fig1.add_trace(go.Scatterpolar(
        r=list(data_student1) + [data_student1[0]],  # Đóng vòng
        theta=['Math Score', 'Reading Score', 'Writing Score', 'Science Score', 'PE score', 'Math Score'],
        fill='none',
        name=f'Student {student1}',
        line_color='red'
    ))

    # Dữ liệu học sinh 2
    fig1.add_trace(go.Scatterpolar(
        r=list(data_student2) + [data_student2[0]],  # Đóng vòng
        theta=['Math Score', 'Reading Score', 'Writing Score', 'Science Score', 'PE score', 'Math Score'],
        fill='none',
        name=f'Student {student2}',
        line_color='blue'
    ))

    # Dữ liệu điểm trung bình
    fig1.add_trace(go.Scatterpolar(
        r=list(average_scores) + [average_scores[0]],  # Đóng vòng
        theta=['Math Score', 'Reading Score', 'Writing Score', 'Science Score', 'PE score', 'Math Score'],
        fill='none',
        name='Average of all students',
        line_color='green'
    ))

    # Cập nhật layout để loại bỏ các vòng tròn
    fig1.update_layout(
        polar=dict(
            radialaxis=dict(visible=False, showticklabels=False, range=[0, 100], showline=False, gridcolor='rgba(0, 0, 0, 0)'),
            angularaxis=dict(showline=False, gridcolor='rgba(0, 0, 0, 0)', tickfont=dict(size=12, color='black', weight='bold')),
            bgcolor='rgba(0,0,0,0)'  # Loại bỏ nền
        ),
        showlegend=True,
        width=1000,
        height=400,
        title=f"Comparison between Student ID {student1}, Student ID {student2}, and Average",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.1,
            xanchor="left",
            x=1
        )
    )

    return fig1

# Donut chart
def create_donut_chart():
    # Percent for academic performance 
    performance_counts = df['academic performance'].value_counts()
    performance_percentages = performance_counts / performance_counts.sum() * 100

    # Create donut chart
    fig = go.Figure(data=[go.Pie(
        labels=performance_percentages.index,
        values=performance_percentages.values,
        hole=0.4,  
        textinfo='percent',  
        textfont=dict(size=13, color='black', family='Arial', weight='bold'),
        marker=dict(colors=color_list),  # Màu sắc cho các phần
    )])

    fig.update_layout(
        title='Academic Performance Distribution',
        showlegend=True,
        width = 400,
        height = 400,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="right",
            x=1
        )
    )

    return fig

# Progess bar
def get_top_students(selected_subject):
    # Sắp xếp theo môn học được chọn và average score
    top_students = df.nlargest(15, [selected_subject, 'average score'])
    
    # Đặt lại chỉ số
    top_students.reset_index(drop=True, inplace=True)

    return top_students

# Histogram
def plot_histogram(df):
    st.title("Histogram of Scores")

    # Cho phép người dùng chọn loại điểm để hiển thị
    score_type = st.selectbox(
        "Select score type to display histogram",
        options=["math score", "reading score", "writing score", "average score"]
    )
    color_discrete_map = {'Male': 'red', 'Female': 'green'}
    # Vẽ biểu đồ histogram với plotly express
    fig = px.histogram(
        df, 
        x=score_type, 
        color="gender",  # Chọn màu theo giới tính
        color_discrete_map=color_discrete_map,
        marginal="violin",  # Hiển thị phân phối dưới dạng rug plot
        hover_data=df.columns  # Hiển thị tất cả các dữ liệu khi hover
    )

    # Thiết lập tiêu đề cho biểu đồ
    fig.update_layout(title=f"Histogram of {score_type}")

    # Hiển thị biểu đồ trên Streamlit
    st.plotly_chart(fig)

# Example usage within a Streamlit app
if __name__ == "__main__":
    st.title("Student Performance Dashboard")
    
    st.header("Sankey Diagram")
    draw_sankey(df_dict, final_dict, unique_source_target, selected_columns, color_list)

    # Using columns to split the interface for Sankey and Radar chart
    col2, col3 = st.columns((7, 3), gap='medium')

    with col2:
        
        if student1 and student2:  # Kiểm tra xem cả hai sinh viên đã được chọn
            radar_fig = radar_chart(student1, student2)
            st.plotly_chart(radar_fig)
        
        # Lọc dữ liệu của hai học sinh đã chọn
        selected_students = df[df['studentID'].isin([student1, student2])]
        selected_students = selected_students.set_index('studentID').loc[[student1, student2]].reset_index()
        
        # Hiển thị DataFrame của hai học sinh đã chọn
        st.write(f"Information of Student {student1} and Student {student2}")
        st.dataframe(selected_students)

    with col3:
        st.markdown('#### Top Student Grade')
        # Lấy top 5 sinh viên theo môn học đã chọn
        top_students = get_top_students(selected_subject)

        # Tạo DataFrame mới với giá trị đã làm tròn cho điểm số
        top_students['selected_subject'] = top_students[selected_subject]

        st.dataframe(top_students,
                    column_order=("studentID", "selected_subject"),
                    hide_index=True,
                    width=500,
                    height = 500,
                    column_config={
                        "studentID": st.column_config.TextColumn(
                            "studentID",
                        ),
                        "selected_subject": st.column_config.ProgressColumn(
                            "Score",  # Đặt tên cho cột thanh tiến độ
                            format="%.2f",  # Định dạng điểm số với 2 chữ số sau dấu phẩy
                            min_value=0,
                            max_value=100  # Giả sử điểm tối đa là 100
                        )}
                    )