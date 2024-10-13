import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D


df = pd.read_csv('https://raw.githubusercontent.com/J1nn6o/Seminar_DS/refs/heads/main/newdataframe.csv')
# Thêm hậu tố để tránh trùng lặp giữa 'academic_performance' và 'final_academic_performance'
df['final_academic_performance'] = df['final_academic_performance'].astype(str) + ' '
# Thêm hậu tố để tránh trùng lặp giữa 'part_time_job' và 'extracurricular_activities'
df['extracurricular_activities'] = df['extracurricular_activities'].astype(str) + ' '
#Sửa tên cột
df.rename(columns={'career_aspiration': 'major'}, inplace=True)

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
        categorical_cols = df.select_dtypes(include=['object', 'category', 'boolean']).drop(columns=['first_name', 'last_name']).columns
        
        # Create a selectbox widget "Number of categories"
        num_categories = st.selectbox(
            label="Number of categories to display",
            options=list(range(2, len(categorical_cols) + 1)),
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
        student1 = st.selectbox('Select the first student ID', df['id'])
        student2 = st.selectbox('Select the second student ID', df['id'])

        # Histogram Chart Section
        st.header("Top 10 Highest Subject")
        subject_list = ['math_score', 'history_score', 'physics_score', 'chemistry_score', 'biology_score', 'english_score', 'geography_score', 'average_score']
        selected_subject = st.selectbox('Select a subject to find top 10:', subject_list)

def groupby_df(selected_columns, df):
    dfs = []  # list to store subdataframes
    final_df = None  # khởi tạo biến cho final_df
    categorical_features = ['academic_performance', 'final_academic_performance']  # danh sách các feature cần categorical

    # Định nghĩa thứ tự phân loại cho các giá trị
    performance_order_academic = ['excellent', 'good', 'average', 'below average']
    performance_order_final = ['excellent ', 'good ', 'average ', 'below average ']

    # Kiểm tra xem có categorical feature nào trong selected_columns không
    for i in range(len(selected_columns) - 1):
        # Chuyển đổi thành kiểu string
        df[selected_columns[i]] = df[selected_columns[i]].astype(str)
        df[selected_columns[i + 1]] = df[selected_columns[i + 1]].astype(str)

        # Nhóm dữ liệu theo cặp các feature
        grouped_df = df.groupby([selected_columns[i], selected_columns[i + 1]])['id'].count().reset_index()
        grouped_df.columns = ['source', 'target', 'value']

        # Kiểm tra và sắp xếp cho từng categorical feature
        if selected_columns[i] == 'academic_performance':
            grouped_df['source'] = pd.Categorical(grouped_df['source'], categories=performance_order_academic, ordered=True)

        if selected_columns[i] == 'final_academic_performance':
            grouped_df['source'] = pd.Categorical(grouped_df['source'], categories=performance_order_final, ordered=True)

        if selected_columns[i + 1] == 'academic_performance':
            grouped_df['target'] = pd.Categorical(grouped_df['target'], categories=performance_order_academic, ordered=True)

        if selected_columns[i + 1] == 'final_academic_performance':
            grouped_df['target'] = pd.Categorical(grouped_df['target'], categories=performance_order_final, ordered=True)

        # Sắp xếp theo thứ tự đã định nghĩa cho cả source và target
        grouped_df = grouped_df.sort_values(by=['source', 'target'])

        dfs.append(grouped_df)

        # Gán final_df cho cặp feature cuối cùng
        if i == (len(selected_columns) - 2):
            final_df = grouped_df

    # Gộp tất cả các DataFrame lại
    overall_df = pd.concat(dfs, axis=0)

    return overall_df, final_df

grouped_df, final_df = groupby_df(selected_columns, df)

# Ánh xạ các nodes
def find_unique_mapping(df1, df2):
    unique_source_target = list(pd.unique(df1[['source', 'target']].values.ravel('K')))
    mapping_dict = {value: idx for idx, value in enumerate(unique_source_target)}
    df1['source'] = df1['source'].map(mapping_dict)
    df1['target'] = df1['target'].map(mapping_dict)
    df2['source'] = df2['source'].map(mapping_dict)
    df2['target'] = df2['target'].map(mapping_dict)

    df_dict = df1.to_dict(orient='list')
    final_dict = df2.to_dict(orient='list')

    return unique_source_target, df_dict, final_dict

unique_source_target, df_dict, final_dict = find_unique_mapping(grouped_df, final_df)


def calculate_custom_node_positions(df_dict, unique_source_target, selected_columns):
    df = pd.DataFrame(df_dict)

    # Tính tổng giá trị cho từng node
    total_values = df.groupby('source')['value'].sum().to_dict()

    x_positions = {}
    y_positions = {}

    # Giá trị y cố định cho từng node
    fixed_y_values = {
        'gender': [0.25, 0.75],  # female, male
        'class': [0.1, 0.35, 0.6, 0.85],  # A, A1, B, C
        'academic_performance': [0.18, 0.62, 0.93, 1],  # excellent, good, average, below average
        'part_time_job': [0.4, 0.9],
        'final_academic_performance': [0.1, 0.5, 0.85, 1],
        'over_absence_days': [0.2, 0.8],
        'extracurricular_activities': [0.38, 0.9],
        'major':[0.02, 0.08, 0.13, 0.2, 0.27, 0.32, 0.38, 0.45, 0.51, 0.57, 0.63, 0.68, 0.75, 0.82, 0.88, 0.93, 1],
        'major_group':[0.1, 0.22, 0.45, 0.63, 0.7, 0.76, 0.82, 0.87, 0.93 , 1]

    }

    # Phân loại các node theo nhóm feature
    feature_groups = {
        'gender': [node for node in unique_source_target if 'female' in node or 'male' in node],
        'part_time_job': [node for node in unique_source_target if node in ['True', 'False']],
        'extracurricular_activities': [node for node in unique_source_target if node in ['True ', 'False ']],
        'class': [node for node in unique_source_target if node in ['A', 'A1', 'B', 'C']],
        'academic_performance': [node for node in unique_source_target if node in ['excellent', 'good', 'average', 'below average']],
        'over_absence_days': [node for node in unique_source_target if node in ['Yes', 'No']],  # assuming this tracks students with over absence days
        'final_academic_performance': [node for node in unique_source_target if node in ['excellent ', 'good ', 'average ', 'below average ']],
        'major': [node for node in unique_source_target if node in ['Software Engineer', 'Business Owner', 'Unknown', 'Banker', 'Lawyer', 'Accountant', 
                                                                     'Doctor', 'Real Estate Developer', 'Stock Investor', 'Construction Engineer', 
                                                                    'Artist', 'Game Developer',  'Government Officer', 'Teacher', 'Designer', 
                                                                    'Scientist','Writer']],
        'major_group': [node for node in unique_source_target if node in ['economics', 'computer science', 'unknown', 'law', 'art',
                                                                          'healthcare', 'politology', 'teaching', 'science researching', 'journalism']]
    }

    # Tính toán vị trí x và y cho từng node trong từng nhóm feature
    for feature in selected_columns:
        nodes = feature_groups.get(feature, [])
        # Gán giá trị x cho nhóm feature
        num_features = len(selected_columns)
        if num_features > 0:
            spacing = (0.9 - 0.1) / (num_features - 1) if num_features > 1 else 0
            x_position = 0.1 + (selected_columns.index(feature) * spacing)
        else:
            x_position = 0.5  # Giá trị mặc định nếu không có feature nào được chọn

        x_positions.update({node: x_position for node in nodes})

        # Kiểm tra nếu có giá trị y cố định, nếu không có, chia đều vị trí y cho các node trong feature
        if feature in fixed_y_values:
            for i, node in enumerate(nodes):
                y_positions[node] = fixed_y_values[feature][i]
        else:
            # Tính toán vị trí y mặc định (chia đều cho các node)
            num_nodes = len(nodes)
            y_step = 1 / (num_nodes + 1)  # chia đều khoảng cách giữa các node
            for i, node in enumerate(nodes):
                y_positions[node] = y_step * (i + 1)

    return x_positions, y_positions

# Dummy Data cho màu sắc
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

def ensure_color_list_length(color_list, unique_source_target_length):
    if len(color_list) < unique_source_target_length:
        color_list = (color_list * (unique_source_target_length // len(color_list) + 1))[:unique_source_target_length]
    return color_list

def calculate_total_values_by_source(df_dict):
    df = pd.DataFrame(df_dict)
    total_values_by_source = df.groupby('source')['value'].sum().to_dict()
    return total_values_by_source

def calculate_total_values_by_target(final_df):
    df = pd.DataFrame(final_df)
    total_values_by_target = df.groupby('target')['value'].sum().to_dict()
    return total_values_by_target

# Hàm vẽ Sankey với x/y vị trí đã điều chỉnh
def draw_sankey(df_dict, final_dict, unique_source_target, color_list):
    color_list = ensure_color_list_length(color_list, len(unique_source_target))

    total_values_by_source = calculate_total_values_by_source(df_dict)
    total_values_by_target = calculate_total_values_by_target(final_dict)

    total_value = sum(df_dict['value'])
    x_positions, y_positions = calculate_custom_node_positions(pd.DataFrame(df_dict), unique_source_target, selected_columns)

    # Gán các node với total values và tính vị trí
    labeled_nodes = []
    for i, label in enumerate(unique_source_target):
        source_value = total_values_by_source.get(i, 0)
        target_value = total_values_by_target.get(i, 0)
        if source_value != 0:
            labeled_nodes.append(f"{label}: {source_value}")
        else:
            labeled_nodes.append(f"{label}: {target_value}")

    link_colors = [lighten_color(color_list[src]) for src in df_dict['source']]

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labeled_nodes,
            x=[x_positions[node] for node in unique_source_target],
            y=[y_positions[node] for node in unique_source_target],
            color=color_list[:len(unique_source_target)]
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
            color=link_colors
        )
    )])

    # Dynamic title for the Sankey diagram
    title_text = "Student Performance due to " + " and ".join(selected_columns)
    fig.update_layout(title_text=title_text, font_size=24, width=4200, height=550)

    # Show the figure in Streamlit
    st.plotly_chart(fig)

def radar_factory(num_vars, frame='circle'):
    """Create a radar chart with `num_vars` axes."""
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):
        def transform_path_non_affine(self, path):
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):
        name = 'radar'
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, color='black', **kwargs):
            """Vẽ các đường nối từ tâm đến các đỉnh."""
            lines = super().plot(*args, color=color, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)
            self.set_yticks([])  # Bỏ trục số
            # Điều chỉnh khoảng cách cho nhãn
            for i, label in enumerate(self.get_xticklabels()):
                angle_rad = np.radians(theta[i])
                distance = - 0.7  # Khoảng cách mặc định (có thể điều chỉnh theo ý muốn)

                x = np.cos(angle_rad) * distance
                y = np.sin(angle_rad) * distance
                label.set_position((x, y))

        def _gen_axes_patch(self):
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars, radius=0.5, edgecolor="k")
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

        def draw(self, renderer):
            super().draw(renderer)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5) + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta


# Hàm biểu đồ radar
def radar_chart(student1, student2):
    data_student1 = df[df['id'] == student1][['math_score', 'history_score', 'physics_score', 'chemistry_score', 'biology_score', 'english_score', 'geography_score']].values.flatten()
    data_student2 = df[df['id'] == student2][['math_score', 'history_score', 'physics_score', 'chemistry_score', 'biology_score', 'english_score', 'geography_score']].values.flatten()

    categories = ['Math', 'History', 'Physics', 'Chemistry', 'Biology', 'English', 'Geography']
    theta = radar_factory(len(categories), frame='polygon')

    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(projection='radar'))
    # Kiểm tra nếu hai student ID bằng nhau
    if student1 == student2:
        ax.plot(theta, data_student1, label=f'Student {student1}', color='#cf464b')  # Vẽ một đường
        ax.fill(theta, data_student1, color='#d96c70', alpha=0.1)
    else:
        ax.plot(theta, data_student1, label=f'Student {student1}', color='#cf464b')
        ax.fill(theta, data_student1, color='#d96c70', alpha=0.3)

        ax.plot(theta, data_student2, label=f'Student {student2}', color='#386b56')
        ax.fill(theta, data_student2, color='#7ce6b7', alpha=0.3)

    ax.set_varlabels(categories)
    plt.title(f"Comparison: Student {student1} vs Student {student2}")
    plt.legend(loc='lower right', bbox_to_anchor=(1.4, -0.1), fontsize='small')

    st.pyplot(fig)  # Hiển thị biểu đồ radar

# Progess bar
def get_top_students(selected_subject):
    # Sắp xếp theo môn học được chọn và average score
    top_students = df.nlargest(12, [selected_subject, 'average_score'])
    top_students.reset_index(drop=True, inplace=True)
    return top_students

# Example usage within a Streamlit app
if __name__ == "__main__":
    st.title("Student Performance Dashboard")
    
    st.header("Sankey Diagram")
    draw_sankey(df_dict, final_dict, unique_source_target, color_list)

    # Using columns to split the interface for Sankey and Radar chart
    col2, col3 = st.columns((4, 6), gap='medium')

    with col2:
        # Plot radar chart
        radar_chart(student1, student2)
        
        # Lọc dữ liệu của hai học sinh đã chọn
        selected_students = df[df['id'].isin([student1, student2])]
        selected_students = selected_students.set_index('id').loc[[student1, student2]].reset_index()
        
        

    with col3:
        st.markdown('#### Top Student Grade')
        # Lấy top 5 sinh viên theo môn học đã chọn
        top_students = get_top_students(selected_subject)

        # Tạo DataFrame mới với giá trị đã làm tròn cho điểm số
        top_students['selected_subject'] = top_students[selected_subject]

        st.dataframe(top_students,
                    column_order=("id", "selected_subject"),
                    hide_index=True,
                    width=500,
                    height = 500,
                    column_config={
                        "id": st.column_config.TextColumn(
                            "id",
                        ),
                        "selected_subject": st.column_config.ProgressColumn(
                            "Score",  # Đặt tên cho cột thanh tiến độ
                            format="%.2f",  # Định dạng điểm số với 2 chữ số sau dấu phẩy
                            min_value=0,
                            max_value=100  # Giả sử điểm tối đa là 100
                        )}
                    )
    # Hiển thị DataFrame của hai học sinh đã chọn
    st.write(f"Information of Student {student1} and Student {student2}")
    st.dataframe(selected_students)