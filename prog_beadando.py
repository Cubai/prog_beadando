import streamlit as st
import requests
from io import StringIO
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Pandas warningok letiltása oszlopműveletek eredményének visszaírásakor
pd.options.mode.copy_on_write = True

# Streamlit oldal konfiguráció
st.set_page_config(page_title="Oktatási Statisztika Elemzés", layout="wide")

def download_csv_content(url: str) -> str:
    request_result = requests.get(url)
    status_code = request_result.status_code
    if status_code != 200:  # HTTP 200 OK
        raise FileNotFoundError(f"Hiba, a URL-ről nem elérhető az adat. Hibakód: {status_code}.")
    return request_result.text

def convert_column_to_number(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    df[column_name] = df[column_name].str.replace('\\s+', '', regex=True)
    df[column_name] = df[column_name].str.replace(',', '.')
    try:
        df[column_name] = df[column_name].astype('int64')
    except ValueError:
        df[column_name] = df[column_name].astype('float64')
    return df

def cleanup(df: pd.DataFrame) -> pd.DataFrame:
    df = df.iloc[:, :5]
    df.columns = ["school_year", "school_number", "classroom_number", "number_of_teachers", "number_of_students"]
    df[["year_start", "year_end"]] = df["school_year"].str.split("/", expand=True)
    for column_name in df.columns[1:]:
        df = convert_column_to_number(df, column_name)
    return df

def set_diagram_labels(x_label="", y_label="", title=""):
    if title:
        plt.title(title)
    if x_label:
        plt.xlabel(x_label)
    if y_label:
        plt.ylabel(y_label)

def show_line_diagram(x_values, y_columns, colors=None, x_label="", y_label="", title=""):
    if colors is None:
        colors = []
    num_colors = len(colors)
    plt.figure(figsize=(10, 5))
    for idx, y_column_name in enumerate(y_columns):
        y_values = y_columns[y_column_name]
        if num_colors > 0:
            plt.plot(x_values, y_values, marker='o', linestyle='-', color=colors[idx % num_colors], label=y_column_name)
        else:
            plt.plot(x_values, y_values, marker='o', linestyle='-', label=y_column_name)
    if not title and x_label and y_label:
        title = f'Vonaldiagram: {x_label} vs {y_label}'
    set_diagram_labels(x_label, y_label, title)
    plt.grid(True)
    plt.legend()
    st.pyplot(plt)

def show_scatter_diagram(x_values, y_values, x_label="", y_label="", title=""):
    plt.figure(figsize=(10, 5))
    plt.scatter(x_values, y_values, marker='x', label=y_label)
    if not title and x_label and y_label:
        title = f'Pontdiagram: {x_label} vs {y_label}'
    set_diagram_labels(x_label, y_label, title)
    plt.grid(True)
    plt.legend()
    st.pyplot(plt)

def show_mixed_diagram(x_values, y_columns, title, labels, colors):
    num_colors = len(colors)
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()
    axes = [ax1, ax2]
    for ax, label, color in zip(axes, labels, colors[::2]):
        ax.set_ylabel(label, color=color)
        ax.tick_params(axis='y', labelcolor=color)
    active_axis = ax1
    for idx, y_column_name in enumerate(y_columns):
        if idx >= y_columns.shape[1] // 2:
            active_axis = ax2
        y_values = y_columns[y_column_name]
        if idx % 2 == 0:
            active_axis.scatter(x_values, y_values, marker='x', color=colors[idx % num_colors], label=y_column_name)
        else:
            active_axis.plot(x_values, y_values, linestyle='-', color=colors[idx % num_colors], label=y_column_name)

    plt.title(title)
    plt.grid(True)
    plt.legend()
    st.pyplot(fig)

def create_linear_model(df: pd.DataFrame, x_col: str, y_col: str) -> LinearRegression:
    X = df[[x_col]].values
    y = df[y_col].values
    model = LinearRegression()
    model.fit(X, y)
    return model

# Fő program
URL = st.text_input("Adat URL megadása:", "https://www.ksh.hu/stadat_files/okt/hu/okt0008.csv")
FUTURE_YEARS = 10

if st.button("Adatok letöltése és feldolgozása"):
    try:
        str_content = download_csv_content(URL)
        data = pd.read_csv(StringIO(str_content), sep=";", header=1)
        data = cleanup(data)
        st.dataframe(data.head(15))
        columns_needed = ["number_of_students", "number_of_teachers", "school_number", "classroom_number"]
        st.write(data[columns_needed].describe())
        show_line_diagram(data["year_start"], data[columns_needed], x_label="Tanév kezdete", y_label="Értékek")
        show_scatter_diagram(data["number_of_students"], data["number_of_teachers"], "Diákok", "Tanárok")
        show_scatter_diagram(data["school_number"], data["number_of_teachers"], "Iskolák", "Tanárok")
        show_scatter_diagram(data["school_number"], data["number_of_students"], "Iskolák", "Diákok")

        all_years_with_future = list(range(min(data["year_start"]), max(data["year_start"]) + FUTURE_YEARS))
        student_model = create_linear_model(data, "year_start", "number_of_students")
        student_predicted = student_model.predict([[xx] for xx in all_years_with_future])
        student_real_and_predicted = list(data["number_of_students"]) + list(student_predicted[len(student_real_and_predicted):])

        teacher_model = create_linear_model(data, "year_start", "number_of_teachers")
        teacher_predicted = teacher_model.predict([[xx] for xx in all_years_with_future])
        teacher_real_and_predicted = list(data["number_of_teachers"]) + list(teacher_predicted[len(teacher_real_and_predicted):])

        data_pred = pd.DataFrame({
            "year_start": all_years_with_future,
            "student_predicted": student_predicted,
            "student_real_and_predicted": student_real_and_predicted,
            "teacher_predicted": teacher_predicted,
            "teacher_real_and_predicted": teacher_real_and_predicted
        })

        show_mixed_diagram(
            data_pred["year_start"],
            data_pred[["student_real_and_predicted", "student_predicted", "teacher_real_and_predicted", "teacher_predicted"]],
            "Diákok és tanárok száma",
            ["diákok", "tanárok"],
            colors=["blue", "lightblue", "red", "orange"]
        )
    except FileNotFoundError as e:
        st.error(f"Hiba: {e}")
