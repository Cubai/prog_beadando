# importok
import requests
from io import StringIO
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Pandas warningok letiltása oszlopműveletek eredményének visszaírásakor
pd.options.mode.copy_on_write = True


def download_csv_content(url: str) -> str:
    """
    File letöltése, hiba esetén kivételt dob
    :param url: stringként kell megadni az URL-t
    :return: stringként visszaadja az ott talált tartalmat
    """
    request_result = requests.get(url)
    status_code = request_result.status_code
    if status_code != 200:  # HTTP 200 OK
        raise FileNotFoundError(f"Hiba, a URL-ről nem elérhető az adat. Hibakód: {status_code}.")
    return request_result.text


def convert_column_to_number(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    számmá alakítja a Pandas DataFrame megadott oszlopát
    :param df: Pandas DataFrame, amit át kell alakítani ezzel a függénnyel
    :param column_name: Oszlopnév, aminek az adatait számmá kell alakítani
    :return: Pandas DataFrame az átalakított adatokkal
    """
    df[column_name] = df[column_name].str.replace('\\s+', '', regex=True)  # space-ek eltávolítása
    df[column_name] = df[column_name].str.replace(',', '.')  # vessző pontra cserélése a számokban
    try:
        df[column_name] = df[column_name].astype('int64')  # integerré konvertálás megkísérlése
    except ValueError:
        df[column_name] = df[column_name].astype(
            'float64')  # ha az integerré alakítás nem sikerül, mert . volt benne, floattá alakítás
    # más típusú kivételek nincsenek lekezelve
    return df


def cleanup(df: pd.DataFrame) -> pd.DataFrame:
    """
    Projekt függő adattisztítás
    :param df: eredeti DataFrame
    :return: tisztított DataFrame
    """
    df = df.iloc[:, :5]  # az első 5 oszlop megtartása, a többi törlése
    df.columns = ["school_year", "school_number", "classroom_number", "number_of_teachers",
                  "number_of_students"]  # az öt oszlop elnevezésének beállítása

    # a DataFrame első oszlopának eredeti adata "1990/1991" szerkezetű, ezért a / jelnél szétvágásra kerül
    # és két új oszlopba kerül year_start és year_end néven a DataFrame végére
    df[["year_start", "year_end"]] = df["school_year"].str.split("/", expand=True)

    for column_name in df.columns[1:]:  # az első oszlop kivételével, ami már nem szükséges, végigmegy az oszlopokon
        df = convert_column_to_number(df, column_name)  # és számmá konvertálja az adatokat

    return df


def set_diagram_labels(x_label="", y_label="", title="") -> None:
    """
    Beállítja a label-ket és a title-t, ha meg van adva
    """
    if title:
        plt.title(title)
    if x_label:
        plt.xlabel(x_label)
    if y_label:
        plt.ylabel(y_label)


def show_line_diagram(x_values, y_columns, colors=None, x_label="", y_label="", title="") -> None:
    """
    Vonaldiagram létrehozása és kirajzolása
    :param x_values: x tengely értékei (pl. 1991)
    :param y_columns: y tengely értékei, egyszerre több is (pl. 1991-ben hány diák volt, hány tanár volt, hány osztályterem volt)
    :param colors: opcionális lista a színek neveivel, ezek sorban használhatóak, ha a lista ki van töltve
    :param x_label: x tengely felirata
    :param y_label: y tengely felirata
    :param title: opcionálisan a diagram címsora
    """
    if colors is None:  # ha colors lista nincs kitöltve, akkor kap egy üres lista default értéket
        colors = []
    num_colors = len(colors)  # színek száma
    plt.figure(figsize=(10, 5))  # diagram létrehozása
    for idx, y_column_name in enumerate(
            y_columns):  # az y oszlopokon végig iterál és az indexeket és az oszlopneveket elkéri
        y_values = y_columns[y_column_name]  # y_values nevű változóba belerakja a tényleges értékeket
        if num_colors > 0:  # ha van színlista, akkor onnan veszi a következő színt
            plt.plot(x_values, y_values, marker='o', linestyle='-', color=colors[idx % num_colors],
                     label=y_column_name)  # kirajzolja a diagramot a következő megadott színnel
        else:
            plt.plot(x_values, y_values, marker='o', linestyle='-',
                     label=y_column_name)  # kirajzolja a diagramot a következő default színnel
    if not title and x_label and y_label:  # ha nincs megadva title, de az x és y label igen, akkor abból egy alapértelmezett title beállítása
        title = f'Vonaldiagram: {x_label} vs {y_label}'
    set_diagram_labels(x_label, y_label, title)  # label-ket beállító függvény meghívása
    plt.grid(True)  # hálós megjelenítés bekapcsolása
    plt.legend()  # jelmagyarázat beállítása, a ponthalmaz neve
    plt.show()  # megjeleníti a diagramot, és megállítja a program futását amíg az ablak bezárásra nem kerül


def show_scatter_diagram(x_values, y_values, x_label="", y_label="", title="") -> None:
    plt.figure(figsize=(10, 5))  # új diagram készítése 10 a mérete vízszintes és 5 a függőleges irányban

    # scatter plot készítése, az első két paraméter az x és a hozzá tartozó y, a marker x-ekkel jelöl, a label pedig megadja, hogy milyen címke tartozik az adott ponthalmazhoz
    plt.scatter(x_values, y_values, marker='x', label=y_label)
    if not title and x_label and y_label:  # ha nincs megadva title, de az x és y label igen, akkor abból egy alapértelmezett title beállítása
        title = f'Pontdiagram: {x_label} vs {y_label}'
    set_diagram_labels(x_label, y_label, title)  # label-ket beállító függvény meghívása
    plt.grid(True)  # hálós megjelenítés bekapcsolása
    plt.legend()  # jelmagyarázat beállítása, a ponthalmaz neve
    plt.show()  # megjeleníti a diagramot, és megállítja a program futását amíg az ablak bezárásra nem kerül


def show_mixed_diagram(x_values, y_columns, title, labels, colors) -> None:
    """
    A vonal- és pontdiagramot egyszerre rajzolja ki, de külön tengelyekkel a szemléltetés kedvéért
    :param x_values: x tengely értékei (pl. 1991)
    :param y_columns: y tengely értékei, egyszerre több is (pl. 1991-ben hány diák volt, hány tanár volt, hány osztályterem volt)
    :param title: a diagram címsora
    :param labels: tengelyek feliratának listája
    :param colors: lista a színek neveivel, ezek sorban használhatóak
    """
    num_colors = len(colors)  # színek száma
    fig, ax1 = plt.subplots()  # több diagram egy ábrán való megjelenítése
    ax2 = ax1.twinx()  # közös x tengely, külön y tengely
    axes = [ax1, ax2]  # egy listában az y tengelyek
    for ax, label, color in zip(axes, labels, colors[
                                              ::2]):  # a ciklus végigmegy a tengelyeken, a hozzá tartozó címkéken, és a színeken
        ax.set_ylabel(label, color=color)  # címke és szín beállítása, így a számok is színesek
        ax.tick_params(axis='y', labelcolor=color)  # a label színének a beállítása
    active_axis = ax1  # az aktív tengely megadása
    for idx, y_column_name in enumerate(
            y_columns):  # az y oszlopokon végig iterál és az indexeket és az oszlopneveket elkéri
        if idx >= y_columns.shape[1] // 2:  # a felénél tengelyt vált
            active_axis = ax2  # az új aktív tengely megadása
        y_values = y_columns[y_column_name]  # y_values nevű változóba belerakja a tényleges értékeket
        if idx % 2 == 0:  # a párosadik diagramok pontdiagramok
            active_axis.scatter(x_values, y_values, marker='x', color=colors[idx % num_colors],
                                label=y_column_name)  # az ábrához a pontdiagram hozzáadása
        else:  # a páratlanadik diagramok pedig a hozzájuk tartozó vonaldiagramok
            active_axis.plot(x_values, y_values, linestyle='-', color=colors[idx % num_colors],
                             label=y_column_name)  # az ábrához a vonaldiagram hozzáadása

    plt.title(title)  # cím (title) beállítása
    plt.grid(True)  # hálós megjelenítés bekapcsolása
    plt.legend()  # jelmagyarázat beállítása, a ponthalmaz neve
    plt.show()  # megjeleníti a diagramot, és megállítja a program futását amíg az ablak bezárásra nem kerül


def create_linear_model(df: pd.DataFrame, x_col: str, y_col: str) -> LinearRegression:
    """
    Lineáris regressziós modell számítása
    :param df: a megadott DataFrame
    :param x_col: x oszlop, ami alapján felállítja a modellt
    :param y_col: y oszlop, amit próbál prediktálni
    :return: a modell
    """
    # Lineáris regresszió
    X = df[[x_col]].values  # a DataFrame bemeneti adatainak elkérése
    y = df[y_col].values  # az x-ekhez tartozó értékek elkérése

    # Lineáris regresszió modell betanítása
    model = LinearRegression()  # a modell létrehozása
    model.fit(X, y)  # a modell betanítása
    return model  # a függvény visszatér a modellel.


# Menü jobb felső sarkából
# forrás: https://www.ksh.hu/stadat_files/okt/hu/okt0008.html
URL = "https://www.ksh.hu/stadat_files/okt/hu/okt0008.csv"
# URL = "http://192.168.1.2/okt0008.csv"  # fejlesztéshez, (KSH terhelés miatt)
FUTURE_YEARS = 10  # konstans a lineáris modellhez, az exrapolált évek száma

str_content = ""  # a változó inicializálása
try:
    str_content = download_csv_content(URL)  # az adat letöltése és az str_contentbe való lementése
except FileNotFoundError as e:  # hibakezelés
    print("Error: ", e)
    exit(1)

# string átkonvertálása fájllá, a separator beállítása ;-re, és mivel a csv első sora nem a headert tartalmazza, egy sort át kell ugrani, így a második sor lesz a header
data = pd.read_csv(StringIO(str_content), sep=";", header=1)
data = cleanup(data)  # az adattisztító függvény, azaz a cleanup() meghívása
print(data.head(15))  # első 15 sor kiírása, hogy látszódjon, hogy milyen adatok vannak a DataFrameben
columns_needed = ["number_of_students", "number_of_teachers", "school_number",
                  "classroom_number"]  # oszlopok kiválasztása a vonaldiagramhoz
print(data[
          columns_needed].describe())  # statisztika készítése a kiválasztott oszlopokról (átlag, min, max, medián, 25 és 75%-os percentilis), szórás
show_line_diagram(data["year_start"], data[columns_needed], x_label="Tanév kezdete",
                  y_label="Értékek")  # vonaldiagram megjelenítése
show_scatter_diagram(data["number_of_students"], data["number_of_teachers"], "Diákok",
                     "Tanárok")  # összefüggés a diákok és a tanárok száma között pontdiagramon
show_scatter_diagram(data["school_number"], data["number_of_teachers"], "Iskolák",
                     "Tanárok")  # összefüggés az iskolák és a tanárok száma között pontdiagramon
show_scatter_diagram(data["school_number"], data["number_of_students"], "Iskolák",
                     "Diákok")  # összefüggés az iskolák és a diákok száma között pontdiagramon

all_years_with_future = list(range(min(data["year_start"]), max(
    data["year_start"]) + FUTURE_YEARS))  # a múltra és a jövőre vonatkozó évek listája

student_model = create_linear_model(data, "year_start",
                                    "number_of_students")  # egy egyenes illesztése arra, hogy melyik évben hány diák tanult
student_predicted = student_model.predict(
    [[xx] for xx in all_years_with_future])  # a modell szerint melyik évben hány diák tanult
student_real_and_predicted = list(data["number_of_students"])  # múltbeli valós adatok változóba mentése
student_real_and_predicted += list(student_predicted[
                                   len(student_real_and_predicted):])  # összefűzi a múltbeli valós adatokat a jövőre vonatkozó prediktált adatokkal

teacher_model = create_linear_model(data, "year_start",
                                    "number_of_teachers")  # egy egyenes illesztése arra, hogy melyik évben hány tanár tanított
teacher_predicted = teacher_model.predict(
    [[xx] for xx in all_years_with_future])  # a modell szerint melyik évben hány tanár tanított
teacher_real_and_predicted = list(data["number_of_teachers"])  # múltbeli valós adatok változóba mentése
teacher_real_and_predicted += list(teacher_predicted[
                                   len(teacher_real_and_predicted):])  # összefűzi a múltbeli valós adatokat a jövőre vonatkozó prediktált adatokkal

show_scatter_diagram(data["year_start"], data["number_of_students"], "Év",
                     "Diákok")  # összefüggés az év és a diákok száma között pontdiagramon
show_scatter_diagram(data["year_start"], data["number_of_teachers"], "Év",
                     "Tanárok")  # összefüggés az év és a tanárok száma között pontdiagramon

# új DataFrame létrehozása, amiben benne vannak a már eltelt és a közeljövő évszámai, a lineáris modellek, és a valós adatok
data_pred = pd.DataFrame({"year_start": all_years_with_future,  # évszámok
                          "student_predicted": student_predicted,  # diákok számára vonatkozó lineáris modell értékei
                          "student_real_and_predicted": student_real_and_predicted,
                          # diákok valós száma, kiegészítve a jövőre vonatkozó modell adataival
                          "teacher_predicted": teacher_predicted,  # tanárok számára vonatkozó lineáris modell értékei
                          "teacher_real_and_predicted": teacher_real_and_predicted})  # tanárok valós száma, kiegészítve a jövőre vonatkozó modell adataival

# kevert diagram megjelenítése az új DataFrame-mel
# a diákok száma jelentősen meghaladja a tanárokét, emiatt a kéttengelyes megjelenítés sokkal szemléletesebb
# a diákok adatai a kék, bal tengelyhez tartoznak, a valós számuk kék X, a lineáris pedig világoskék vonal
# a tanárok a piros, jobb, tengelyhez tartoznak, valós számuk piros X, a lineáris modell narancssárga vonal
show_mixed_diagram(data_pred["year_start"], data_pred[[
    "student_real_and_predicted", "student_predicted",
    "teacher_real_and_predicted", "teacher_predicted"
]], "Diákok és tanárok száma", ["diákok", "tanárok"], colors=["blue", "lightblue", "red", "orange"])