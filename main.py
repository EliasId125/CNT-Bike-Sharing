# ============================================================
# TRABAJO - BIKE SHARING DATASET
# Aprendizaje Automático y Minería de Datos
# Tema 1 - Lectura de datos y análisis descriptivo
#
# Variable respuesta: cnt
# Dataset: UCI Bike Sharing Dataset
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# ============================================================
# CONFIGURACIÓN GENERAL DE LOS GRAFICOS
# ============================================================
# Estilo de los gráficos
sns.set_theme(style="whitegrid")
# Mostrar números con 2 decimales
pd.set_option("display.float_format", lambda x: "%.2f" % x)
# ============================================================
# CARGA DE LOS DATOS
# ============================================================
archivo = "hour.csv"
try:
    df = pd.read_csv(archivo)
    print("\n====================================================")
    print("        DATASET HA SIDO CARGADO CORRECTAMENTE")
    print("====================================================")
except FileNotFoundError:
    print("\nERROR: No se pudo encontrar el archivo 'hour.csv'.")
    print("Por favor, coloca hour.csv en la misma carpeta del programa.")
    input("\nPresiona la tecla ENTER para salir...")
    exit()
# ============================================================
# CONVERSION DE LA FECHA
# ============================================================
df["dteday"] = pd.to_datetime(df["dteday"])
# ============================================================
# MOSTRAR INFORMACIÓN GENERAL DEL DATASET
# ============================================================
print("\n====================================================")
print("             INFORMACIÓN DEL DATASET")
print("====================================================")
print("\nNúmero de filas: ", df.shape[0])
print("Número de columnas: ", df.shape[1])
print("\nNombres de las variables: ")
print(df.columns.tolist())
print("\nLas primeras 5 filas:")
print(df.head())
# ============================================================
#  MOSTRAR INFORMACIÓN DE LOS TIPOS DE DATOS
# ============================================================
print("\n====================================================")
print("                TIPOS DE DATOS")
print("====================================================")
print(df.dtypes)
# ============================================================
# MOSTRAR VALORES FALTANTES
# ============================================================
print("\n====================================================")
print("             VALORES FALTANTES")
print("====================================================")
missing = df.isnull().sum()
print(missing)
total_missing = missing.sum()
print("\nEste es el total de valores faltantes: ", total_missing)
if total_missing == 0:
    print("No se encuentran valores faltantes en el dataset.")
else:
    print("Existen algunos valores faltantes y deben ser tratados.")
# ============================================================
# DATOS DE ESTADÍSTICA DESCRIPTIVA
# ============================================================
print("\n====================================================")
print("          ESTADÍSTICA DESCRIPTIVA")
print("====================================================")
print(df.describe())
# ============================================================
# MOSTRAR INFORMACIÓN ESPECÍFICA DE CNT
# ============================================================
print("\n====================================================")
print("           VARIABLE RESPUESTA: CNT")
print("====================================================")
print(df["cnt"].describe())
print("\nMedia de CNT: ", df["cnt"].mean())
print("Mediana de CNT: ", df["cnt"].median())
print("Desviación estándar: ", df["cnt"].std())
print("Valor mínimo: ", df["cnt"].min())
print("Valor máximo: ", df["cnt"].max())
# ============================================================
# COMPROBACION CNT = CASUAL + REGISTERED
# ============================================================
print("\n====================================================")
print("       COMPROBACIÓN DE LA VARIABLE CNT")
print("====================================================")

comprobacion = (
    df["cnt"] ==
    df["casual"] + df["registered"]
).all()
print("¿cnt = casual + registered?", comprobacion)
if comprobacion:
    print(
        "\nIMPORTANTE: casual y registered no deben "
        "utilizarse como predictores de cnt."
    )
    print(
        "Utilizarlas produciría fuga de información "
        "(data leakage)."
    )
# ============================================================
# VARIABLES NUMÉRICAS PARA LA CORRELACIÓN
# ============================================================

variables_numericas = [
    "temp",
    "atemp",
    "hum",
    "windspeed",
    "cnt"
]

# ============================================================
# MATRIZ DE CORRELACIÓN
# ============================================================

correlaciones = df[variables_numericas].corr()
print("\n====================================================")
print("             MATRIZ DE CORRELACIÓN")
print("====================================================")
print(correlaciones)
print("\nCorrelación de las variables con CNT: ")
cor_cnt = (
    correlaciones["cnt"]
    .sort_values(ascending=False)
)
print(cor_cnt)

# ============================================================
# SEPARACIÓN MODELIZACIÓN / VALIDACIÓN
# ============================================================

print("\n====================================================")
print("       MODELIZACIÓN Y VALIDACIÓN")
print("====================================================")

# Se hace uso del 80% inicial para la modelización
# y el 20% final para la validación.
#
# Se mantiene el orden temporal ya que estamos
# trabajando con variables horarias.

porcentaje_modelizacion = 0.80
punto_corte = int(
    len(df) * porcentaje_modelizacion
)
datos_modelizacion = df.iloc[:punto_corte].copy()
datos_validacion = df.iloc[punto_corte:].copy()
print(
    "Registros de modelización:",
    len(datos_modelizacion)
)
print(
    "Registros de validación:",
    len(datos_validacion)
)
print(
    "\nFecha inicial de modelización:",
    datos_modelizacion["dteday"].min()
)
print(
    "Fecha final de modelización:",
    datos_modelizacion["dteday"].max()
)
print(
    "\nFecha inicial de validación:",
    datos_validacion["dteday"].min()
)
print(
    "Fecha final de validación:",
    datos_validacion["dteday"].max()
)

# ============================================================
# SACANDO ALGUNOS PROMEDIOS IMPORTANTES
# ============================================================

print("\n====================================================")
print("             PROMEDIOS DE CNT")
print("====================================================")

# Promedio por hora
promedio_hora = (
    df.groupby("hr")["cnt"]
    .mean()
)
print("\nPromedio de CNT por hora:")
print(promedio_hora)

# Promedio por estación
promedio_estacion = (
    df.groupby("season")["cnt"]
    .mean()
)
print("\nPromedio de CNT por estación:")
print(promedio_estacion)

# Promedio por situación meteorológica
promedio_clima = (
    df.groupby("weathersit")["cnt"]
    .mean()
)
print("\nPromedio de CNT por situación meteorológica:")
print(promedio_clima)

# Promedio por año
promedio_anio = (
    df.groupby("yr")["cnt"]
    .mean()
)
print("\nPromedio de CNT por año:")
print(promedio_anio)

# Promedio por día laboral
promedio_dia_laboral = (
    df.groupby("workingday")["cnt"]
    .mean()
)
print("\nPromedio de CNT según día laboral:")
print(promedio_dia_laboral)

# ============================================================
# SE DEFINEN LAS FUNCIONES PARA LOS GRÁFICOS
# ============================================================

# ------------------------------------------------------------
# EL GRÁFICO Distribución de CNT
# ------------------------------------------------------------

def grafico_distribucion_cnt():
    plt.figure(figsize=(10, 6))
    sns.histplot(
        df["cnt"],
        bins=50,
        kde=True
    )
    plt.title(
        "Distribución de la variable CNT",
        fontsize=16
    )
    plt.xlabel(
        "Número de bicicletas alquiladas (CNT)"
    )
    plt.ylabel(
        "Frecuencia"
    )
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------
# EL GRÁFICO Matriz de correlación
# ------------------------------------------------------------

def grafico_matriz_correlacion():
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        df[
            [
                "temp",
                "atemp",
                "hum",
                "windspeed",
                "cnt"
            ]
        ].corr(),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        square=True
    )
    plt.title(
        "Matriz de correlación",
        fontsize=16
    )
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------
# EL GRÁFICO Promedio de CNT por hora
# ------------------------------------------------------------

def grafico_cnt_por_hora():
    promedio = (
        df.groupby("hr")["cnt"]
        .mean()
    )
    plt.figure(figsize=(11, 6))
    plt.plot(
        promedio.index,
        promedio.values,
        marker="o",
        linewidth=2
    )
    plt.title(
        "Promedio de CNT por hora",
        fontsize=16
    )
    plt.xlabel(
        "Hora del día"
    )
    plt.ylabel(
        "Promedio de bicicletas alquiladas"
    )
    plt.xticks(
        range(24)
    )
    plt.grid(
        True,
        alpha=0.3
    )
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------
# EL GRÁFICO Temperatura frente a CNT
# ------------------------------------------------------------

def grafico_temperatura_cnt():
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df,
        x="temp",
        y="cnt",
        alpha=0.25
    )
    # Línea de tendencia
    sns.regplot(
        data=df,
        x="temp",
        y="cnt",
        scatter=False
    )
    plt.title(
        "Temperatura frente a CNT",
        fontsize=16
    )
    plt.xlabel(
        "Temperatura normalizada (temp)"
    )

    plt.ylabel(
        "Número de bicicletas alquiladas (CNT)"
    )
    plt.tight_layout()
    plt.show()

# ============================================================
# EL GRAFICO DE HUMEDAD FRENTE A CNT
# ============================================================

def grafico_humedad_cnt():
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df,
        x="hum",
        y="cnt",
        alpha=0.25
    )
    sns.regplot(
        data=df,
        x="hum",
        y="cnt",
        scatter=False
    )
    plt.title(
        "Humedad frente a CNT",
        fontsize=16
    )
    plt.xlabel(
        "Humedad normalizada"
    )
    plt.ylabel(
        "Número de bicicletas alquiladas (CNT)"
    )
    plt.tight_layout()
    plt.show()

# ============================================================
# EL GRAFICO CNT SEGÚN ESTACIÓN
# ============================================================

def grafico_cnt_estacion():
    promedio = (
        df.groupby("season")["cnt"]
        .mean()
    )
    plt.figure(figsize=(8, 6))
    plt.bar(
        promedio.index.astype(str),
        promedio.values
    )
    plt.title(
        "Promedio de CNT según estación",
        fontsize=16
    )
    plt.xlabel(
        "Estación"
    )
    plt.ylabel(
        "Promedio de CNT"
    )
    plt.tight_layout()
    plt.show()

# ============================================================
# EL GRAFICO CNT SEGÚN SITUACIÓN METEOROLÓGICA
# ============================================================

def grafico_cnt_clima():
    promedio = (
        df.groupby("weathersit")["cnt"]
        .mean()
    )
    plt.figure(figsize=(8, 6))
    plt.bar(
        promedio.index.astype(str),
        promedio.values
    )
    plt.title(
        "Promedio de CNT según situación meteorológica",
        fontsize=16
    )
    plt.xlabel(
        "Situación meteorológica"
    )
    plt.ylabel(
        "Promedio de CNT"
    )
    plt.tight_layout()
    plt.show()

# ============================================================
# MENÚ PARA ESCOGER EL GRÁFICO
# ============================================================

def menu_graficos():
    while True:
        print("\n")
        print("====================================================")
        print("             MENÚ DE GRÁFICOS")
        print("====================================================")
        print("1. Distribución de CNT")
        print("2. Matriz de correlación")
        print("3. Promedio de CNT por hora")
        print("4. Temperatura frente a CNT")
        print("5. Humedad frente a CNT")
        print("6. Promedio de CNT según estación")
        print("7. Promedio de CNT según clima")
        print("0. Salir")
        print("====================================================")
        opcion = input(
            "\nPor favor selecciona una opción: "
        )
        # ------------------------------------------------
        # OPCIÓN 1
        # ------------------------------------------------
        if opcion == "1":
            grafico_distribucion_cnt()
        # ------------------------------------------------
        # OPCIÓN 2
        # ------------------------------------------------
        elif opcion == "2":
            grafico_matriz_correlacion()
        # ------------------------------------------------
        # OPCIÓN 3
        # ------------------------------------------------
        elif opcion == "3":
            grafico_cnt_por_hora()
        # ------------------------------------------------
        # OPCIÓN 4
        # ------------------------------------------------
        elif opcion == "4":
            grafico_temperatura_cnt()
        # ------------------------------------------------
        # OPCIÓN 5
        # ------------------------------------------------
        elif opcion == "5":
            grafico_humedad_cnt()
        # ------------------------------------------------
        # OPCIÓN 6
        # ------------------------------------------------
        elif opcion == "6":
            grafico_cnt_estacion()
        # ------------------------------------------------
        # OPCIÓN 7
        # ------------------------------------------------
        elif opcion == "7":
            grafico_cnt_clima()
        # ------------------------------------------------
        # OPCIÓN 0
        # ------------------------------------------------
        elif opcion == "0":
            print("\nPrograma finalizado.")
            break
        # ------------------------------------------------
        # OPCIÓN INCORRECTA
        # ------------------------------------------------
        else:
            print(
                "\nOpción no válida."
                " Por favor selecciona un número del 0 al 8."
            )
# ============================================================
# EJECUCION DEL MENÚ
# ============================================================
menu_graficos()