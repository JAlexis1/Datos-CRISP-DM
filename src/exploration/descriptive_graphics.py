import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class DescriptiveGraphics:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.df["Timestamp"] = pd.to_datetime(self.df["Timestamp"])
        self.numeric_cols = self.df.select_dtypes(include=['int64','float64']).columns

    def __target_distribution(self):
        print("Generando gráfico 1: Distribución de etiquetas")

        plt.figure(figsize=(6,4))
        self.df["Fraud_Label"].value_counts().plot(
            kind='bar',
            color=['green','red']
        )
        plt.title("Distribución de la variable objetivo (Fraude)")
        plt.xlabel("Etiqueta de fraude (0 = No fraude, 1 = Fraude)")
        plt.ylabel("Cantidad de transacciones")
        plt.xticks(rotation=0)
        plt.savefig("1_distribucion_etiquetas.png")
        plt.clf()

    def __probability_of_fraude(self):
        print("Generando gráfico 2: Probabilidad de Fraude")

        plot_df = self.df.copy()

        # Asegurar que Fraud_Label sea numérico binario
        plot_df["Fraud_Label"] = plot_df["Fraud_Label"].astype(int)

        # Crear deciles del Risk Score
        plot_df["Risk_Decile"] = pd.qcut(
            plot_df["Risk_Score"], q=10, labels=False
        ) + 1

        plt.figure(figsize=(12, 6))  # MÁS GRANDE PARA EVITAR SOLAPES

        sns.barplot(
            data=plot_df,
            x="Risk_Decile",
            y="Fraud_Label",
            hue="Risk_Decile",
            palette="Reds",
            errorbar=None,
            legend=False
        )

        plt.title(
            "Probabilidad de fraude según el nivel de riesgo (Deciles 1 a 10)",
            fontsize=16,
            pad=18,
        )

        plt.ylabel("Probabilidad de fraude", fontsize=13)
        plt.xlabel("Decil del puntaje de riesgo\n(1 = Menos riesgoso, 10 = Más riesgoso)", fontsize=13)

        plt.xticks(rotation=0, fontsize=11)
        plt.yticks(fontsize=11)

        ax = plt.gca()
        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f", padding=3, fontsize=11)

        plt.ylim(0, 1.1)

        plt.tight_layout()

        plt.savefig("2_barras_probabilidad_fraude.png")
        plt.clf()

    def __box_plot_transaction_amount_fraud_label(self):
        print("Generando gráfico 3: Boxplot monto vs fraude")

        plt.figure(figsize=(6,4))
        sns.boxplot(data=self.df, x="Fraud_Label", y="Transaction_Amount")
        plt.title("Monto de la transacción (USD) según etiqueta de fraude")
        plt.ylabel("Monto de la transacción (USD)")
        plt.xlabel("Fraude (0 = No fraude, 1 = Fraude)")
        plt.savefig("3_boxplot_monto_por_etiqueta.png")
        plt.clf()

    def __bar_failed_transaction(self):
        print("Generando gráfico 4: Barras de intentos fallidos")

        plt.figure(figsize=(7,4))
        sns.barplot(
            x="Failed_Transaction_Count_7d",
            y="Fraud_Label",
            hue="Failed_Transaction_Count_7d",
            data=self.df,
            palette="Reds",
            errorbar=None,
            legend=False
        )

        plt.title("Probabilidad de fraude según intentos fallidos previos")
        plt.ylabel("Probabilidad de fraude (0 a 1)")
        plt.xlabel("Transacciones fallidas en los últimos 7 días")
        plt.grid(axis="y", alpha=0.3)

        plt.savefig("4_barras_intentos_fallidos.png")
        plt.clf()

    def __heatmap(self):
        print("Generando gráfico 5: Mapa de calor de correlaciones")

        plt.figure(figsize=(18,14))
        corr = self.df[self.numeric_cols].corr()

        sns.heatmap(
            corr,
            annot=True,
            fmt=".2f",
            cmap='coolwarm',
            linewidths=.5,
            cbar_kws={"shrink": .8}
        )

        plt.title("Matriz de correlación entre variables numéricas", fontsize=16)
        plt.xticks(rotation=45, ha="right", fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        plt.tight_layout()

        plt.savefig("5_heatmap.png")
        plt.clf()

    def __scatter_amount_vs_risk(self):
        print("Generando gráfico 6: Monto vs riesgo")

        plt.figure(figsize=(8,5))
        sns.scatterplot(
            data=self.df,
            x='Transaction_Amount',
            y='Risk_Score',
            hue='Fraud_Label',
            s=20,
            palette=['green', 'red']
        )

        plt.title("Monto de la transacción (USD) vs Puntuación de riesgo")
        plt.ylabel("Puntuación de riesgo")
        plt.xlabel("Monto de la transacción (USD)")
        plt.savefig("6_monto_vs_riesgo.png")
        plt.clf()

    def save(self):
        self.__target_distribution()
        self.__probability_of_fraude()
        self.__box_plot_transaction_amount_fraud_label()
        self.__bar_failed_transaction()
        self.__heatmap()
        self.__scatter_amount_vs_risk()
        print("✔ Todos los gráficos generados correctamente")
