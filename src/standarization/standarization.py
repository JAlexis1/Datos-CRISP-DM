import pandas as pd
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns


class Standarization:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def __standardize_column_names(self):
        self.df.columns = (
            self.df.columns.str.lower()
            .str.replace(r"\s+", "_", regex=True)
            .str.replace(r"[^a-z0-9_]", "", regex=True)
            .str.replace(r"_+", "_", regex=True)
            .str.strip("_")
        )

    def __verify_dtypes(self):
        print("→ Verificando y corrigiendo tipos de datos...")

        for col in self.df.columns:
            if self.df[col].dtype == "object":
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

                if self.df[col].dtype != "object":
                    continue

            if self.df[col].dtype == "object":
                try:
                    self.df[col] = pd.to_datetime(
                        self.df[col], format="%Y-%m-%d %H:%M:%S"
                    )

                    continue
                except ValueError as e:
                    print(
                        f"Advertencia: La columna '{col}' no pudo convertirse a fecha: {e}"
                    )
                    # Si falla, simplemente pasa y la columna se queda como 'object'

        print("✔ Tipos de datos verificados.")

    def __drop_unused_columns(self):
        categorical_cols = [
            "transaction_id",
            "user_id",
            "timestamp",
            "transaction_type",
            "device_type",
            "location",
            "merchant_category",
            "card_type",
            "authentication_method",
        ]

        self.df = self.df.drop(columns=categorical_cols)

    def __detected_outliers_zscore(self):
        # Calcular z-score
        self.df["zscore_amount"] = zscore(self.df["transaction_amount"])

        # Umbral
        threshold = 3

        # Outliers detectados
        outliers = self.df[self.df["zscore_amount"].abs() > threshold]
        print(f"Cantidad de outliers con zscore: {outliers.shape[0]}")

        # Límites para winsorización
        upper_limit = self.df[self.df["zscore_amount"] <= threshold][
            "transaction_amount"
        ].max()
        lower_limit = self.df[self.df["zscore_amount"] >= -threshold][
            "transaction_amount"
        ].min()

        # Crear columna nueva winsorizada
        self.df["transaction_amount_winsorized"] = self.df["transaction_amount"].copy()

        # Aplicar winsorización
        self.df.loc[
            self.df["zscore_amount"] > threshold, "transaction_amount_winsorized"
        ] = upper_limit
        self.df.loc[
            self.df["zscore_amount"] < -threshold, "transaction_amount_winsorized"
        ] = lower_limit

        min_outlier_value = outliers["transaction_amount"].min()
        print("→ Valores considerados atípicos (Z-score > 3):")
        print(f"   - Desde: {min_outlier_value}")

    def __plot_winsorization_comparison(self):
        print("Generando gráfico comparativo: antes vs después de la winsorización")

        plt.figure(figsize=(12, 5))

        # Boxplot original
        plt.subplot(1, 2, 1)
        sns.boxplot(y=self.df["transaction_amount"])
        plt.title("Antes de Winsorización")
        plt.ylabel("Monto de la transacción (USD)")

        # Boxplot después de winsorización
        plt.subplot(1, 2, 2)
        sns.boxplot(y=self.df["transaction_amount_winsorized"])
        plt.title("Después de Winsorización")
        plt.ylabel("Monto de la transacción (USD)")

        plt.tight_layout()
        plt.savefig("boxplot_comparacion_winsorizacion.png")
        plt.clf()

    def __replace_original_amount(self):
        # Reemplaza el valor original por el winsorizado
        self.df["transaction_amount"] = self.df["transaction_amount_winsorized"]

        self.df.drop(
            columns=["transaction_amount_winsorized", "zscore_amount"], inplace=True
        )

    def run(self):
        print("=== Iniciando estandarización del dataset ===")
        self.__standardize_column_names()
        self.__drop_unused_columns()
        self.__verify_dtypes()
        self.__detected_outliers_zscore()
        self.__plot_winsorization_comparison()
        self.__replace_original_amount()

        print("=== Proceso completado ===")
        return self.df.copy()
