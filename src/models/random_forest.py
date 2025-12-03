import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns


class RandomForestModel:
    def __init__(self, df: pd.DataFrame):
        self.__df = df.copy()
        self.__weights = {0: 1, 1: 5}  # Peso para clases: Legal (0), Fraude (1)

    def __define_features_and_target(self):
        features = [
            "risk_score",
            "failed_transaction_count_7d",
            "ip_address_flag",
            "previous_fraudulent_activity",
            "transaction_amount",
            "avg_transaction_amount_7d",
            "daily_transaction_count",
            "transaction_distance",
            "account_balance",
        ]

        X = self.__df[features]
        y = self.__df["fraud_label"]

        return X, y

    def __save_smote_plot(self, y_train, y_train_resampled):
        plt.figure(figsize=(12, 6))

        # --- Gráfico 1: Antes de SMOTE ---
        plt.subplot(1, 2, 1)
        ax1 = sns.countplot(
            x=y_train, hue=y_train, palette=["skyblue", "#d62728"], legend=False
        )
        plt.title(f"Antes de SMOTE\n(Total: {len(y_train)})")
        plt.xlabel("Clase (0=Legal, 1=Fraude)")
        plt.ylabel("Cantidad de Registros")

        # Poner etiquetas de cantidad sobre las barras
        for container in ax1.containers:
            ax1.bar_label(container)

        # --- Gráfico 2: Después de SMOTE ---
        plt.subplot(1, 2, 2)
        ax2 = sns.countplot(
            x=y_train_resampled,
            hue=y_train_resampled,
            palette=["skyblue", "#d62728"],
            legend=False,
        )
        plt.title(f"Después de SMOTE\n(Total: {len(y_train_resampled)})")
        plt.xlabel("Clase (0=Legal, 1=Fraude)")
        plt.ylabel("")  # Quitamos label Y para limpiar

        # Poner etiquetas
        for container in ax2.containers:
            ax2.bar_label(container)

        plt.tight_layout()
        plt.savefig("smote_comparison.png", dpi=300)
        print("Gráfico guardado exitosamente como 'smote_comparison.png'")
        plt.close()

    def __train_test_split(self, test_size: float = 0.2, random_state: int = 42):
        X, y = self.__define_features_and_target()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        print("--- CONTEO ORIGINAL (Set de Entrenamiento) ---")
        print(y_train.value_counts())

        smote = SMOTE(random_state=random_state)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

        print("\n--- CONTEO CON SMOTE (Set de Entrenamiento) ---")
        print(y_train_resampled.value_counts())

        self.__save_smote_plot(y_train, y_train_resampled)

        return X_train_resampled, X_test, y_train_resampled, y_test

    def __train_algorithm(self, X_train, y_train):
        print("\n--- INICIANDO ENTRENAMIENTO (RANDOM FOREST) ---")

        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight=self.__weights,
            n_jobs=-1,
        )

        # Entrenamiento
        model.fit(X_train, y_train)
        print("Modelo entrenado exitosamente.")

        return model

    def __print_text_metrics(self, y_test, y_pred):
        print("\n--- EVALUACIÓN DE RESULTADOS (Reporte) ---")
        print(classification_report(y_test, y_pred))

    def __save_confusion_matrix(self, y_test, y_pred):
        print("Generando Matriz de Confusión...")
        plt.figure(figsize=(8, 6))

        ConfusionMatrixDisplay.from_predictions(
            y_test,
            y_pred,
            cmap="Blues",
            display_labels=["Legal (0)", "Fraude (1)"],
            colorbar=False,
        )
        plt.title("Matriz de Confusión Final")

        filename = "confusion_matrix.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"Gráfico guardado exitosamente como '{filename}'")

        plt.close()

    def train(self):
        # Paso A: Obtener datos preparados (Split + SMOTE)
        X_train_resampled, X_test, y_train_resampled, y_test = self.__train_test_split()

        # Paso B: Entrenar el modelo
        self.__model = self.__train_algorithm(X_train_resampled, y_train_resampled)

        # Paso C: Predecir (Usando datos de prueba originales)
        y_pred = self.__model.predict(X_test)

        # Paso D: Evaluar (Texto y Gráfico)
        self.__print_text_metrics(y_test, y_pred)
        self.__save_confusion_matrix(y_test, y_pred)

        return self.__model
