import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns


class RandomForestModel:
    def __init__(self, df: pd.DataFrame):
        self.__df = df.copy()
        self.__weights = {0: 1, 1: 5}

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

        # --- Antes de SMOTE ---
        plt.subplot(1, 2, 1)
        ax1 = sns.countplot(
            x=y_train, 
            hue=y_train, 
            palette=["skyblue", "#d62728"], 
            legend=False
        )
        plt.title(f"Antes de SMOTE\n(Total: {len(y_train)})")
        plt.xlabel("Clase (0=Legal, 1=Fraude)")
        plt.ylabel("Cantidad")

        for container in ax1.containers:
            ax1.bar_label(container)

        # --- Después de SMOTE ---
        plt.subplot(1, 2, 2)
        ax2 = sns.countplot(
            x=y_train_resampled,
            hue=y_train_resampled,
            palette=["skyblue", "#d62728"],
            legend=False,
        )
        plt.title(f"Después de SMOTE\n(Total: {len(y_train_resampled)})")
        plt.xlabel("Clase (0=Legal, 1=Fraude)")
        plt.ylabel("")

        for container in ax2.containers:
            ax2.bar_label(container)

        plt.tight_layout()
        plt.savefig("smote_comparison.png", dpi=300)
        plt.close()

    def __train_test_split(self, test_size: float = 0.2, random_state: int = 42):
        X, y = self.__define_features_and_target()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        print("--- CONTEO ORIGINAL (TRAIN) ---")
        print(y_train.value_counts())

        # SMOTE
        smote = SMOTE(random_state=random_state)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

        print("\n--- CONTEO CON SMOTE (TRAIN) ---")
        print(y_train_resampled.value_counts())

        self.__save_smote_plot(y_train, y_train_resampled)

        return X_train_resampled, X_test, y_train_resampled, y_test

    def __train_algorithm(self, X_train, y_train):
        print("\n--- ENTRENANDO RANDOM FOREST ---")

        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight=self.__weights,
            n_jobs=-1,
        )

        model.fit(X_train, y_train)
        print("Modelo entrenado.")

        return model

    def __print_metrics(self, y_test, y_pred):
        print("\n--- REPORT (TEST) ---")
        print(classification_report(y_test, y_pred))

    def __save_confusion_matrix(self, y_test, y_pred):
        plt.figure(figsize=(8, 6))
        ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred, cmap="Blues",
            display_labels=["Legal (0)", "Fraude (1)"],
            colorbar=False
        )
        plt.title("Matriz de Confusión")
        plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
        plt.close()

    def __save_threshold_analysis(self, y_test, probs):
        precision, recall, thresholds = precision_recall_curve(y_test, probs)

        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, precision[:-1], label="Precisión")
        plt.plot(thresholds, recall[:-1], label="Recall")
        plt.xlabel("Umbral")
        plt.ylabel("Valor")
        plt.grid(True)
        plt.legend()
        plt.title("Precisión vs Recall según el Umbral (Threshold)")
        plt.savefig("threshold_analysis.png", dpi=300, bbox_inches="tight")
        plt.close()

    def __print_cross_validation(self, X_train, y_train):
        scores = cross_val_score(
            RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight=self.__weights,
                n_jobs=-1,
            ),
            X_train,
            y_train,
            cv=5,
            scoring="f1",
            n_jobs=-1
        )

        print("\n--- VALIDACIÓN CRUZADA (TRAIN) ---")
        print(f"F1 Promedio: {scores.mean():.4f}")
        print(f"Std: {scores.std():.4f}")

    def __evaluate_model(self, X_test, y_test):
        print("\n=== EVALUACIÓN COMPLETA ===")

        probs = self.__model.predict_proba(X_test)[:, 1]
        y_pred = self.__model.predict(X_test)

        self.__print_metrics(y_test, y_pred)
        self.__save_confusion_matrix(y_test, y_pred)
        self.__save_threshold_analysis(y_test, probs)
        self.__print_cross_validation(self.X_train_resampled, self.y_train_resampled)

    def train(self):
        X_train_resampled, X_test, y_train_resampled, y_test = self.__train_test_split()

        self.X_train_resampled = X_train_resampled
        self.y_train_resampled = y_train_resampled

        self.__model = self.__train_algorithm(X_train_resampled, y_train_resampled)

        self.__evaluate_model(X_test, y_test)

        return self.__model
