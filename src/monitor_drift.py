
from scipy.stats import ks_2samp
from sklearn.datasets import load_iris
from evidently import Report
from evidently.presets import DataDriftPreset


def check_data_drift():

    iris = load_iris(as_frame=True)

    reference = iris.frame.sample(n=100, random_state=42)

    current = iris.frame.sample(n=100, random_state=99)

    # Simulate drift in two features
    current['sepal length (cm)'] = (
        current['sepal length (cm)'] * 1.5
    )

    current['petal length (cm)'] = (
        current['petal length (cm)'] * 1.5
    )

    try:
        report = Report(metrics=[DataDriftPreset()])

        snapshot = report.run(
            reference_data=reference,
            current_data=current
        )

        snapshot.save_html("drift_report.html")

        print("HTML drift report generated successfully.")

    except Exception as e:
        print(f"HTML Report rendering skipped: {e}")

    drifted_columns = 0
    feature_cols = iris.feature_names

    for col in feature_cols:

        p_val = ks_2samp(
            reference[col],
            current[col]
        ).pvalue

        print(f"{col} p-value: {p_val:.4f}")

        if p_val < 0.05:
            drifted_columns += 1

    dataset_drift = drifted_columns >= (
        len(feature_cols) / 2
    )

    print(f"Drifted features: {drifted_columns}/{len(feature_cols)}")
    print(f"Dataset Drift Detected: {dataset_drift}")

    return dataset_drift


if __name__ == "__main__":
    check_data_drift()
