from src.eda import run_eda
from src.model_training import train_all_models
from src.preprocessing import basic_cleaning, load_data


def main():
    dataset_path = "data/creditcard.csv"
    df = load_data(dataset_path)
    df = basic_cleaning(df)
    run_eda(df, output_dir="reports")
    train_all_models(dataset_path=dataset_path)
    print("\nPipeline finished. Check reports/, models/, and run Streamlit app.")


if __name__ == "__main__":
    main()

