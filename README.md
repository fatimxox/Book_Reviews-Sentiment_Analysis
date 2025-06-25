
## Prerequisites

To run this project locally, you need to have Python installed on your system.

*   **Python:** Version 3.6 or higher is recommended.
*   **Python Libraries:** Install the required libraries using pip. It's highly recommended to use a virtual environment.
    ```bash
    pip install Flask scikit-learn pandas numpy nltk joblib Pillow seaborn matplotlib wordcloud
    ```
    *   `Flask`: To build the web application.
    *   `scikit-learn`: For the Logistic Regression model, TF-IDF vectorizer, train/test split, and evaluation metrics.
    *   `pandas`: For data loading and manipulation.
    *   `numpy`: For numerical operations.
    *   `nltk`: For natural language processing tasks (tokenization, stopwords).
    *   `joblib`: For saving and loading the trained model and vectorizer (an alternative to pickle, often better for large NumPy arrays).
    *   `Pillow`: Required by `wordcloud` for image handling.
    *   `seaborn`, `matplotlib`: For data visualization in the notebook.
    *   `wordcloud`: For generating word clouds in the notebook.

*   **NLTK Data:** The project requires specific NLTK data packages. The `app.py` and the notebook attempt to download these (`stopwords`, `punkt`, `punkt_tab`), but you might need to run `python -m nltk.downloader popular` or download them manually if automatic download fails.
*   **Dataset:** The `Books_rating.csv` file is required. While not included in the repository (due to size/licensing), the notebook expects to load it. You can find this dataset on platforms like Kaggle (e.g., "Amazon Books Reviews" by Chirag Aggarwal). Place the file in the project's root directory.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/fatimxox/Book_Reviews-Sentiment_Analysis.git
    cd Book_Reviews-Sentiment_Analysis
    ```

2.  **Set up a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:** With your virtual environment activated, install the required libraries:
    ```bash
    pip install -r requirements.txt # If you create a requirements.txt
    # OR manually install if you don't have requirements.txt
    pip install Flask scikit-learn pandas numpy nltk joblib Pillow seaborn matplotlib wordcloud
    ```
    *(You can generate a `requirements.txt` file after installing dependencies using `pip freeze > requirements.txt`)*

4.  **Download NLTK Data:** Run the application (`python app.py`) or the notebook once with internet access to download the necessary NLTK data (`stopwords`, `punkt`).
5.  **Obtain the dataset:** Download the `Books_rating.csv` dataset and place it in the project's root directory.
6.  **Run the Notebook:** Execute all cells in `Amazon Book Reviews - Sentiment Analysis 86%.ipynb`. This will perform the analysis, train the model and vectorizer, and save them as `sentiment_model_oversample.pkl` and `vectorizer_oversample.pkl` in the `/kaggle/working/` directory (if running on Kaggle) or the current directory (if running locally). **Ensure these `.pkl` files are saved in the same directory as `app.py`**.

## How it Works

1.  **Data Analysis & Model Training (Notebook):** The Jupyter Notebook (`Amazon Book Reviews - Sentiment Analysis 86%.ipynb`) handles the heavy lifting of data understanding, preprocessing, vectorization, model selection, training, and evaluation. It focuses on traditional NLP techniques and addresses class imbalance by oversampling the minority class. The notebook saves the trained `LogisticRegression` model and the fitted `TfidfVectorizer`.
2.  **Model Loading (Flask App):** The `app.py` script loads the pre-trained `sentiment_model_oversample.pkl` and `vectorizer_oversample.pkl` files into memory when the Flask application starts.
3.  **Text Cleaning Function:** The `clean_text` function in `app.py` replicates the preprocessing steps performed in the notebook (lowercasing, removing punctuation/numbers, stopwords, short words, and checking for Arabic). This ensures consistency between training and prediction.
4.  **Prediction Endpoint (`/predict`):**
    *   Receives review text (single review or from batch processing) via a POST request.
    *   Applies the same `clean_text` function to the input review.
    *   Uses the loaded `vectorizer` to transform the cleaned text into a TF-IDF vector.
    *   Uses the loaded `model` to predict the sentiment class ('positive' or 'negative') and the probability for each class (`predict_proba`).
    *   Calculates confidence as the probability of the predicted class.
    *   Returns the predicted sentiment, confidence, and a numerical sentiment score (probability of 'positive') as a JSON response.
5.  **Web Interface (HTML + JavaScript):**
    *   `index.html` provides the user interface with form elements and display areas.
    *   JavaScript embedded in `index.html` handles frontend interactions:
        *   Capturing user input from the textarea or file upload.
        *   Sending the review text to the Flask `/predict` endpoint using `fetch` (AJAX).
        *   Receiving the JSON response from the backend.
        *   Updating the UI to display the sentiment, confidence, emoji, and sentiment gauge pointer.
        *   Handling file drag-and-drop and selection for batch processing.
        *   Reading CSV file content and iteratively calling the `/predict` endpoint for each review in the batch.
        *   Managing analysis history using browser Local Storage.
        *   Providing export functionality for batch results.
        *   Implementing basic frontend validation and loading states.

## Model Performance (from Notebook)

The notebook `Amazon Book Reviews - Sentiment Analysis 86%.ipynb` reports the following performance metrics after training on the oversampled data and evaluating on the test set:

*   **Accuracy:** ~0.86 (86%)
*   **Classification Report:** Provides Precision, Recall, and F1-Score for both 'Negative' and 'Positive' classes, indicating reasonable performance for both classes after oversampling.
*   **Confusion Matrix:** Visualizes the number of correct and incorrect predictions for each class, showing the distribution of True Positives, True Negatives, False Positives, and False Negatives.

These metrics suggest the model is capable of classifying sentiment with decent accuracy, especially considering the original class imbalance and the use of traditional NLP techniques.

## Web Application Usage

1.  **Start the Flask server:** Follow the installation steps and run `python app.py` in your terminal. Ensure your virtual environment is activated.
2.  **Open in browser:** Navigate to `http://127.0.0.1:5000/` (or the address specified by Flask) in your web browser.
3.  **Single Review Analysis:**
    *   Stay on the "Single Review" tab (default).
    *   Paste your book review into the textarea.
    *   Observe the word count update.
    *   Click the "Analyze Sentiment" button.
    *   The results section will appear below the form, showing the predicted sentiment, confidence level, and visualizing the sentiment on a gauge.
    *   You can click "Save Analysis" to store the review and result in your local history (browser storage, capped at 10 items).
4.  **Batch Processing:**
    *   Click on the "Batch Processing" tab.
    *   Drag and drop a CSV file containing a column named "review" onto the drop zone, or click the drop zone to browse for the file.
    *   *(Note: Arabic text in batch reviews will be skipped.)*
    *   Click the "Process Batch" button (it will be enabled after you select a valid CSV).
    *   The batch results section will appear. The progress bar and count will update as reviews are processed.
    *   Results for each review will be listed, indicating sentiment or if the review was skipped/errored.
    *   You can click "Export Results" to download a CSV containing the analysis results for the entire batch.
    *   You can click "Save All to History" to add all successfully processed reviews from the batch to your local history (subject to the 10-item cap).
5.  **Analysis History:**
    *   The sidebar displays your saved analysis history from browser local storage.
    *   Click the trash icon to clear all history items.
    *   Click the redo icon next to a history item to load that review back into the single review textarea.
6.  **Example Reviews:** Use the buttons in the "Example Reviews to Try" section to quickly populate the single review textarea with pre-defined positive or negative examples.

## Future Improvements

*   **Fine-grained Sentiment:** Train a model to predict sentiment on a scale (e.g., 1-5 stars) instead of just binary.
*   **Advanced NLP Models:** Explore using more sophisticated models like recurrent neural networks (RNNs), LSTMs, GRUs, or transformer-based models (like BERT) for potentially higher accuracy, although this would require more computational resources.
*   **Handle Emojis and Slang:** Enhance the text cleaning to handle emojis, common internet slang, and more complex text patterns.
*   **Phrase Extraction:** Implement techniques to identify and display the most indicative phrases contributing to the positive or negative sentiment (e.g., using LIME or SHAP for explainability).
*   **Improve Frontend UI/UX:** Enhance the web interface design, add better loading indicators, and provide more detailed feedback during batch processing.
*   **Database Integration:** Store analysis history or batch results in a proper database instead of browser local storage for persistence across sessions and users.
*   **Asynchronous Batch Processing:** For very large CSV files, implement asynchronous processing on the server to avoid long wait times and potential timeouts.
*   **Dockerization:** Create a Dockerfile to containerize the Flask application for easier deployment.
*   **API Documentation:** Provide clear API documentation for the `/predict` endpoint if it's intended for programmatic use.

## Contributing

Contributions are welcome! If you have suggestions for improvements, bug fixes, or want to add new features, please follow the standard GitHub workflow:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/your-feature-name`).
3.  Make your changes.
4.  Commit your changes (`git commit -am 'Add some feature'`).
5.  Push to the branch (`git push origin feature/your-feature-name`).
6.  Create a new Pull Request.

Please ensure your code adheres to good practices, includes relevant documentation or comments, and passes any tests (if applicable).

## License

This project is licensed under the MIT License. See the `LICENSE` file in the repository for details.

## Contact

If you have any questions about the project, feel free to open an issue on the GitHub repository.

---
