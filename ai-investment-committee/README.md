# AI Investment Committee

This project is a Streamlit web application designed for analyzing stock data using various AI agents. The application allows users to input stock tickers, run Monte Carlo simulations, and receive aggregated recommendations from different investment strategies.

## Project Structure

- **app.py**: The main application script that sets up the Streamlit web app.
- **agents/**: A directory containing various agent implementations for stock analysis:
  - **value_agent.py**: Implements value investing principles.
  - **growth_agent.py**: Implements growth investing principles.
  - **risk_agent.py**: Assesses risk associated with stocks.
  - **quant_agent.py**: Performs quantitative analysis of stocks.
  - **chairperson_agent.py**: Aggregates recommendations from all agents.
- **market_data.py**: Functions for fetching and processing market data.
- **monte_carlo.py**: Functions for running Monte Carlo simulations on stock prices.
- **requirements.txt**: Lists the Python dependencies required for the project.
- **Dockerfile**: Instructions for building a Docker image for the application.
- **.gitignore**: Specifies files and directories to be ignored by Git.

## Setup Instructions

1. **Clone the repository**:
   ```
   git clone <repository-url>
   cd ai-investment-committee
   ```

2. **Install dependencies**:
   It is recommended to use a virtual environment. You can create one using:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
   Then install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. **Run the application**:
   Start the Streamlit app with:
   ```
   streamlit run app.py
   ```

## Usage

- Enter a stock ticker symbol (e.g., AAPL for Apple) in the input field.
- Choose to analyze a single ticker or compare two tickers.
- Specify the number of Monte Carlo simulations to run.
- Click the "Analyze" button to view the results, including agent recommendations and Monte Carlo simulation outputs.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.