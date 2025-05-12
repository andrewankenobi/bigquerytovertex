from flask import Flask, render_template, jsonify
from google.cloud import bigquery
import vertexai # Import the SDK
from vertexai.preview.generative_models import GenerativeModel
import json
import time
from functools import wraps
from concurrent.futures import ThreadPoolExecutor # Import ThreadPoolExecutor
import os # Import os to access environment variables
import datetime # Import datetime for type checking

def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__} Took {total_time:.4f} seconds')
        return result
    return timeit_wrapper

app = Flask(__name__)

# --- Configuration --- #
# Project ID and Dataset ID are retrieved from environment variables
# GOOGLE_CLOUD_PROJECT is set automatically by App Engine.
# BIGQUERY_DATASET must be set in app.yaml or environment.
PROJECT_ID = os.environ.get('GOOGLE_CLOUD_PROJECT')
LOCATION = "us-central1"    # The location for Vertex AI, e.g., us-central1
DATASET_ID = os.environ.get('BIGQUERY_DATASET')
GEMINI_MODEL_NAME = "gemini-2.5-flash-preview-04-17" # The requested Gemini model

if not PROJECT_ID:
    print("Error: GOOGLE_CLOUD_PROJECT environment variable not set.")
    # Handle missing project ID (e.g., exit or raise exception)
if not DATASET_ID:
    print("Error: BIGQUERY_DATASET environment variable not set.")
    # Handle missing dataset ID

# Initialize Vertex AI SDK
try:
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    print(f"Vertex AI SDK initialized for project '{PROJECT_ID}' in '{LOCATION}'")
    # Initialize the Gemini Model after Vertex AI SDK is initialized
    model = GenerativeModel(GEMINI_MODEL_NAME)
    print(f"Gemini model '{GEMINI_MODEL_NAME}' initialized.")
except Exception as e:
    print(f"Error initializing Vertex AI SDK or Gemini Model: {e}")
    model = None # Ensure model is None if initialization fails

# Initialize BigQuery client
try:
    # Project ID is usually inferred automatically on GCP environments like App Engine
    client = bigquery.Client()
    # If running locally and need to specify, use: client = bigquery.Client(project=PROJECT_ID)
    print(f"BigQuery Client initialized for project '{client.project}'") # Log the resolved project ID
except Exception as e:
    print(f"Error initializing BigQuery Client: {e}")
    model = None # Set model to None so functions can check for it

# Construct the BigQuery table prefix using the environment variables
BQ_TABLE_PREFIX = f"{PROJECT_ID}.{DATASET_ID}" if PROJECT_ID and DATASET_ID else None

@app.route('/')
@timeit # Keep timing the initial page load
def index():
    # Fetch only BQ data needed for initial render (charts, downloads)
    bq_data = get_bq_data()
    # Pass BQ data directly to the template
    # Convert non-serializable types for JSON embedding in HTML later
    try:
        serializable_bq_data = json.loads(json.dumps(bq_data, default=str))
    except Exception as e:
        print(f"Error serializing BQ data for template embedding: {e}")
        serializable_bq_data = {
            'combined_forecast_data': [],
            'arima_evaluation': {},
            'contribution_insights': []
        } # Provide empty defaults on error

    return render_template('index.html', bq_data=serializable_bq_data)

# Removed the old /get_data endpoint as BQ data is now loaded directly in index
# You could repurpose it if needed for dynamic updates later

@timeit # Keep timing this core function
def get_bq_data():
    """Fetches only the BigQuery data needed for the initial page render."""
    if not BQ_TABLE_PREFIX:
        print("Error: BigQuery table prefix is not configured due to missing PROJECT_ID or DATASET_ID.")
        return {
            'combined_forecast_data': [],
            'arima_evaluation': {},
            'contribution_insights': []
        }
    # Fetch BigQuery data sequentially
    combined_forecast_data = fetch_data_from_bq('ga_combined_forecast_data')
    arima_evaluation = fetch_data_from_bq('ga_arima_evaluation')
    contribution_insights = fetch_data_from_bq('ga_contribution_insights')

    # Removed Gemini calls and ThreadPoolExecutor
    
    return {
        'combined_forecast_data': combined_forecast_data,
        'arima_evaluation': arima_evaluation,
        'contribution_insights': contribution_insights,
    }

@timeit
def fetch_data_from_bq(table_name):
    """Generic function to fetch all data from a specified BigQuery table."""
    if not BQ_TABLE_PREFIX:
        print(f"Error fetching {table_name}: BigQuery table prefix not configured.")
        return [] # Or handle error appropriately

    # Base query
    query = f"SELECT * FROM `{BQ_TABLE_PREFIX}.{table_name}`"

    # Add ORDER BY only if it's not the insights table (which has ARRAYs)
    if table_name != 'ga_contribution_insights':
        query += "\nORDER BY 1" # Usually order by the date/timestamp column if it's the first

    print(f"Executing BQ Query for {table_name}:\n{query}") # Log the query
    try:
        query_job = client.query(query)
        results = query_job.result() # Waits for the job to complete.
        # Convert RowIterator to a list of dictionaries, handling potential None values
        data = []
        for row in results:
            row_dict = {}
            for key, value in row.items():
                # Attempt to serialize complex types like timestamps if needed, otherwise use default
                try:
                    # Example: Convert Timestamp to ISO format string
                    if isinstance(value, (datetime.datetime, datetime.date, datetime.time)):
                         row_dict[key] = value.isoformat()
                    # Add other type conversions if needed (e.g., Decimal to float)
                    else:
                        row_dict[key] = value
                except Exception:
                    row_dict[key] = str(value) # Fallback to string representation
            data.append(row_dict)

        # Special handling for evaluation data as it might be a single row
        if table_name == 'ga_arima_evaluation' and len(data) == 1:
            return data[0] # Return the dict directly if only one row

        return data
    except Exception as e:
        print(f"Error fetching data from {table_name}: {e}")
        return [] # Return empty list on error

@timeit
def generate_arima_interpretation(evaluation_metrics):
    if not model:
        return "<div class='interpretation error'>Gemini model not initialized. Cannot generate interpretation.</div>"
    if not PROJECT_ID:
         return "<div class='interpretation error'>GCP Project ID not configured.</div>"
    if not evaluation_metrics:
        return "<div class='interpretation error'>Could not fetch ARIMA evaluation metrics.</div>"

    prompt = f"""
    Sei Fabio Caressa e Beppe Bergomi, commentatori sportivi italiani. Devi spiegare i risultati di un modello di previsione ARIMA per le entrate giornaliere, usando metafore calcistiche.

    Dati di valutazione del modello ARIMA:
    {json.dumps(evaluation_metrics, indent=2)}

    La tua spiegazione deve essere entusiasmante, quantitativa e usare il vostro stile iconico. Includi:
    1.  Una panoramica dei "fondamentali" del modello (RMSE, MAE, MAPE, AIC): cosa misurano in termini semplici (errori di previsione, complessità del modello)?
    2.  I "risultati specifici" per queste metriche chiave, presi dai dati sopra.
    3.  Una valutazione della "prestazione in campo" del modello basata su questi numeri. È stato un "risultato netto" o una "partita sofferta"?
    4.  Spiega la "formazione" scelta dall'AUTO_ARIMA (parametri p, d, q): cosa significa ogni parametro in termini di "memoria storica" (p), "aggressività nel seguire il trend" (d), e "correzione degli errori passati" (q)? Usa i valori specifici trovati (es., ARIMA(2,1,1)).
    5.  Commenta se sono stati rilevati elementi come "cambi di modulo" (step changes), "giocate stagionali" (seasonality), "infortuni improvvisi" (spikes/dips), o "l'effetto calendario" (holiday effect). Spiega cosa significano per le entrate.
    6.  Dai un giudizio finale: questo modello è "pronto per la Champions" o ha bisogno di "tornare ad allenarsi"?

    Usa **solo** tag HTML per la formattazione: <b> per il grassetto, <ul> e <li> per le liste puntate, <p> per i paragrafi. La risposta verrà visualizzata direttamente su un sito web. Non usare Markdown.

    Mantieni il tono entusiasta e colloquiale di una telecronaca. Usa esclamazioni! Fai sentire l'adrenalina della partita dei dati!
    """
    try:
        # Add generation_config with temperature=0.0
        generation_config = {"temperature": 0.0}
        response = model.generate_content(prompt, generation_config=generation_config)
        html_response = response.text
        return html_response
    except Exception as e:
        print(f"Error generating ARIMA interpretation with {GEMINI_MODEL_NAME}: {e}")
        return f"<div class='interpretation error'>Errore durante la generazione dell'interpretazione ARIMA ({e}).</div>"

@timeit
def generate_contribution_interpretation(insights):
    if not model:
        return "<div class='interpretation error'>Gemini model not initialized. Cannot generate interpretation.</div>"
    if not PROJECT_ID:
         return "<div class='interpretation error'>GCP Project ID not configured.</div>"
    if not insights:
        return "<div class='interpretation error'>Could not fetch contribution insights.</div>"

    prompt = f"""
    Immagina di essere Paola Cortellesi e Francesca Fagnani. Devi spiegare i risultati di un'analisi di contribuzione (ML.GET_INSIGHTS) che mostra quali fattori (categorie di dispositivi, paesi, canali di marketing) hanno influenzato di più la variazione delle entrate tra due periodi. 
    Francesca (Fagnani), con il suo stile diretto e incisivo da intervistatrice di 'Belve', mette a nudo i fatti e i numeri senza sconti. 
    Paola (Cortellesi), con la sua ironia tagliente e capacità di osservazione critica, commenta i risultati evidenziando le dinamiche sottostanti e le implicazioni, magari con un tocco di sarcasmo.

    Dati di insight sulla contribuzione:
    {json.dumps(insights, indent=2)}

    La vostra analisi deve:
    1.  **Francesca:** Iniziare spiegando l'obiettivo: qui non si fanno sconti, vogliamo capire chi o cosa ha *realmente* spostato le entrate. Chi sono i protagonisti, nel bene e nel male, di questo cambiamento? Senza giri di parole.
    2.  **Francesca:** Identificare i segmenti chiave, quelli con la 'contribution' o 'unexpected_difference' più alta (positiva o negativa). Chiamiamoli per nome e cognome (es., 'Dispositivo Mobile da Stati Uniti via Ricerca Organica'). Questi sono i fatti, nudi e crudi.
    3.  **Paola:** Commentare questi segmenti significativi. La loro performance è stata una sorpresa o una delusione rispetto all'andamento generale? Hanno mantenuto le promesse o hanno deluso le aspettative? Usare i numeri (es. 'difference', 'contribution') per sottolineare l'impatto, magari con un commento ironico sulla loro 'impresa'.
    4.  **Insieme:** Spiegare i concetti chiave con il vostro stile: 'difference' è l'impatto netto, il risultato finale del segmento; 'relative_difference' è quanto pesa percentualmente questo risultato sul totale; 'contribution' è il fattore inaspettato, la vera 'belvata' (positiva o negativa) che ha sparigliato le carte rispetto a quello che ci si aspettava.
    5.  **Paola:** Concludere con delle riflessioni finali. Cosa ci dicono questi dati? Dove bisognerebbe 'indagare' meglio o concentrare le risorse per il futuro? Qualche consiglio non richiesto, ma necessario.

    Usate **solo** tag HTML per la formattazione: <b> per il grassetto, <ul> e <li> per le liste puntate, <p> per i paragrafi. Non usare Markdown.
    Mantenete un tono che alterni l'incisività diretta di Fagnani all'ironia critica di Cortellesi.
    """
    try:
        # Add generation_config with temperature=0.0
        generation_config = {"temperature": 0.0}
        response = model.generate_content(prompt, generation_config=generation_config)
        html_response = response.text
        return html_response
    except Exception as e:
        print(f"Error generating contribution interpretation with {GEMINI_MODEL_NAME}: {e}")
        return f"<div class='interpretation error'>Errore durante la generazione dell'interpretazione dei contributi ({e}).</div>"

# --- New API Endpoints for Interpretations ---

@app.route('/get_arima_interpretation')
@timeit
def get_arima_interpretation_api():
    # Fetch the necessary data
    arima_evaluation = fetch_data_from_bq('ga_arima_evaluation')
    # Generate interpretation
    interpretation_html = generate_arima_interpretation(arima_evaluation)
    return jsonify({'html': interpretation_html})

@app.route('/get_contribution_interpretation')
@timeit
def get_contribution_interpretation_api():
    # Fetch the necessary data
    contribution_insights = fetch_data_from_bq('ga_contribution_insights')
    # Generate interpretation
    interpretation_html = generate_contribution_interpretation(contribution_insights)
    return jsonify({'html': interpretation_html})

# Need to import datetime for type checking in fetch_data_from_bq
import datetime

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080) # Make accessible on the network if needed