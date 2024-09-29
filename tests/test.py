
# Add the parent directory to the Python path
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import sparknlp

import pytest

from preprocessing import lemmatize_and_count_words, preprocess
from pyspark.sql.functions import col, lower, regexp_replace


@pytest.fixture(scope="session")
def spark():
    """Create a Spark session for the test."""
    return sparknlp.start()

@pytest.fixture
def mock_data(spark):
    """Create a mock DataFrame with dummy petition data."""
    data = [
        {"abstract": {"_value": "This is a test petition'4 for health issues and goverment"}, "label": {"_value": "Reform the health system"},"numberOfSignatures": 176},
        {"abstract": {"_value": "The hospital is full. Many patients"},"label": {"_value": "Reform the NHS"},"numberOfSignatures": 17},
        {"abstract": {"_value": "Hospitality"},"label": {"_value": "Tourism"},"numberOfSignatures": 1},
        {"abstract": {"_value": "Another ^*petition about health & hospitals."}, "label": {"_value": "This is a test label"},"numberOfSignatures": 13},
        {"abstract": {"_value": "Hospitals should have more doctors and nurses - better health."}, "label": {"_value": "This is another test label"},"numberOfSignatures": 90},
        {"abstract": {"_value": "Hospitals should have more nurses."}, "label": {"_value": "Nurses"},"numberOfSignatures": 70},
        {"abstract": {"_value": "Let's talk about hospitals"},"label": {"_value": "Test"},"numberOfSignatures": 199},
        {"abstract": {"_value": "Nursing is a profession rooted in compassion and care"}, "label": {"_value": "Goverment"},"numberOfSignatures": 17},
        {"abstract": {"_value": "Nursing is a profession improving the health and well-being of others."},"label": {"_value": "Healthcare"},"numberOfSignatures": 66},
        {"abstract": {"_value": " MP should attend attend attend attending attending attending attending all debates, not merely turn up and vote or strike pairing deals. With other commitments, a five day Commons is not workable for MPs: I suggest three full days (9am to 6pm minimum), with one or two days for Committees, leaving at least one day for constituency work."},"label": {"_value": "Healthcare"},"numberOfSignatures": 66},
    ]
    
    # Create DataFrame
    df = spark.createDataFrame(data)
    
    # Remove punctuation 'abstract._value' column
    df_cleaned = df.withColumn('cleaned_abstract', 
        lower(regexp_replace(col("abstract._value"), "[^a-zA-Z\\s]", " "))
    )

    return df_cleaned

def test_lemmatize_and_count_words(mock_data):
    """Test if the 'lemmatize_and_count_words' function correctly identifies and counts the top 20 words."""
    
    # Run the function on mock data
    df_word_counts = lemmatize_and_count_words(mock_data, 'cleaned_abstract')
    print(df_word_counts)

    # Get the 20 most common words
    top_20_words = df_word_counts.limit(20).select('word').rdd.flatMap(lambda x: x).collect()

    # Expected top words based on the mock data
    expected_top_words = ["hospital", "health", "petition", "nurse"]
    
    # Check if the top 20 words contain expected words
    for word in expected_top_words:
        assert word in top_20_words, f"Word '{word}' is not found in the top 20 words."

def test_word_count_correctness(mock_data):
    """Test if the 'lemmatize_and_count_words' function correctly counts the occurrences of top 20 words."""
    
    # Run the function to get word counts
    df_word_counts = lemmatize_and_count_words(mock_data,'cleaned_abstract')
    
    # Collect the word counts as a dictionary
    word_counts = {row['word']: row['count'] for row in df_word_counts.collect()}

    # Check if the counts for specific words are correct
    assert word_counts["hospital"] == 5, "Incorrect count for the word 'hospital'"
    assert word_counts["health"] == 4, "Incorrect count for the word 'health'"
    assert word_counts["petition"] == 2, "Incorrect count for the word 'petition'"
    assert word_counts["nurse"] == 4, "Incorrect count for the word 'nurse'"

def test_preprocess_output_column_names(mock_data):
    """"
    Test if the 'preprocess' function correctly creates DF with 21 columns, one for petition_id, and one for each top 20 words
    """

    # Preprocess the data
    df_final = preprocess(mock_data)

    # Check that 'petition_id' exists
    assert 'petition_id' in df_final.columns, "'petition_id' column is missing."

    # Get the names of top 20 words in df_final
    top_20_words_from_df = [col for col in df_final.columns if col != 'petition_id']
    print(top_20_words_from_df)

    # Ensure that the expected number of word columns is 20
    assert len(top_20_words_from_df) == 20, f"Expected 20 word columns, but found {len(top_20_words_from_df)}."

    # Check that certain column exists
    assert 'health' in df_final.columns, "'health' column is missing."
    assert 'hospital' in df_final.columns, "'hospital' column is missing."
    assert 'petition' in df_final.columns, "'petition' column is missing."