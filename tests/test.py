
# Add the parent directory to the Python path
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import sparknlp

import pytest

from preprocessing import lemmatize_and_count_words

from pyspark.sql.functions import col, lower, regexp_replace


@pytest.fixture(scope="session")
def spark():
    """Create a Spark session for the test."""
    return sparknlp.start()

@pytest.fixture
def mock_data(spark):
    """Create a mock DataFrame with dummy petition data."""
    data = [
        {"abstract": {"_value": "This is a test petition'4 for health issues"}},
        {"abstract": {"_value": "The hospital is full."}},
        {"abstract": {"_value": "Hospitality"}},
        {"abstract": {"_value": "Another ^*petition about health & hospitals."}},
        {"abstract": {"_value": "Hospitals should have more doctors and nurses - better health."}},
        {"abstract": {"_value": "Hospitals should have more nurses."}},
        {"abstract": {"_value": "Let's talk about hospitals"}},
        {"abstract": {"_value": "Nursing is a profession rooted in compassion and care"}},
        {"abstract": {"_value": "Nursing is a profession improving the health and well-being of others."}},
    ]
    
    # Create DataFrame
    df = spark.createDataFrame(data)
    
    # Remove punctuation 'abstract._value' column
    df_cleaned = df.withColumn('cleaned_abstract', 
        lower(regexp_replace(col("abstract._value"), "[^a-zA-Z\\s]", " "))
    )

    return df_cleaned

def test_lemmatize_and_count_words(mock_data):
    """Test if the function correctly identifies and counts the top 20 words."""
    
    # Run the function on mock data
    df_word_counts = lemmatize_and_count_words(mock_data, 'cleaned_abstract')

    # Get the 20 most common words
    top_20_words = df_word_counts.limit(20).select('word').rdd.flatMap(lambda x: x).collect()

    # Expected top words based on the mock data
    expected_top_words = ["hospital", "health", "petition", "nurse"]
    
    # Check if the top 20 words contain expected words
    for word in expected_top_words:
        assert word in top_20_words, f"Word '{word}' is not found in the top 20 words."