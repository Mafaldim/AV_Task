import argparse

import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *

from pyspark.sql.functions import explode, col, lower, regexp_replace, split, size, concat, lit, substring,md5
from pyspark.ml import Pipeline
import pyspark.sql.functions as F
from pyspark.sql.types import StructType


# Function to process petitions and count lemmatized words
def lemmatize_and_count_words(df, text_col, min_word_length=5):
    """
    Processes a column of text data in a PySpark DataFrame to extract and count the most common lemmatized words
    that meet a specified minimum word length. The function uses Spark NLP to clean, tokenize, remove stop words,
    and lemmatize the text.

    Args:
        df (pyspark.sql.dataframe.DataFrame): Input PySpark DataFrame containing the text data.
        text_col (str): Name of the column in the DataFrame containing the text to be processed.
        min_word_length (int, optional): Minimum length of words to be considered in the final word count.
                                         Defaults to 5, meaning only words with 5 or more letters will be included.

    Returns:
        pyspark.sql.dataframe.DataFrame: A DataFrame containing two columns: 
                                         - 'word': the unique lemmatized words.
                                         - 'count': the frequency of each word across all rows in the input DataFrame,
                                                    sorted in descending order by count.
    """
    

    # Define stages for Spark NLP pipeline
    document_assembler = DocumentAssembler() \
        .setInputCol(text_col) \
        .setOutputCol("document")

    # Tokenizer to split text into words
    tokenizer = Tokenizer() \
        .setInputCols(["document"]) \
        .setOutputCol("token")

    # Use Spark NLP's StopWordsCleaner to remove stop words
    stopwords_cleaner = StopWordsCleaner() \
        .setInputCols(["token"]) \
        .setOutputCol("cleanTokens") \
        .setCaseSensitive(False)

    # Lemmatizer to apply lemmatization using a pretrained model
    lemmatizer = LemmatizerModel.pretrained("lemma_antbnc", "en")\
        .setInputCols(["cleanTokens"])\
        .setOutputCol("lemma")

    # Build the pipeline with document assembler, tokenizer, stopwords cleaner, and lemmatizer
    pipeline = Pipeline(stages=[document_assembler, tokenizer, stopwords_cleaner, lemmatizer])

    # Fit and transform the pipeline
    df_result = pipeline.fit(df).transform(df)

    # Explode the lemma column to get individual words
    df_exploded = df_result.withColumn('word', explode(col('lemma.result')))

    # Filter words with a minimum number of letters
    df_filtered = df_exploded.filter(F.length(col('word')) >= min_word_length)

    # Count the unique lemmatized words
    df_word_count = df_filtered.groupBy('word').count()

    # Return the resulting DataFrame
    return df_word_count.orderBy(col('count').desc())

def preprocess(df):
        """
        Preprocesses the input Spark DataFrame by cleaning text, lemmatizing words, counting word occurrences, 
        and generating unique petition identifiers. The function also removes duplicates, extracts 
        the 20 most common words, and adds them as columns with their word counts for each petition.

        Args:
            df (pyspark.sql.dataframe.DataFrame): Input PySpark DataFrame containing petition data with fields such as
                                                'abstract', 'label', and 'number_of_signatures'.

        Returns:
            pyspark.sql.dataframe.DataFrame: A DataFrame containing the following:
                                            - 'petition_id': Unique identifier for each petition, generated via an MD5 hash 
                                            of the concatenation of 'label', the first 100 characters of 'abstract', and 'number_of_signatures'.
                                            - 20 columns corresponding to the 20 most common lemmatized words, each containing the count of how 
                                            many times that word appears in each petition.
        """
        

        # Drop duplicate rows based on 'abstract', 'label', and 'number_of_signatures'
        df = df.dropDuplicates(['abstract', 'label', 'numberOfSignatures'])
        
        # Remove punctuation    
        df_cleaned = df.withColumn('cleaned_abstract', 
            lower(regexp_replace(col('abstract._value'), "[^a-zA-Z\\s]", ""))
        )

        # Get the 20 most common words
        df_word_counts = lemmatize_and_count_words(df_cleaned, 'cleaned_abstract')
        top_20_words = df_word_counts.limit(20).select('word').rdd.flatMap(lambda x: x).collect()

        # Count the occurrences of the top 20 words in each petition
        for word in top_20_words:
            df_cleaned = df_cleaned.withColumn(word, size(split(col("cleaned_abstract"), f"\\b{word}\\b")) - 1)
            
        # Create petition_id as a concatenation of label + first 100 characters of abstract + number_of_signatures
        df_cleaned = df_cleaned.withColumn('hash_string',
            concat(
                col('label._value'), 
                lit('_'), 
                substring(col('cleaned_abstract'), 1, 100), 
                lit('_'), 
                col('numberOfSignatures')
            )
        )
        # Generate a unique identifier using md5 hash
        df_cleaned = df_cleaned.withColumn('petition_id', md5(col('hash_string')))

        # Select only the petition_id and the 20 most common words columns
        df_final = df_cleaned.select(['petition_id'] + top_20_words)

        return df_final

if __name__ == "__main__":
    
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Preprocess JSON data.')
    parser.add_argument('json_path', type=str, help='Path to the input JSON file')
    args = parser.parse_args() # Parse the arguments

    # Get input parameters
    json_path = args.json_path  # Path to the input JSON file
    
    # Start Spark NLP session
    spark = sparknlp.start() 
    
    # Read json input file
    df = spark.read.json(json_path)
    print('Length of original DF', df.count())
    
    # Preprocess input dataframe
    df_final = preprocess(df)
    print('Length of final DF', df_final.count())


    # Save the DataFrame to CSV
    df_final.coalesce(1).write.csv('./output_data', header=True)
    print("CSV file created successfully.")