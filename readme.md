# VOSviewer Thesaurus Generator

## Overview

This Python application generates thesaurus files for **VOSviewer**, a software tool for constructing and visualizing bibliometric networks. The thesaurus generator identifies similar terms from a list of keywords and creates a standardized mapping that can improve the quality of visualizations by reducing redundancy in term usage.

## Features

- Processes large volumes of keywords efficiently through batch processing
- Identifies similar terms using TF-IDF vectorization and cosine similarity
- Customizable similarity threshold for determining term relationships
- Memory-optimized for handling extensive keyword lists
- Generates a formatted thesaurus file compatible with VOSviewer

## Requirements

The application requires the following Python libraries:

- `pandas (>=1.5.0)`
- `sentence-transformers (>=2.2.0)`
- `scikit-learn (>=1.0.2)`
- `numpy (>=1.23.0)`
- `nltk`
- `tqdm`

## Installation

1. Clone the repository or download the source files.

2. Create a virtual environment (recommended):

   ```bash
   python -m venv vosviewer_env
   ```

3. Activate the virtual environment:

   - **Windows:**
     ```bash
     vosviewer_env\Scripts\activate
     ```
   - **macOS/Linux:**
     ```bash
     source vosviewer_env/bin/activate
     ```

4. Install required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Preparing Input Data - only if the format is not already .csv

- (Optional) Use the `tocsv.py` script to format your keywords if needed:

  ```bash
  python tocsv.py > formatted_keywords.txt
  ```

### Generating the Thesaurus

Run the main script to generate the thesaurus file:

```bash
python generator.py
```

This will:

- Read keywords from `keywords.txt`
- Process the keywords to identify similar terms
- Generate a thesaurus file named `vosviewer_thesaurus.txt` in the project directory

### Customizing Parameters

You can modify the following parameters in the main function of `generator.py`:

- `similarity_threshold`: Controls how similar terms must be to be grouped (default: `0.75`)
- `batch_size`: Adjusts the number of terms processed in each batch (default: `500`)

**Example:**

```python
generator = ThesaurusGenerator(similarity_threshold=0.8, batch_size=1000)
```

## Output Format

The generated thesaurus file contains entries in the format:

```
synonym,preferred_term
```

Where:

- `synonym` is the term to be replaced
- `preferred_term` is the standardized term that will be used in the visualization

## Importing to VOSviewer

1. Open **VOSviewer**
2. When creating a map, select the option to use a **thesaurus file**
3. Browse to and select your generated `vosviewer_thesaurus.txt` file
4. Continue with map creation as normal

## Memory Management

For very large keyword lists:

- Increase the `batch_size` if you have more RAM available
- Decrease the `batch_size` if you encounter memory issues
- The script uses garbage collection to manage memory during batch processing

## Troubleshooting

If you encounter NLTK-related errors, ensure the required NLTK data is downloaded:

```python
import nltk
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
```

For memory errors:

- Try reducing the batch size
- Ensure you have closed memory-intensive applications
